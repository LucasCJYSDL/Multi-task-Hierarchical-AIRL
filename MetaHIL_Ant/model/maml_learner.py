import copy
import numpy as np
from collections import OrderedDict
import torch
from torch.nn.utils import clip_grad_value_
from .option_policy import MAMLPolicy

def replace_grad(parameter_gradients, parameter_name):
    def replace_grad_(module):
        return parameter_gradients[parameter_name]

    return replace_grad_

class MAMLLearner(object):
    def __init__(self, config, dim_s, dim_a):
        self.dim_s = dim_s
        self.dim_a = dim_a
        self.policy = MAMLPolicy(config, dim_s, dim_a)
        self.device = torch.device(config.device)
        self.inner_train_update_lr = config.inner_train_update_lr
        self.outer_meta_update_lr = config.outer_meta_update_lr
        self.meta_update_times = config.meta_update_times
        self.meta_update_times_test = config.meta_update_times_test
        self.loss_multiplier = config.loss_multiplier
        self.clip_max = config.clip_max
        self.gradient_order = config.gradient_order

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.outer_meta_update_lr)


    def step(self, demos, is_train=True):
        update_demos = []
        validate_demos = []
        for i in range(0, len(demos), 2): # TODO: extend the number of trajs for updating
            update_demos.append(demos[i])
            validate_demos.append(demos[i+1])
        assert len(update_demos) == len(validate_demos)

        create_graph = (True if self.gradient_order == 2 else False) and is_train # essential

        task_gradients = []
        task_losses = []
        for task_idx in range(len(update_demos)):
            # we are iterating through the meta batches
            s_train, a_train = update_demos[task_idx]
            s_val, a_val = validate_demos[task_idx]

            # Create a fast model using the current meta model weights
            fast_weights = OrderedDict(self.policy.named_parameters())

            for inner_batch in range(self.meta_update_times):
                # Perform update of model weights
                pred_a = self.policy.functional_forward(s_train, fast_weights, post_update=False)
                loss = torch.mean((self.loss_multiplier * (pred_a - a_train))**2)
                gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph, allow_unused=True)

                # Update weights manually
                fast_weights = OrderedDict(
                    (name, param - self.inner_train_update_lr * grad) if grad is not None else (name, param)
                    for ((name, param), grad) in zip(fast_weights.items(), gradients))

            # Do a pass of the model on the validation data from the current task
            pred_a = self.policy.functional_forward(s_val, fast_weights, post_update=True)
            loss = torch.mean((self.loss_multiplier * (pred_a - a_val))**2)
            loss.backward(retain_graph=True)

            # Accumulate losses and gradients
            task_losses.append(loss)
            gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph, allow_unused=True)
            if not create_graph:
                named_grads = {name: g if g is not None else torch.zeros_like(param, device=self.device) for ((name, param), g) in zip(fast_weights.items(), gradients)}
            else:
                named_grads = {name: g for ((name, _), g) in zip(fast_weights.items(), gradients)}

            # if not create_graph, the gradients in named_grads will not contain grad info
            # although when create_graph, the grad for pre_head is None, the gradients for the the other parts contain
            # updating info for the pre_head, which can be achieved through the chain rule
            task_gradients.append(named_grads)

        if self.gradient_order == 1:
            if is_train:
                sum_task_gradients = {k: torch.stack([grad[k] for grad in task_gradients]).mean(dim=0)
                                      for k in task_gradients[0].keys()}
                # TODO: with or without this gradient clip
                # for key in sum_task_gradients.keys():
                #     sum_task_gradients[key] = sum_task_gradients[key].clamp(-self.clip_max, self.clip_max)

                # the gradient info for the pre head is zero since this is a first order method
                hooks = []
                for name, param in self.policy.named_parameters():
                    hooks.append(param.register_hook(replace_grad(sum_task_gradients, name)))

                self.policy.train()
                self.optimizer.zero_grad()
                # Dummy pass in order to create `loss` variable
                # Replace dummy gradients with mean task gradients using hooks
                pred_a = self.policy.forward(torch.zeros((1, self.dim_s), dtype=torch.float32).to(self.device), post_update=True)
                loss = torch.mean((self.loss_multiplier * (pred_a - torch.zeros((1, self.dim_a), dtype=torch.float32).to(self.device)))**2)
                loss.backward()
                self.optimizer.step()

                for h in hooks:
                    h.remove()

                # for name, para in self.policy.named_parameters():
                #     if name == 'pre_head.bias':
                #         print(name, para)

            return torch.stack(task_losses).mean().item()

        else:
            assert self.gradient_order == 2
            self.policy.train()
            self.optimizer.zero_grad()
            meta_batch_loss = torch.stack(task_losses).mean()

            if is_train:
                meta_batch_loss.backward() # the gradients regarding the pre_head should not be 0 or None
                # TODO: with or without this gradient clip
                clip_grad_value_(self.policy.parameters(), self.clip_max)
                self.optimizer.step()

                # for name, para in self.policy.named_parameters():
                #     if name == 'pre_head.bias':
                #         print(name, para)

            return meta_batch_loss.item()


    def eval(self, env, test_task_batch):
        all_r = []
        create_graph = False
        # print("before: ", OrderedDict(self.policy.named_parameters()))
        for task_idx in range(len(test_task_batch)):
            s_train, a_train = test_task_batch[task_idx]['demos'][0] # only one

            fast_weights = OrderedDict(self.policy.named_parameters())

            for inner_batch in range(self.meta_update_times_test):
                # Perform update of model weights
                pred_a = self.policy.functional_forward(s_train, fast_weights, post_update=False)
                loss = torch.mean((self.loss_multiplier * (pred_a - a_train)) ** 2)
                gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph,
                                                allow_unused=True)

                # Update weights manually
                fast_weights = OrderedDict(
                    (name, param - self.inner_train_update_lr * grad) if grad is not None else (name, param)
                    for ((name, param), grad) in zip(fast_weights.items(), gradients))

            r_array = []
            cur_context = test_task_batch[task_idx]['context']
            s, done = env.reset(cur_context, is_expert=False), False
            while not done:
                st = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
                at = self.policy.functional_forward(st, fast_weights, post_update=True).detach() # evaluation
                at = at.clamp(-1e6, 1e6) # for numeric safety
                s, r, done = env.step(at.cpu().squeeze(dim=0).numpy())
                r_array.append(r)
            r_array = torch.as_tensor(r_array, dtype=torch.float32, device=self.device).unsqueeze(dim=-1)
            all_r.append(r_array)

        rsums = [tr.sum().item() for tr in all_r]
        steps = [tr.size(0) for tr in all_r]
        # print("after: ", OrderedDict(self.policy.named_parameters()))
        info_dict = {"r-max": np.max(rsums), "r-min": np.min(rsums), "r-avg": np.mean(rsums),
                     "step-max": np.max(steps), "step-min": np.min(steps)}

        return info_dict