import random

import torch
import time
from copy import deepcopy
from torch.multiprocessing import Process, Pipe, Lock, Value
from model.option_policy import OptionPolicy
from model.MHA_option_policy_critic import MHAOptionPolicy
from utils.common_utils import set_seed


__all__ = ["Sampler"]

def option_loop(env, policy, is_expert, fixed, task_list=None, contain_context=False):
    with torch.no_grad():
        a_array = []
        c_array = []
        s_array = []
        r_array = []
        if task_list is not None: # when testing, we will specify the list of tasks
            context = random.choice(task_list)
        else:
            context = env.sample_context()
        cnt_dim = len(context)
        s, done = env.reset(context, is_expert=is_expert), False
        ct = torch.empty(1, 1, dtype=torch.long, device=policy.device).fill_(policy.dim_c)
        c_array.append(ct)
        while not done:
            st = torch.as_tensor(s, dtype=torch.float32, device=policy.device).unsqueeze(0)
            if not contain_context:
                st = st[:, :-cnt_dim]
            ct = policy.sample_option(st, ct, fixed=fixed).detach()
            at = policy.sample_action(st, ct, fixed=fixed).detach()
            s_array.append(st)
            c_array.append(ct)
            a_array.append(at)
            s, r, done = env.step(at.cpu().squeeze(dim=0).numpy())
            r_array.append(r)
        a_array = torch.cat(a_array, dim=0)
        c_array = torch.cat(c_array, dim=0)
        s_array = torch.cat(s_array, dim=0)
        r_array = torch.as_tensor(r_array, dtype=torch.float32, device=policy.device).unsqueeze(dim=-1)

        # print("s_dim", s_array.shape)
    return s_array, c_array, a_array, r_array

def loop(env, policy, is_expert, fixed, task_list=None, contain_context=False):
    with torch.no_grad():
        a_array = []
        s_array = []
        r_array = []
        if task_list is not None: # when testing, we will specify the list of tasks
            context = random.choice(task_list)
        else:
            context = env.sample_context()
        cnt_dim = len(context)
        s, done = env.reset(context, is_expert=is_expert), False
        # print("1: ", s)
        while not done:
            st = torch.as_tensor(s, dtype=torch.float32, device=policy.device).unsqueeze(0)
            if not contain_context:
                st = st[:, :-cnt_dim]
            at = policy.sample_action(st, fixed=fixed).detach()
            s_array.append(st)
            a_array.append(at)
            s, r, done = env.step(at.cpu().squeeze(dim=0).numpy())
            r_array.append(r)
        a_array = torch.cat(a_array, dim=0)
        s_array = torch.cat(s_array, dim=0)
        r_array = torch.as_tensor(r_array, dtype=torch.float32, device=policy.device).unsqueeze(dim=-1)
        # print(r_array.shape)

        # print("s_dim", s_array.shape)
    return s_array, a_array, r_array


def repeat_loop(env, policy, is_expert, fixed, repeat_num, contain_context):
    assert contain_context
    traj_list = []
    trans_num = 0
    with torch.no_grad():
        context = env.sample_context()
        for _ in range(repeat_num): # for each context, we repeat certain number of loops
            a_array = []
            s_array = []
            r_array = []
            s, done = env.reset(context, is_expert=is_expert), False
            while not done:
                st = torch.as_tensor(s, dtype=torch.float32, device=policy.device).unsqueeze(0)
                at = policy.sample_action(st, fixed=fixed).detach()
                s_array.append(st)
                a_array.append(at)
                s, r, done = env.step(at.cpu().squeeze(dim=0).numpy())
                r_array.append(r)
            a_array = torch.cat(a_array, dim=0)
            s_array = torch.cat(s_array, dim=0)
            r_array = torch.as_tensor(r_array, dtype=torch.float32, device=policy.device).unsqueeze(dim=-1)
            traj_list.append((s_array, a_array, r_array))
            trans_num += s_array.shape[0]

    return traj_list, trans_num

def repeat_option_loop(env, policy, is_expert, fixed, repeat_num, contain_context):
    assert contain_context
    traj_list = []
    trans_num = 0
    with torch.no_grad():
        context = env.sample_context()
        for _ in range(repeat_num): # for each context, we repeat certain number of loops
            a_array = []
            c_array = []
            s_array = []
            r_array = []
            s, done = env.reset(context, is_expert=is_expert), False
            ct = torch.empty(1, 1, dtype=torch.long, device=policy.device).fill_(policy.dim_c)
            c_array.append(ct)
            while not done:
                st = torch.as_tensor(s, dtype=torch.float32, device=policy.device).unsqueeze(0)

                ct = policy.sample_option(st, ct, fixed=fixed).detach()
                at = policy.sample_action(st, ct, fixed=fixed).detach()
                s_array.append(st)
                c_array.append(ct)
                a_array.append(at)
                s, r, done = env.step(at.cpu().squeeze(dim=0).numpy())
                r_array.append(r)
            a_array = torch.cat(a_array, dim=0)
            c_array = torch.cat(c_array, dim=0)
            s_array = torch.cat(s_array, dim=0)
            r_array = torch.as_tensor(r_array, dtype=torch.float32, device=policy.device).unsqueeze(dim=-1)
            traj_list.append((s_array, c_array, a_array, r_array))
            trans_num += s_array.shape[0]

    return traj_list, trans_num


class _SamplerCommon(object):
    def __init__(self, seed, policy):
        self.device = policy.device

    def collect(self, policy_param, n_sample, fixed=False):
        raise NotImplementedError()

    def filter_demo(self, sa_array):
        print("No filters are adopted.")
        return sa_array


class _SamplerSS(_SamplerCommon):
    def __init__(self, seed, env, policy, loop_func=None, repeat_loop_func=None, is_expert=False, task_list=None, contain_context=False, repeat_num=-1):
        super(_SamplerSS, self).__init__(seed, policy)
        self.env = deepcopy(env)
        self.env.init(display=False)
        self.policy = deepcopy(policy)
        self.loop_func = loop_func
        self.repeat_loop_func = repeat_loop_func
        self.is_expert = is_expert
        self.task_list = task_list
        self.contain_context = contain_context
        self.repeat_num = repeat_num

    def collect(self, policy_param, n_sample, fixed=False):
        self.policy.load_state_dict(policy_param)
        counter = n_sample
        rets = []
        if counter > 0:
            while counter > 0:
                if self.repeat_num < 0:
                    traj = self.loop_func(self.env, self.policy, self.is_expert, fixed=fixed, contain_context=self.contain_context)
                    rets.append(traj)
                    counter -= traj[0].size(0)
                else:
                    trajs, trans_num = self.repeat_loop_func(self.env, self.policy, self.is_expert, fixed=fixed,
                                                             repeat_num=self.repeat_num, contain_context=self.contain_context)
                    rets.extend(trajs)
                    counter -= trans_num
            if self.repeat_num > 0:
                assert len(rets) % self.repeat_num == 0

        else:
            # assert self.task_list is not None
            while counter < 0: # only used for testing, so don't need repeated sampling
                traj = self.loop_func(self.env, self.policy, self.is_expert, fixed=fixed, task_list=self.task_list, contain_context=self.contain_context)
                rets.append(traj)
                counter += 1
        return rets


def Sampler(seed, env, policy, is_expert, task_list=None, contain_context=False, repeat_num=-1) -> _SamplerCommon:

    if isinstance(policy, OptionPolicy) or isinstance(policy, MHAOptionPolicy):
        loop_func = option_loop
        repeat_loop_func = repeat_option_loop
    else:
        loop_func = loop
        repeat_loop_func = repeat_loop

    class_m = _SamplerSS
    return class_m(seed, env, policy, loop_func, repeat_loop_func, is_expert, task_list, contain_context, repeat_num)


if __name__ == "__main__":
    from torch.multiprocessing import set_start_method
    set_start_method("spawn")