import torch
import torch.nn.functional as F
from .option_policy import Policy
from .option_discriminator import Discriminator, StateOnlyDiscriminator
from .context_net import ContextPosterior
from utils.config import Config
from utils.model_util import clip_grad_norm_

class PEMIRL_AIRL(torch.nn.Module):
    def __init__(self, config: Config, dim_s, dim_a, dim_cnt, cnt_limit):
        super(PEMIRL_AIRL, self).__init__()
        self.dim_a = dim_a
        self.dim_s = dim_s
        self.dim_cnt = dim_cnt
        self.cnt_limit = cnt_limit
        self.info_coeff = config.info_coeff
        self.cnt_sampling_fixed = config.cnt_sampling_fixed
        self.cnt_training_iters = config.cnt_training_iterations
        self.cnt_starting_iter = config.cnt_starting_iter
        self.info_training_iters = config.info_training_iters
        self.repeat_num = config.context_repeat_num
        self.device = torch.device(config.device)
        self.mini_bs = config.mini_batch_size
        lr = config.optimizer_lr_discriminator
        if config.state_only:
            self.discriminator = StateOnlyDiscriminator(config, dim_s=dim_s, dim_a=dim_a)
        else:
            self.discriminator = Discriminator(config, dim_s=dim_s, dim_a=dim_a)
        self.policy = Policy(config, dim_s=dim_s, dim_a=dim_a)
        self.criterion = torch.nn.BCELoss()
        self.optim = torch.optim.Adam(self.discriminator.parameters(), lr=lr, weight_decay=3.e-5) # only update the disc and fix the policy

        self.context_posterior = ContextPosterior(input_dim=self.dim_s + self.dim_a - self.dim_cnt,
                                                  hidden_dim=config.bi_run_hid_dim, output_dim=self.dim_cnt,
                                                  context_limit=self.cnt_limit)
        self.context_optim = torch.optim.Adam(self.context_posterior.parameters(), weight_decay=1.e-3)

        self.to(self.device)

    def airl_reward(self, s, a):
        log_sa = self.policy.log_prob_action(s, a) # (N, 1)
        log_sa = log_sa.detach().clone()
        f = self.discriminator.get_unnormed_d(s, a) # (N, 1)
        exp_f = torch.exp(f)
        d = (exp_f / (exp_f + 1.0)).detach().clone()
        # d = (exp_f / (exp_f + torch.exp(log_sa))).detach().clone() # (N, 1)
        # reward = torch.log(d + 1e-8) # corresponding to the PEMIRL paper
        reward = d
        # reward = -torch.log(1 - d + 1e-8)
        # print("here: ", reward)

        return reward

    def step(self, sample_sar, demo_sa, training_itr, n_step=10):
        # Context Posterior training
        if training_itr > self.cnt_starting_iter:
            print("Training the context posterior......")
            for _ in range(self.cnt_training_iters):
                # TODO: the number of trajectories is quite limited especially at the initial training stage,
                # which can be an issue, so maybe start the training of the context posterior at a later stage
                cnt_loss = torch.tensor(0.0, device=self.device)
                for s, a in sample_sar:
                    s_only = s[:, :-self.dim_cnt]
                    cnt = s[0:1, -self.dim_cnt:]
                    cnt_posterior_input = torch.cat([s_only, a], dim=-1).unsqueeze(0)
                    cnt_logp = self.context_posterior.log_prob_context(cnt_posterior_input, cnt)
                    cnt_loss -= cnt_logp.mean()

                cnt_loss /= float(len(sample_sar))
                self.context_optim.zero_grad()
                cnt_loss.backward()
                self.context_optim.step()
                print("Context Loss: ", cnt_loss.detach().clone().item())

        demo_sar = self.convert_demo(demo_sa)

        # Lemma 2 in the PEMIRL paper
        # TODO: move this part to the next loop, so that it can be updated multiple times
        # however, in the pseudo code of PEMIRL, the update of the disc is on two stages which are not correlated
        # the first stage is using lemma 2 which is based on trajectory and more similar with the update of the posterior
        # the second stage is based on the BCE loss, which is based on the transitions
        for _ in range(self.info_training_iters):
            logp_list, f_sum_list = [], []
            for s, a in sample_sar:
                s_only = s[:, :-self.dim_cnt]
                cnt = s[0:1, -self.dim_cnt:]
                cnt_posterior_input = torch.cat([s_only, a], dim=-1).unsqueeze(0)
                cnt_logp = self.context_posterior.log_prob_context(cnt_posterior_input, cnt) # (1, 1)
                f_sum = self.discriminator.get_unnormed_d(s, a).sum(dim=0, keepdim=True) # (1, 1)
                logp_list.append(cnt_logp)
                f_sum_list.append(f_sum)
            logp_tensor = torch.cat(logp_list, dim=0) # (bs, 1)
            f_sum_tensor = torch.cat(f_sum_list, dim=0) # (bs, 1)
            f_sum_mean = f_sum_tensor.view(-1, self.repeat_num, 1).mean(dim=1, keepdim=True).expand(-1, self.repeat_num, -1)
            f_sum_mean = f_sum_mean.reshape(-1, 1)
            info_loss = - self.info_coeff * (logp_tensor * (f_sum_tensor - f_sum_mean)).mean()
            self.optim.zero_grad()
            info_loss.backward()
            self.optim.step()
            print("Info_loss: ", info_loss.detach().clone().item())

        # Normal AIRL objectives
        sp = torch.cat([s for s, a in sample_sar], dim=0)
        se = torch.cat([s for s, a in demo_sar], dim=0)
        ap = torch.cat([a for s, a in sample_sar], dim=0)
        ae = torch.cat([a for s, a in demo_sar], dim=0)
        # huge difference compared with gail
        tp = torch.zeros(self.mini_bs, 1, dtype=torch.float32, device=self.device) # label for the generated state-action pairs
        te = torch.ones(self.mini_bs, 1, dtype=torch.float32, device=self.device) # label for the expert state-action pairs

        for _ in range(n_step):
            inds = torch.randperm(sp.size(0), device=self.device)
            for ind_p in inds.split(self.mini_bs):
                sp_b, ap_b, tp_b = sp[ind_p], ap[ind_p], tp[:ind_p.size(0)]
                ind_e = torch.randperm(se.size(0), device=self.device)[:ind_p.size(0)]
                se_b, ae_b, te_b = se[ind_e], ae[ind_e], te[:ind_p.size(0)]

                for _ in range(1):
                    # for the generated data
                    f_b = self.discriminator.get_unnormed_d(sp_b, ap_b)
                    log_sa_b = self.policy.log_prob_action(sp_b, ap_b)
                    log_sa_b = log_sa_b.detach().clone()
                    exp_f_b = torch.exp(f_b)

                    # d_b = exp_f_b / (exp_f_b + torch.exp(log_sa_b)) #  a prob between 0. to 1.
                    d_b = exp_f_b / (exp_f_b + 1.0)
                    d_b = torch.clamp(d_b, min=1e-3, max=1 - 1e-3)
                    loss_b = self.criterion(d_b, tp_b)
                    # for the expert data
                    f_e = self.discriminator.get_unnormed_d(se_b, ae_b)
                    log_sa_e = self.policy.log_prob_action(se_b, ae_b)
                    log_sa_e = log_sa_e.detach().clone()
                    exp_f_e = torch.exp(f_e)
                    # d_e = exp_f_e / (exp_f_e + torch.exp(log_sa_e))
                    d_e = exp_f_e / (exp_f_e + 1.0)
                    d_e = torch.clamp(d_e, min=1e-3, max=1 - 1e-3)
                    loss_e = self.criterion(d_e, te_b)
                    loss = loss_b + loss_e
                    loss += self.discriminator.gradient_penalty(sp_b, ap_b, lam=10.) #TODO
                    loss += self.discriminator.gradient_penalty(se_b, ae_b, lam=10.)

                    self.optim.zero_grad()
                    loss.backward()
                    # for p in self.discriminator.parameters():
                    #     print("before: ", p.data.norm(2))
                    # clip_grad_norm_(self.discriminator.parameters(), max_norm=20, norm_type=2)
                    # for p in self.discriminator.parameters():
                    #     print("after: ", p.data.norm(2))
                    self.optim.step()

    def convert_demo(self, demo_sa):
        with torch.no_grad():
            out_sample = []
            for s_array, a_array in demo_sa:
                # the s_array does not contain the context info for now
                # first, estimate the context using the context posterior
                assert s_array.shape[1] == self.dim_s - self.dim_cnt
                epi_len = s_array.shape[0]
                cnt_posterior_input = torch.cat([s_array, a_array], dim=-1).unsqueeze(0)
                cnt = self.context_posterior.sample_context(cnt_posterior_input,
                                                            fixed=self.cnt_sampling_fixed).detach().clone()  # (1, cnt_dim)
                s_array = torch.cat([s_array, cnt.expand(epi_len, -1)], dim=-1)
                out_sample.append((s_array, a_array))

        return out_sample

    def convert_sample(self, sample_sar):
        with torch.no_grad():
            out_sample = []
            r_sum_avg = 0.
            r_sum_max = -10000
            for s_array, a_array, r_real_array in sample_sar:
                # r_fake_array = self.airl_reward(s_array, a_array) #TODO
                out_sample.append((s_array, a_array))
                r_sum = r_real_array.sum().item()
                r_sum_avg += r_sum
                if r_sum > r_sum_max:
                    r_sum_max = r_sum
            r_sum_avg /= len(sample_sar)
        return out_sample, r_sum_avg, r_sum_max

    def get_il_reward(self, sample_sar):
        with torch.no_grad():
            out_sample = []
            for s_array, a_array in sample_sar:
                r_fake_array = self.airl_reward(s_array, a_array)
                out_sample.append((s_array, a_array, r_fake_array))

        return out_sample