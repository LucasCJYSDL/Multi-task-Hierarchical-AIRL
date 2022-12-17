import time

import torch
import torch.nn.functional as F
from .option_policy import Policy
from .option_discriminator import Discriminator, SMILEStateOnlyDiscriminator, StateOnlyDiscriminator
from .context_net import MLPContextEncoder, ContextPosterior
from utils.config import Config
from utils.model_util import clip_grad_norm_

class SMILE_AIRL(torch.nn.Module):
    def __init__(self, config: Config, dim_s, dim_a, dim_cnt, cnt_limit):
        super(SMILE_AIRL, self).__init__()
        self.dim_a = dim_a
        self.dim_s = dim_s
        self.dim_cnt = dim_cnt
        self.cnt_limit = cnt_limit

        self.repeat_num = config.context_repeat_num # for each task, repeat sampling self.repeat_num trajectories
        self.device = torch.device(config.device)
        self.mini_bs = config.mini_batch_size
        lr = config.optimizer_lr_discriminator
        if config.state_only:
            # TODO: use the StateOnlyDiscriminator instead which is defined from AIRL paper
            # as mentioned in the Smile paper, they change the input from (s, a) to (s, s'), which is weird
            # maybe because the author didn't actually understand the AIRL paper
            self.discriminator = SMILEStateOnlyDiscriminator(config, dim_s=dim_s, dim_a=dim_a)
            # self.discriminator = StateOnlyDiscriminator(config, dim_s=dim_s, dim_a=dim_a)
        else:
            self.discriminator = Discriminator(config, dim_s=dim_s, dim_a=dim_a)
        self.policy = Policy(config, dim_s=dim_s, dim_a=dim_a)
        self.criterion = torch.nn.BCELoss()
        self.optim = torch.optim.Adam(self.discriminator.parameters(), lr=lr, weight_decay=3.e-5) # only update the disc and fix the policy

        self.state_only = config.state_only
        self.exp_traj_batch = config.exp_traj_batch

        if config.state_only:
            cnt_input_dim = int((self.dim_s - self.dim_cnt) * 2)
        else:
            cnt_input_dim = int((self.dim_s - self.dim_cnt) * 2 + self.dim_a)

        if config.use_mlp_encoder: # from the smile implementation, but we made modifications to make the stucture similar with others'
            self.context_encoder = MLPContextEncoder(input_dim=cnt_input_dim, hidden_dim=config.bi_run_hid_dim,
                                                         inter_dim=config.enc_inter_dim, output_dim=self.dim_cnt,
                                                         context_limit=self.cnt_limit)
        else:
            self.context_encoder = ContextPosterior(input_dim=cnt_input_dim,
                                                     hidden_dim=config.bi_run_hid_dim, output_dim=self.dim_cnt,
                                                     context_limit=self.cnt_limit)

        self.context_optim = torch.optim.Adam(self.context_encoder.parameters(), lr=1e-5, weight_decay=1.e-3)

        self.to(self.device)

    def airl_reward(self, s, a):
        log_sa = self.policy.log_prob_action(s, a) # (N, 1)
        log_sa = log_sa.detach().clone()
        f = self.discriminator.get_unnormed_d(s, a) # (N, 1)
        exp_f = torch.exp(f)
        # d = (exp_f / (exp_f + 1.0)).detach().clone()
        d = (exp_f / (exp_f + torch.exp(log_sa))).detach().clone() # (N, 1)
        # reward = torch.log(d + 1e-8) - torch.log(1 - d + 1e-8) # corresponding to the SMILE paper
        reward = d
        # reward = - torch.log(1 - d + 1e-8)
        # print("here: ", reward)

        return reward

    def _get_expert_data(self, demo_sar):
        s_list = []
        a_list = []
        c_list = []
        for s, a in demo_sar:
            seq_len = s.shape[0]
            assert s.shape[1] == self.dim_s - self.dim_cnt
            temp_s = s.detach().clone()
            next_s = temp_s[1:]
            last_s = temp_s[-1]
            next_s = torch.cat([next_s, last_s.unsqueeze(0)], dim=0)
            if self.state_only:
                input = torch.cat([s, next_s], dim=-1)
            else:
                input = torch.cat([s, a, next_s], dim=-1)
            input = input.unsqueeze(0)

            cnt = self.context_encoder.sample_context(input, fixed=False) # cannot be detached, since we need gradient for this encoder
            c_list.append(cnt.expand(seq_len, -1))
            s_list.append(s)
            a_list.append(a)

        se = torch.cat(s_list, dim=0)
        ae = torch.cat(a_list, dim=0)
        ce = torch.cat(c_list, dim=0)

        return se, ae, ce


    def _get_batch_expert_data(self, demo_sar):
        input_list = []
        for s, a in demo_sar:
            assert s.shape[1] == self.dim_s - self.dim_cnt
            temp_s = s.detach().clone()
            next_s = temp_s[1:]
            last_s = temp_s[-1]
            next_s = torch.cat([next_s, last_s.unsqueeze(0)], dim=0)
            if self.state_only:
                input = torch.cat([s, next_s], dim=-1)
            else:
                input = torch.cat([s, a, next_s], dim=-1)
            input = input.unsqueeze(0)
            input_list.append(input)

        bs = len(input_list)
        start_idx = 0
        cnt_list = []
        while start_idx < bs:
            end_idx = min(start_idx + self.exp_traj_batch, bs)
            cnts = self.context_encoder.sample_contexts(seq_list=input_list[start_idx: end_idx], fixed=False)
            cnt_list.append(cnts)
            start_idx = start_idx + self.exp_traj_batch

        cnt_tensor = torch.cat(cnt_list, dim=0) # (bs, c_dim)
        s_list = []
        a_list = []
        c_list = []
        for i in range(len(demo_sar)):
            s, a = demo_sar[i]
            seq_len = s.shape[0]
            s_list.append(s)
            a_list.append(a)
            c_list.append(cnt_tensor[i].unsqueeze(0).expand(seq_len, -1))

        se = torch.cat(s_list, dim=0)
        ae = torch.cat(a_list, dim=0)
        ce = torch.cat(c_list, dim=0)

        return se, ae, ce


    def step(self, sample_sar, demo_sar, training_itr, n_step=10):

        sp = torch.cat([s for s, a in sample_sar], dim=0)
        ap = torch.cat([a for s, a in sample_sar], dim=0)

        tp = torch.zeros(self.mini_bs, 1, dtype=torch.float32, device=self.device) # label for the generated state-action pairs
        te = torch.ones(self.mini_bs, 1, dtype=torch.float32, device=self.device) # label for the expert state-action pairs

        for _ in range(n_step//2): # Danger
            inds = torch.randperm(sp.size(0), device=self.device)
            for ind_p in inds.split(self.mini_bs):
                sp_b, ap_b, tp_b = sp[ind_p], ap[ind_p], tp[:ind_p.size(0)]

                # big difference compared with other algorithms
                if self.exp_traj_batch == 1:
                    se, ae, ce = self._get_expert_data(demo_sar)
                else:
                    se, ae, ce = self._get_batch_expert_data(demo_sar)
                ind_e = torch.randperm(se.size(0), device=self.device)[:ind_p.size(0)]
                se_b, ae_b, ce_b, te_b = se[ind_e], ae[ind_e], ce[ind_e], te[:ind_p.size(0)] # ce_b must contain gradient info

                # for the generated data
                f_b = self.discriminator.get_unnormed_d(sp_b, ap_b)
                log_sa_b = self.policy.log_prob_action(sp_b, ap_b)
                log_sa_b = log_sa_b.detach().clone()
                exp_f_b = torch.exp(f_b)
                # d_b = exp_f_b / (exp_f_b + 1.0)
                d_b = exp_f_b / (exp_f_b + torch.exp(log_sa_b)) #  a prob between 0. to 1.
                d_b = torch.clamp(d_b, min=1e-3, max=1 - 1e-3)
                loss_b = self.criterion(d_b, tp_b)
                # for the expert data
                f_e = self.discriminator.get_unnormed_d(torch.cat([se_b, ce_b], dim=-1), ae_b)
                log_sa_e = self.policy.log_prob_action(torch.cat([se_b, ce_b], dim=-1), ae_b)
                log_sa_e = log_sa_e.detach().clone()
                exp_f_e = torch.exp(f_e)
                d_e = exp_f_e / (exp_f_e + torch.exp(log_sa_e))
                # d_e = exp_f_e / (exp_f_e + 1.0)
                d_e = torch.clamp(d_e, min=1e-3, max=1 - 1e-3)
                loss_e = self.criterion(d_e, te_b)
                loss = loss_b + loss_e
                loss += self.discriminator.gradient_penalty(sp_b, ap_b, lam=10.)
                # very important, context encoder is not updated with the GP loss, so we do a detach here
                loss += self.discriminator.gradient_penalty(torch.cat([se_b, ce_b.clone().detach()], dim=-1), ae_b, lam=10.)

                self.optim.zero_grad()
                self.context_optim.zero_grad()
                loss.backward()
                # for p in self.discriminator.parameters():
                #     print("before: ", p.data.norm(2))
                # clip_grad_norm_(self.discriminator.parameters(), max_norm=20, norm_type=2)
                # for p in self.discriminator.parameters():
                #     print("after: ", p.data.norm(2))
                self.optim.step()
                self.context_optim.step()

    # def convert_demo(self, demo_sa): # actually did nothing
    #     with torch.no_grad():
    #         out_sample = []
    #         r_sum_avg = 0.
    #         for s_array, a_array in demo_sa:
    #             # the s_array does not contain the context info for now
    #             # first, estimate the context using the context posterior
    #             assert s_array.shape[1] == self.dim_s - self.dim_cnt
    #             epi_len = s_array.shape[0]
    #
    #             # r_fake_array = self.airl_reward(s_array, a_array)
    #             r_fake_array = torch.zeros([epi_len, 1]) # only for evaluation
    #             out_sample.append((s_array, a_array, r_fake_array))
    #             r_sum_avg += r_fake_array.sum().item()
    #         r_sum_avg /= len(demo_sa)
    #     return out_sample, r_sum_avg

    def convert_sample(self, sample_sar):
        with torch.no_grad():
            out_sample = []
            r_sum_avg = 0.
            r_sum_max = -10000
            for s_array, a_array, r_real_array in sample_sar:
                # r_fake_array = self.airl_reward(s_array, a_array)
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