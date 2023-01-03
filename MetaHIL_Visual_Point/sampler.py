import random

import torch
import time
from copy import deepcopy
from torch.multiprocessing import Process, Pipe, Lock, Value
from model.option_policy import OptionPolicy
from model.MHA_option_policy_critic import MHAOptionPolicy
from utils.common_utils import set_seed


__all__ = ["Sampler"]

def option_loop(env, policy, is_expert, fixed, task_cnt=None):
    with torch.no_grad():
        a_array = []
        c_array = []
        s_array = []
        r_array = []
        if task_cnt is not None: # when testing, we will specify the list of tasks
            context = task_cnt
        else:
            context = env.sample_context()
        s, done = env.reset(context, is_expert=is_expert), False
        ct = torch.empty(1, 1, dtype=torch.long, device=policy.device).fill_(policy.dim_c)
        c_array.append(ct)
        epi_id = 0
        while not done:
            epi_id += 1
            st = torch.as_tensor(s, dtype=torch.float32, device=policy.device).unsqueeze(0)

            ct = policy.sample_option(st, ct, fixed=fixed).detach()
            at = policy.sample_action(st, ct, fixed=fixed).detach()
            at = at.clamp(-1e6, 1e6)
            s_array.append(st)
            c_array.append(ct)
            a_array.append(at)
            s, r, done = env.step(at.cpu().squeeze(dim=0).numpy())
            r_array.append(r)
            if task_cnt is not None:
                if epi_id > 500:
                    done = True
        a_array = torch.cat(a_array, dim=0)
        c_array = torch.cat(c_array, dim=0)
        s_array = torch.cat(s_array, dim=0)
        r_array = torch.as_tensor(r_array, dtype=torch.float32, device=policy.device).unsqueeze(dim=-1)
    return s_array, c_array, a_array, r_array


def repeat_option_loop(env, policy, is_expert, fixed, repeat_num):
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
                at = at.clamp(-1e6, 1e6)
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
    def __init__(self, seed, env, test_env, policy, loop_func=None, repeat_loop_func=None, is_expert=False, repeat_num=-1, task_list=None):
        super(_SamplerSS, self).__init__(seed, policy)
        self.env = deepcopy(env)
        self.test_env = deepcopy(test_env)
        self.env.init(display=False)
        self.test_env.init(display=False)
        self.policy = deepcopy(policy)
        self.loop_func = loop_func
        self.repeat_loop_func = repeat_loop_func
        self.is_expert = is_expert
        self.repeat_num = repeat_num
        self.task_list = task_list

    def collect(self, policy_param, n_sample, fixed=False):
        self.policy.load_state_dict(policy_param)
        counter = n_sample
        rets = []
        if counter > 0:
            while counter > 0:
                if self.repeat_num < 0:
                    traj = self.loop_func(self.env, self.policy, self.is_expert, fixed=fixed)
                    rets.append(traj)
                    counter -= traj[0].size(0)
                else:
                    trajs, trans_num = self.repeat_loop_func(self.env, self.policy, self.is_expert, fixed=fixed, repeat_num=self.repeat_num)
                    rets.extend(trajs)
                    counter -= trans_num
            if self.repeat_num > 0:
                assert len(rets) % self.repeat_num == 0
        else:
            # assert self.task_list is not None
            while counter < 0: # only used for testing, so don't need repeated sampling
                # print("111: ", counter-n_sample, self.task_list[counter - n_sample])
                traj = self.loop_func(self.test_env, self.policy, self.is_expert, fixed=fixed, task_cnt=self.task_list[counter - n_sample])
                rets.append(traj)
                counter += 1
        return rets


def Sampler(seed, env, test_env, policy, is_expert, repeat_num=-1, task_list=None) -> _SamplerCommon:

    loop_func = option_loop
    repeat_loop_func = repeat_option_loop

    class_m = _SamplerSS
    return class_m(seed, env, test_env, policy, loop_func, repeat_loop_func, is_expert, repeat_num, task_list)


if __name__ == "__main__":
    from torch.multiprocessing import set_start_method
    set_start_method("spawn")
