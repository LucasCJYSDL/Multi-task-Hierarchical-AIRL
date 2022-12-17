from envir.mujoco_env import MujocoEnv as Env
from model.option_policy import OptionPolicy
from model.MHA_option_policy_critic import MHAOptionPolicy

import torch
import random
import numpy as np
from copy import deepcopy
from functools import partial
from multiprocessing import Pipe, Process


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            action = data
            next_s, reward, done = env.step(action)
            remote.send({"next_state": next_s, "reward": reward, "done": done})
        elif cmd == "reset":
            cnt = data['context']
            is_expert = data['is_expert']
            init_s = env.reset(cnt, is_expert)
            remote.send({"state": init_s})
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "set_seed":
            seed = data
            env.seed(seed)
        elif cmd == "sample_context":
            cnt = env.sample_context()
            remote.send({"context": cnt})
        else:
            raise NotImplementedError

class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

def env_fn(env_id):
    # return gym.make(env_id)
    temp_env = Env(env_id)
    temp_env.init()
    return temp_env

class EnvWrapper(object):
    def __init__(self, seed, env_id, env_num):
        self.env_num = env_num

        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.env_num)])
        self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, env_id)))) for worker_conn in self.worker_conns]

        for p in self.ps:
            p.daemon = True
            p.start()

        # for idx in range(self.env_num):
        #     # temp_seed = seed + idx + 1 # TODO
        #     self.parent_conns[idx].send(('set_seed', seed))

    def close(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def sample_context(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("sample_context", None))

        context_list = []
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            context_list.append(data['context'])

        return context_list

    def reset(self, context_list, is_expert):
        for idx, parent_conn in enumerate(self.parent_conns):
            parent_conn.send(("reset", {'context': context_list[idx], 'is_expert': is_expert}))

        init_states = []
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            init_states.append(data["state"])

        return np.array(init_states)

    def step(self, action_array, done_vec, s):
        for idx, parent_conn in enumerate(self.parent_conns):
            if not done_vec[idx]:
                parent_conn.send(("step", action_array[idx]))

        next_s = np.zeros_like(s, dtype=np.float32)
        r = np.zeros((self.env_num, 1), dtype=np.float32)
        done = [True for _ in range(self.env_num)]

        for idx, parent_conn in enumerate(self.parent_conns):
            if not done_vec[idx]:
                data = parent_conn.recv()
                next_s[idx] = data['next_state']
                r[idx] = data['reward']
                done[idx] = data['done']

        return next_s, r, done


def no_option_loop(env, policy, is_expert, fixed, task_list=None, is_repeat=False):
    env_num = env.env_num
    with torch.no_grad():

        if task_list is not None: # when testing, we will specify the list of tasks
            context_list = []
            for i in range(env_num):
                context_list.append(random.choice(task_list))
        else:
            context_list = env.sample_context()
            if is_repeat:
                context_list = [context_list[0] for _ in range(env_num)]

        s = env.reset(context_list, is_expert=is_expert) # (env_num, s_dim)
        horizons = [0 for _ in range(env_num)]
        done_vec = [False for _ in range(env_num)]
        # print("1: ", s)
        s_list, a_list, r_list = [], [], []
        while True:
            st = torch.as_tensor(s, dtype=torch.float32, device=policy.device) # (env_num, s_dim)
            at = policy.sample_action(st, fixed=fixed).detach() # (env_num, a_dim)
            at = at.cpu().numpy()
            next_s, rewards, done_vec = env.step(at, done_vec, s) # rewards: (env_num, 1)

            s_list.append(s.copy())
            a_list.append(at.copy())
            r_list.append(rewards.copy())

            s = next_s

            for idx in range(len(done_vec)):
                if not done_vec[idx]:
                    horizons[idx] += 1

            if np.array(done_vec).all():
                break

        rets = []
        for e_id in range(env_num):
            s_array, a_array, r_array = [], [], []
            temp_horizon = horizons[e_id] + 1
            for t_id in range(temp_horizon):
                s_array.append(torch.as_tensor([s_list[t_id][e_id]], dtype=torch.float32, device=policy.device))
                a_array.append(torch.as_tensor([a_list[t_id][e_id]], dtype=torch.float32, device=policy.device))
                r_array.append(torch.as_tensor([r_list[t_id][e_id]], dtype=torch.float32, device=policy.device))
            a_array = torch.cat(a_array, dim=0)
            s_array = torch.cat(s_array, dim=0)
            r_array = torch.cat(r_array, dim=0)
            # print("1: ", s_array.shape, a_array.shape, r_array.shape)
            rets.append((s_array, a_array, r_array))

        trans_num = np.sum(horizons)

    return rets, trans_num


def option_loop(env, policy, is_expert, fixed, task_list=None, is_repeat=False):
    assert not is_repeat
    env_num = env.env_num

    with torch.no_grad():
        if task_list is not None: # when testing, we will specify the list of tasks
            context_list = []
            for i in range(env_num):
                context_list.append(random.choice(task_list))
        else:
            context_list = env.sample_context()

        s = env.reset(context_list, is_expert=is_expert)  # (env_num, s_dim)
        horizons = [0 for _ in range(env_num)]
        done_vec = [False for _ in range(env_num)]
        # print("1: ", s)
        s_list, a_list, r_list, c_list = [], [], [], []

        ct = torch.empty(env_num, 1, dtype=torch.long, device=policy.device).fill_(policy.dim_c)
        c_list.append(ct.unsqueeze(1))

        while True:
            st = torch.as_tensor(s, dtype=torch.float32, device=policy.device) # (env_num, s_dim)
            ct = policy.sample_option(st, ct, fixed=fixed).detach() # (env_num, 1)
            at = policy.sample_action(st, ct, fixed=fixed).detach() # (env_num, a_dim)
            at = at.cpu().numpy()
            next_s, rewards, done_vec = env.step(at, done_vec, s) # rewards: (env_num, 1)

            s_list.append(s.copy())
            a_list.append(at.copy())
            r_list.append(rewards.copy())
            c_list.append(ct.unsqueeze(1))

            s = next_s

            for idx in range(len(done_vec)):
                if not done_vec[idx]:
                    horizons[idx] += 1

            if np.array(done_vec).all():
                break

        rets = []
        for e_id in range(env_num):
            s_array, a_array, r_array, c_array = [], [], [], []
            temp_horizon = horizons[e_id] + 1
            for t_id in range(temp_horizon):
                s_array.append(torch.as_tensor([s_list[t_id][e_id]], dtype=torch.float32, device=policy.device))
                a_array.append(torch.as_tensor([a_list[t_id][e_id]], dtype=torch.float32, device=policy.device))
                r_array.append(torch.as_tensor([r_list[t_id][e_id]], dtype=torch.float32, device=policy.device))
                c_array.append(torch.as_tensor(c_list[t_id][e_id], dtype=torch.long, device=policy.device))
            c_array.append(torch.as_tensor(c_list[temp_horizon][e_id], dtype=torch.long, device=policy.device))

            a_array = torch.cat(a_array, dim=0)
            s_array = torch.cat(s_array, dim=0)
            r_array = torch.cat(r_array, dim=0)
            c_array = torch.cat(c_array, dim=0)

            # print("1: ", a_array.shape, s_array.shape, r_array.shape, c_array.shape)

            rets.append((s_array, c_array, a_array, r_array))

        trans_num = np.sum(horizons)

    return rets, trans_num


class VecSampler(object):
    def __init__(self, seed, env_id, env_num, policy, is_expert=False, repeat_num=-1, task_list=None):
        self.vec_env = EnvWrapper(seed, env_id, env_num)
        self.env_num = env_num
        self.is_expert = is_expert
        self.repeat_num = repeat_num
        self.task_list = task_list
        self.policy = deepcopy(policy)
        if isinstance(policy, OptionPolicy) or isinstance(policy, MHAOptionPolicy):
            self.loop_func = option_loop
        else:
            self.loop_func = no_option_loop

    def filter_demo(self, sa_array):
        print("No filters are adopted.")
        return sa_array

    def collect(self, policy_param, n_sample, fixed=False):
        # n_sample < 0 for testing, denoting the number of trajs; n_sample > 0 for training, denoting the number of trans
        self.policy.load_state_dict(policy_param)
        counter = n_sample
        rets = []

        if counter > 0:
            while counter > 0:
                if self.repeat_num < 0:
                    trajs, trans_num = self.loop_func(self.vec_env, self.policy, self.is_expert, fixed=fixed, is_repeat=False)
                else:
                    trajs, trans_num = no_option_loop(self.vec_env, self.policy, self.is_expert, fixed=fixed, is_repeat=True)
                rets.extend(trajs)
                counter -= trans_num
            if self.repeat_num > 0:
                assert len(rets) % self.repeat_num == 0
        else:
            # assert self.task_list is not None
            while counter < 0: # only used for testing, so don't need repeated sampling
                trajs, _ = self.loop_func(self.vec_env, self.policy, self.is_expert, fixed=fixed, task_list=self.task_list)
                rets.extend(trajs)
                counter += self.env_num

        return rets
