import random

import torch
import time
from copy import deepcopy
from torch.multiprocessing import Process, Pipe, Lock, Value
from model.option_policy import OptionPolicy
from model.MHA_option_policy_critic import MHAOptionPolicy
from utils.common_utils import set_seed


__all__ = ["Sampler"]

class _sQueue(object):
    def __init__(self, pipe_rw, r_lock, w_lock):
        self.rlock = r_lock
        self.wlock = w_lock
        self.pipe_rw = pipe_rw

    def __del__(self):
        self.pipe_rw.close()

    def get(self, time_out=0.):
        d = None
        if self.pipe_rw.poll(time_out):
            with self.rlock:
                d = self.pipe_rw.recv()
        return d

    def send(self, d):
        with self.wlock:
            self.pipe_rw.send(d)


def pipe_pair():
    p_lock = Lock()
    c_lock = Lock()
    pipe_c, pipe_p = Pipe(duplex=True)
    child_q = _sQueue(pipe_c, p_lock, c_lock)
    parent_q = _sQueue(pipe_p, c_lock, p_lock)
    return child_q, parent_q

def option_loop(env, policy, is_expert, fixed, task_list=None):
    with torch.no_grad():
        a_array = []
        c_array = []
        s_array = []
        r_array = []
        if task_list is not None: # when testing, we will specify the list of tasks
            context = random.choice(task_list)
        else:
            context = env.sample_context()
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
    return s_array, c_array, a_array, r_array

def loop(env, policy, is_expert, fixed, task_list=None):
    with torch.no_grad():
        a_array = []
        s_array = []
        r_array = []
        if task_list is not None: # when testing, we will specify the list of tasks
            context = random.choice(task_list)
        else:
            context = env.sample_context()
        s, done = env.reset(context, is_expert=is_expert), False
        # print("1: ", s)
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
        # print(r_array.shape)
    return s_array, a_array, r_array

def repeat_loop(env, policy, is_expert, fixed, repeat_num):
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

class _SamplerCommon(object):
    def __init__(self, seed, policy):
        self.device = policy.device

    def collect(self, policy_param, n_sample, fixed=False):
        raise NotImplementedError()

    def filter_demo(self, sa_array):
        print("No filters are adopted.")
        return sa_array


class _Sampler(_SamplerCommon):
    def __init__(self, seed, env, policy, loop_func=None, is_expert=False, repeat_num=-1, task_list=None, n_thread=4):
        super(_Sampler, self).__init__(seed, policy)
        self.counter = Value('i', 0)
        self.state = Value('i', n_thread)
        child_q, self.queue = pipe_pair()
        self.procs = [Process(target=self.worker, name=f"subproc_{seed}",
                              args=(seed, env, policy, loop_func, is_expert, repeat_num, task_list, self.state, self.counter, child_q))
                      for _ in range(n_thread)]
        self.pids = []
        for p in self.procs:
            p.daemon = True
            p.start()
            self.pids.append(p.pid)

        while self.state.value > 0: # wait for all the workers to prepare well
            time.sleep(0.1)

    def collect(self, policy_param, n_sample, fixed=False):
        # n_sample <0 for number of trajectories, >0 for number of sa pairs
        for _ in self.procs: # send the msg from the parent node to the child node
            self.queue.send((policy_param, fixed))

        with self.state.get_lock(): # updating the value of state or counter needs to be done with in get_lock(); tell the workers to receive the pi parameters
            self.state.value = -len(self.procs)

        while self.state.value < 0: # wait for all the workers to receive and load the parameters
            time.sleep(0.1)

        with self.counter.get_lock():
            self.counter.value = n_sample

        with self.state.get_lock(): # tell the workers to start to collect counter.value datas
            self.state.value = len(self.procs)

        ret = []
        while self.state.value > 0: # wait until the workers to collect enough data
            d = self.queue.get(0.0001)
            while d is not None:
                traj = d

                if not isinstance(traj, list):
                    ret.append(tuple(x.to(self.device) for x in traj))
                else:
                    for tt in traj:
                        ret.append(tuple(x.to(self.device) for x in tt))

                d = self.queue.get(0.0001)

        return ret

    def __del__(self):
        print(f"agent process is terminated, check if any subproc left: {self.pids}")
        for p in self.procs:
            p.terminate()

    @staticmethod
    def worker(seed: int, env, policy, loop_func, is_expert, repeat_num, task_list, state: Value, counter: Value, queue: _sQueue):
        # state 0: idle, -n: init param, n: sampling
        set_seed(seed)

        env.init(display=False)
        with state.get_lock():
            state.value -= 1

        while True:
            while state.value >= 0: # wait for the policy parameters from collect(); starts from 0 everytime.
                time.sleep(0.1)

            d = None
            while d is None:
                d = queue.get(5)

            net_param, fixed = d
            policy.load_state_dict(net_param)

            with state.get_lock(): # all the works do this step, so the state-value will become 0
                state.value += 1

            while state.value <= 0: # wait for the amount of data to sample from collect()
                time.sleep(0.1)

            while state.value > 0: # wait for collecting enough data
                if counter.value < 0:
                    traj = loop_func(env, policy, fixed=fixed, is_expert=is_expert, task_list=task_list)
                else:
                    if repeat_num < 0:
                        traj = loop_func(env, policy, fixed=fixed, is_expert=is_expert, task_list=None)
                    else:
                        traj, trans_num = repeat_loop(env, policy, fixed=fixed, is_expert=is_expert, repeat_num=repeat_num)

                with counter.get_lock():
                    if counter.value > 0:
                        if not isinstance(traj, list):
                            send_term = tuple(x.cpu() for x in traj)
                            trans_num = traj[0].size(0)
                        else:
                            send_term = []
                            for tt in traj:
                                send_term.append(tuple(x.cpu() for x in tt))

                        queue.send(send_term)
                        counter.value -= trans_num
                        if counter.value <= 0:
                            counter.value = 0
                            with state.get_lock():
                                state.value = 0

                    elif counter.value < 0:
                        queue.send(tuple(x.cpu() for x in traj))
                        counter.value += 1
                        if counter.value >= 0:
                            counter.value = 0
                            with state.get_lock():
                                state.value = 0


class _SamplerSS(_SamplerCommon):
    def __init__(self, seed, env, policy, loop_func=None, is_expert=False, repeat_num=-1, task_list=None, n_thread=1):
        super(_SamplerSS, self).__init__(seed, policy)
        self.env = deepcopy(env)
        self.env.init(display=False)
        self.policy = deepcopy(policy)
        self.loop_func = loop_func
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
                    trajs, trans_num = repeat_loop(self.env, self.policy, self.is_expert, fixed=fixed, repeat_num=self.repeat_num)
                    rets.extend(trajs)
                    counter -= trans_num
            if self.repeat_num > 0:
                assert len(rets) % self.repeat_num == 0
        else:
            # assert self.task_list is not None
            while counter < 0: # only used for testing, so don't need repeated sampling
                traj = self.loop_func(self.env, self.policy, self.is_expert, fixed=fixed, task_list=self.task_list)
                rets.append(traj)
                counter += 1
        return rets


def Sampler(seed, env, policy, n_thread, is_expert, repeat_num=-1, task_list=None) -> _SamplerCommon:

    if isinstance(policy, OptionPolicy) or isinstance(policy, MHAOptionPolicy):
        loop_func = option_loop
    else:
        loop_func = loop
    # class_m = _SamplerSS
    class_m = _Sampler if n_thread > 1 else _SamplerSS
    return class_m(seed, env, policy, loop_func, is_expert, repeat_num, task_list, n_thread)


if __name__ == "__main__":
    from torch.multiprocessing import set_start_method
    set_start_method("spawn")
