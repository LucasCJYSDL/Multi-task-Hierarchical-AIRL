import os
import random
import numpy as np
import torch
try:
    import pybullet_envs
except ImportError:
    print("Warning: pybullet not installed, bullet environments will be unavailable")
import gym
from envir import mujoco_maze, mujoco_manipulation, d4rl_env


class MujocoEnv(object):
    def __init__(self, task_name: str = "HalfCheetah-v2"):
        self.task_name = task_name
        self.env = None
        self.display = False

    def init(self, display=False):
        self.env = gym.make(self.task_name)
        self.display = display
        return self

    def get_context_info(self):
        if self.env is not None:
            return self.env.get_context_dim(), self.env.get_context_limit()
        else:
            env = gym.make(self.task_name)
            dim_cnt = env.get_context_dim()
            cnt_limit = env.get_context_limit()
            env.close()
        return dim_cnt, cnt_limit

    def sample_context(self):
        return self.env.sample_context()

    def reset(self, context, is_expert):
        self.env.apply_context(context, is_expert=is_expert)
        s = self.env.reset()
        return s

    def get_expert_act(self, obs):
        act = self.env.get_expert_action(obs)
        return act

    def render(self):
        self.env.render()

    def seed(self, seed_idx):
        self.env.seed(seed_idx)

    def step(self, a):
        s, reward, terminate, info = self.env.step(a)
        if self.display:
            self.env.render()
        return s, reward, terminate

    def state_action_size(self, is_expert=False):
        if self.env is not None:
            s = self.reset(self.sample_context(), is_expert=is_expert)
            s_dim = s.shape[0]
            a_dim = self.env.action_space.shape[0]
        else:
            env = gym.make(self.task_name)
            env.apply_context(env.sample_context(), is_expert=is_expert)
            s = env.reset()
            s_dim = s.shape[0]
            a_dim = env.action_space.shape[0]
            env.close()
        return s_dim, a_dim


def get_demo(train_path, test_path, n_traj, task_specific=False):
    print(f"Demo Loaded from {train_path} and {test_path}")
    train_set = torch.load(train_path)
    test_set = torch.load(test_path)

    assert n_traj % len(train_set) == 0
    traj_per_task = n_traj // len(train_set)

    # the structure of the demonstration data for all the algorithms are kept the same for fairness
    if not task_specific: # for algorithms other than MAML-IL
        train_demos = []
        for task_idx in train_set: # no need to shuffle the keys of the train set
            train_demos.extend(train_set[task_idx]['demos'][:traj_per_task])
            # if len(train_demos) >= n_traj:
            #     break
        random.shuffle(train_demos) # the sort of the trajectories should not be correlated with the task variable
        test_contexs = []
        for task_idx in test_set:
            test_contexs.append(test_set[task_idx]['context'])

        return train_demos, test_contexs

    # for MAML-IL
    train_demos = {}
    cur_traj = 0
    task_num = 0
    for task_idx in train_set:
        train_demos[task_idx] = {'context': train_set[task_idx]['context'], 'demos': train_set[task_idx]['demos'][:traj_per_task]}
        # train_demos[task_idx] = train_set[task_idx]
        # cur_traj += len(train_set[task_idx]['demos'])
        # if cur_traj >= n_traj:
        #     break
        task_num += 1
        # if task_num >= 8:
        #     break

    return train_demos, test_set


def collect_demo(config, n_task=1000, demo_per_task=10, data_type='train',
                 display=False, is_manual=False, env_name=None, expert_path=None):
    from model.option_policy import Policy, OptionPolicy
    # you must have an expert model first, by running 'run_ppo_expert.py'.
    if not is_manual:
        env = MujocoEnv(config.env_name)
        path = f"./{config.env_name}_sample_" + data_type + '.torch'
    else:
        env = MujocoEnv(env_name)
        path = f"./{env_name}_sample_" + data_type + '.torch'
    dim_s, dim_a = env.state_action_size()
    env.init(display=display)

    if not is_manual:
        config.device = 'cpu'
        policy_state = torch.load(expert_path, map_location='cuda:0')
        policy = Policy(config, dim_s, dim_a)
        # policy = OptionPolicy(config, dim_s, dim_a)
        policy.load_state_dict(policy_state)

    demo_set = {}
    for task_idx in range(n_task):
        context = env.sample_context()
        demo_set[task_idx] = {'context': context}
        trajs = []
        while len(trajs) < demo_per_task:
            with torch.no_grad():
                s_array = []
                a_array = []
                r_array = []
                s, done = env.reset(context, is_expert=True), False
                while not done:
                    st = torch.as_tensor(s, dtype=torch.float32).unsqueeze(dim=0)
                    s_array.append(st.clone())
                    if not is_manual:
                        at = policy.sample_action(st, fixed=True)  # eliminate the randomness of the expert policy
                    else:
                        at = env.get_expert_act(obs=st.clone().numpy()[0])
                        at = torch.tensor(at, dtype=torch.float32, device=st.device).unsqueeze(dim=0)
                    a_array.append(at.clone())
                    s, r, done = env.step(at.squeeze(dim=0).cpu().detach().clone().numpy())
                    r_array.append(r)
                a_array = torch.cat(a_array, dim=0)
                s_array = torch.cat(s_array, dim=0)
                r_array = torch.as_tensor(r_array, dtype=torch.float32).unsqueeze(dim=1)

                print(f"R-Sum={r_array.sum()}, L={r_array.size(0)}")
                if r_array.sum().item() > 300: # or 300
                    print("Keep it!")
                    trajs.append((s_array, a_array, r_array))

        demo_set[task_idx]['demos'] = trajs

    torch.save(demo_set, path)

def get_demo_stat(path=""):
    if os.path.isfile(path):
        print(f"Demo Loaded from {path}")
        samples = torch.load(path) # TODO
        # print(samples)
        aver_r = 0.0
        r_max_array = []
        n_traj = 0
        n_tran = 0
        temp_r_array = []
        for task_idx in samples:
            temp_list = samples[task_idx]['demos']
            for traj in temp_list:
                s, a, r = traj
                print(s.shape, a.shape, r.shape, r.sum())
                temp_r_array.append(r.sum())
                if len(temp_r_array) == 8:
                    r_max_array.append(np.max(temp_r_array))
                    temp_r_array = []
                aver_r += r.sum()
                n_traj += 1
                n_tran += r.shape[0]

        print(aver_r/n_traj, n_traj, n_tran, np.mean(r_max_array), len(r_max_array))


if __name__ == '__main__':

    # collect_demo(config=None, n_demo=10000, is_manual=True, env_name='PointCell-v0')
    #
    # import torch.multiprocessing as multiprocessing
    # from utils.config import Config, ARGConfig
    # from default_config import mujoco_config
    #
    # multiprocessing.set_start_method('spawn')
    #
    # arg = ARGConfig()
    # arg.add_arg("env_type", "mujoco", "Environment type, can be [mujoco, ...]")
    # arg.add_arg("env_name", "HalfCheetahVel-v0", "Environment name")
    # arg.add_arg("algo", "ppo", "Environment type, can be [ppo, option_ppo]")
    # arg.add_arg("device", "cuda:0", "Computing device")
    # arg.add_arg("tag", "default", "Experiment tag")
    # arg.add_arg("seed", 0, "Random seed")
    # arg.parser()
    #
    # config = mujoco_config
    # config.update(arg)
    # if config.env_name.startswith("Ant") or config.env_name.startswith("Walker") or config.env_name.startswith("HalfCheetah"):
    #     config.hidden_policy = (128, 128)
    #     config.hidden_critic = (128, 128)
    #     print(f"Training this env with larger policy network size :{config.hidden_policy}")
    #
    # print(config.algo)
    # config.use_option = True
    # config.use_c_in_discriminator = False  # in fact, there are no discriminators
    # config.use_d_info_gail = False
    # config.use_vae = False
    # config.train_option = True
    # if config.algo == 'ppo':
    #     config.use_option = False
    #     config.train_option = False
    #
    # collect_demo(config, n_task=100, demo_per_task=10, data_type='train', expert_path='./exp_model/HalfCheetahVel/6549.torch')
    # collect_demo(config, n_task=50, demo_per_task=10, data_type='test', expert_path='./exp_model/HalfCheetahVel/6549.torch')

    get_demo_stat('../data/mujoco/HalfCheetahVel-v0_sample_train.torch')
    get_demo_stat('../data/mujoco/HalfCheetahVel-v0_sample_test.torch')

    # train, test = get_demo('PointCell-v0_sample_train.torch', 'PointCell-v0_sample_test.torch', 10, task_specific=True)
    # print(train)
    # print(len(test))