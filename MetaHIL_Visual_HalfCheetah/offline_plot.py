#!/usr/bin/env python3
import os
import torch
from typing import Union
import torch.multiprocessing as multiprocessing
from model.MHA_option_ppo import MHAOptionPPO
from model.hierarchical_ppo import HierarchicalPPO
from model.MHA_option_il import MHAOptionAIRL
from utils.common_utils import validate, reward_validate, get_dirs, set_seed
from sampler import Sampler
from utils.logger import Logger
from utils.config import ARGConfig, Config
from default_config import mujoco_config
from envir.visualize_dst import draw_vel_change


def make_il(config: Config, dim_s, dim_a, dim_cnt, cnt_limit):
    il = MHAOptionAIRL(config, dim_s=dim_s, dim_a=dim_a, dim_cnt=dim_cnt, cnt_limit=cnt_limit)

    if config.use_hppo:
        ppo = HierarchicalPPO(config, il.policy, dim_s)
    else:
        ppo = MHAOptionPPO(config, il.policy, dim_s)

    return il, ppo


def plot(config: Config, msg="default"):
    ## prepare
    env_type = config.env_type
    if env_type == "mujoco":
        from envir.mujoco_env import MujocoEnv as Env, get_demo
    else:
        raise ValueError(f"Unknown env type {env_type}")

    n_traj = config.n_traj
    n_sample = config.n_sample
    n_epoch = config.n_epoch
    seed = config.seed
    env_name = config.env_name

    set_seed(seed)
    log_dir, save_dir, train_set_name, test_set_name, _, time_token = get_dirs(seed, config.algo, env_type, env_name, msg)

    env = Env(env_name)
    dim_s, dim_a = env.state_action_size()
    dim_cnt, cnt_limit = env.get_context_info()

    print("The dimension info of the environment: name:{}, dim_s:{}, dim_a:{}, context_dim:{}, context_limit:{}.".format(
        env_name, dim_s, dim_a, dim_cnt, cnt_limit))

    # the demonstration does not contain task labels
    demo, test_contexts = get_demo(train_set_name, test_set_name, n_traj=n_traj, task_specific=False)
    # the evaluation in only on the tasks contained in the test set, which is controlled to be the same as the MAML-IL

    il, ppo = make_il(config, dim_s=dim_s, dim_a=dim_a, dim_cnt=dim_cnt, cnt_limit=cnt_limit)
    il.load_state_dict(state_dict=torch.load(config.exp_path, map_location='cuda:0'))

    sampling_agent = Sampler(seed, env, il.policy, is_expert=False, task_list=test_contexts)

    # v_l, cs_demo = validate(il.policy, [(tr[0], tr[-2]) for tr in demo_sxar])

    samples = {}
    for i in range(10):

        info_dict, trajs = reward_validate(sampling_agent, il.policy, time_token=time_token, n_sample=-len(test_contexts),
                                               training_episode=i, do_print=True)
        for task_id in range(len(trajs)):
            if task_id not in samples:
                samples[task_id] = {'demos':[]}
            samples[task_id]['demos'].append(trajs[task_id])

    draw_vel_change(samples)
    torch.save(samples, 'offline_plot.torch')


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    arg = ARGConfig()
    arg.add_arg("env_type", "mujoco", "Environment type, can be [mujoco, ...]")
    arg.add_arg("env_name", "HalfCheetahVel-v0", "Environment name") # HalfCheetahVel-v0, WalkerRandParams-v0
    arg.add_arg("algo", "meta_hier_airl", "which algorithm to use, can be [meta_hier_airl, ...]")
    arg.add_arg("device", "cuda:0", "Computing device")
    arg.add_arg("tag", "default", "Experiment tag")
    arg.add_arg("seed", 0, "Random seed")
    arg.add_arg("n_traj", 1000, "Number of trajectories for demonstration") # should be 1000
    arg.parser()

    if arg.env_type == "mujoco":
        config = mujoco_config
    else:
        raise NotImplementedError

    config.update(arg)
    if config.env_name.startswith("Ant") or config.env_name.startswith("Walker"):
        config.hidden_option = (128, 128)
        config.hidden_policy = (128, 128)
        config.hidden_critic = (128, 128)

    elif config.env_name.startswith("Kitchen"):
        # config.n_sample = 512
        config.hidden_option = (256, 256)
        config.hidden_policy = (256, 256)
        config.hidden_critic = (256, 256)


    config.is_airl = True
    config.use_option_posterior = True
    config.use_c_in_discriminator = True # c actually corresponds to the option choice in the paper
    config.use_vae = False

    print(f">>>> Training {config.algo} using {config.env_name} environment on {config.device}")
    config.exp_path = "./result/HalfCheetahVel-v0_2022_12_30_12_4_3_meta_hier_airl_seed_0/model/799.torch"
    plot(config, msg=config.tag)