#!/usr/bin/env python3
import os
import torch
import torch.multiprocessing as multiprocessing
from utils.common_utils import validate, reward_validate, get_dirs, set_seed
from sampler import Sampler
from utils.logger import Logger
from utils.config import ARGConfig, Config
from default_config import mujoco_config
from model.pemirl_airl import PEMIRL_AIRL
from model.smile_airl import SMILE_AIRL
from model.option_ppo import PPO

def make_il(config: Config, dim_s, dim_a, dim_cnt, cnt_limit):
    if config.algo == 'PEMIRL':
        il = PEMIRL_AIRL(config, dim_s=dim_s, dim_a=dim_a, dim_cnt=dim_cnt, cnt_limit=cnt_limit)
        ppo = PPO(config, il.policy)
    elif config.algo == 'SMILE':
        il = SMILE_AIRL(config, dim_s=dim_s, dim_a=dim_a, dim_cnt=dim_cnt, cnt_limit=cnt_limit)
        ppo = PPO(config, il.policy)
    else:
        raise NotImplementedError

    return il, ppo

def sample_batch(il, agent, n_sample, demo_sa_array):
    demo_sa_in = agent.filter_demo(demo_sa_array)
    sample_sxar_in = agent.collect(il.policy.state_dict(), n_sample, fixed=False)
    sample_sxar, sample_rsum, sample_rsum_max = il.convert_sample(sample_sxar_in) # replace the real environment reward with the one generated with IL
    #TODO: time-consuming
    # demo_sxar, demo_rsum = il.convert_demo(demo_sa_in)
    return sample_sxar, sample_rsum, sample_rsum_max

def train_g(ppo, sample_sxar, factor_lr):
    ppo.step(sample_sxar, lr_mult=factor_lr)

def train_d(il, sample_sxar, demo_sa, training_itr, n_step=10):
    il.step(sample_sxar, demo_sa, training_itr=training_itr, n_step=n_step)


def learn(config: Config, msg="default"):
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
    log_dir, save_dir, train_set_name, test_set_name, _ = get_dirs(seed, config.algo, env_type, env_name, msg)
    with open(os.path.join(save_dir, "config.log"), 'w') as f:
        f.write(str(config))  # important for reproducing and visualisaton
    logger = Logger(log_dir)  # tensorboard
    save_name_f = lambda i: os.path.join(save_dir, f"{i}.torch")
    save_name_ppo_f = lambda i: os.path.join(save_dir, f"{i}_critic.torch")

    env = Env(env_name)
    dim_s, dim_a = env.state_action_size()
    dim_cnt, cnt_limit = env.get_context_info()
    print("The dimension info of the environment: name:{}, dim_s:{}, dim_a:{}, context_dim:{}, context_limit:{}.".format(
            env_name, dim_s, dim_a, dim_cnt, cnt_limit))

    # the demonstration does not contain task labels
    demo, test_contexts = get_demo(train_set_name, test_set_name, n_traj=n_traj, task_specific=False)
    # the evaluation in only on the tasks contained in the test set, which is controlled to be the same as the MAML-IL

    il, ppo = make_il(config, dim_s=dim_s, dim_a=dim_a, dim_cnt=dim_cnt, cnt_limit=cnt_limit)
    # demo_sa_array = tuple((s[:, :-dim_cnt].to(il.device), a.to(il.device)) for s, a, r in demo)

    demo_sa_array = tuple((s[:, :-dim_cnt].to(il.device), a.to(il.device)) for s, a, r in demo)
    assert demo_sa_array[0][0].shape[1] == dim_s - dim_cnt
    # note that we have eliminated the context embedding in the expert demonstrations

    if config.algo == 'PEMIRL':
        n_sample = config.context_repeat_num * n_sample
        sampling_agent = Sampler(seed, env, il.policy, is_expert=False, repeat_num=config.context_repeat_num, task_list=test_contexts)
    else:
        sampling_agent = Sampler(seed, env, il.policy, is_expert=False, task_list=test_contexts)
    # the true task info is not available by setting is_expert as False
    sample_sxar, sample_r, sample_r_max= sample_batch(il, sampling_agent, n_sample, demo_sa_array)

    # v_l, cs_demo = validate(il.policy, [(tr[0], tr[-2]) for tr in demo_sxar])
    info_dict, cs_sample = reward_validate(sampling_agent, il.policy, do_print=True)
    logger.log_test_info(info_dict, 0)
    print(f"init: r-sample-avg={sample_r}; {msg}")

    for i in range(n_epoch):
        sample_sxar, sample_r, sample_r_max = sample_batch(il, sampling_agent, n_sample, demo_sa_array) # n_sample is too big
        if i % 3 == 0:
            train_d(il, sample_sxar, demo_sa_array, i)
        # factor_lr = lr_factor_func(i, 1000., 1., 0.0001) # not commented by me
        sample_sxar = il.get_il_reward(sample_sxar)
        train_g(ppo, sample_sxar, factor_lr=1.)

        if (i + 1) % config.log_interval == 0:
            # v_l, cs_demo = validate(il.policy, [(tr[0], tr[-2]) for tr in demo_sxar])
            # logger.log_test("expert_logp", v_l, i)
            info_dict, cs_sample = reward_validate(sampling_agent, il.policy, do_print=True)

            if (i + 1) % (100) == 0:
                torch.save(il.state_dict(), save_name_f(i))
                torch.save(ppo.state_dict(), save_name_ppo_f(i))
            logger.log_test_info(info_dict, i)
            print(f"{i}: r-sample-avg={sample_r}, r-sample-max={sample_r_max}; {msg}")
        else:
            print(f"{i}: r-sample-avg={sample_r}, r-sample-max={sample_r_max}; {msg}")

        logger.log_train("r-sample-avg", sample_r, i)
        logger.log_train("r-sample-max", sample_r_max, i)
        logger.flush()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    arg = ARGConfig()
    arg.add_arg("env_type", "mujoco", "Environment type, can be [mujoco, ...]")
    arg.add_arg("env_name", "HalfCheetahVel-v0", "Environment name")
    arg.add_arg("algo", "SMILE", "which algorithm to use, can be [PEMIRL, SMILE, ...]")
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

    print(f">>>> Training {config.algo} using {config.env_name} environment on {config.device}")

    learn(config, msg=config.tag)