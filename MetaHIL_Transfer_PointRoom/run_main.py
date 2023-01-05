#!/usr/bin/env python3

import os
import torch
from typing import Union
import torch.multiprocessing as multiprocessing
from model.option_ppo import PPO, OptionPPO
from model.MHA_option_ppo import MHAOptionPPO
from model.hierarchical_ppo import HierarchicalPPO, MHAOptionPolicy
from model.option_policy import OptionPolicy, Policy
from sampler import Sampler
from utils.common_utils import lr_factor_func, get_dirs, reward_validate, set_seed
from utils.logger import Logger
from utils.config import Config, ARGConfig
from default_config import mujoco_config


def sample_batch(policy: Union[OptionPolicy, Policy], agent, n_step):
    sample = agent.collect(policy.state_dict(), n_step, fixed=False)
    rsum = sum([sxar[-1].sum().item() for sxar in sample]) / len(sample)
    return sample, rsum


def learn(config: Config, msg="default"):
    env_type = config.env_type
    if env_type == "mujoco":
        from envir.mujoco_env import MujocoEnv as Env
    else: # Meta World tasks are on the way
        raise NotImplementedError

    use_option = config.use_option
    env_name = config.env_name
    n_sample = config.n_sample
    n_epoch = config.n_epoch
    seed = config.seed
    set_seed(seed)

    log_dir, save_dir, train_set_name, test_set_name, _ = get_dirs(seed, config.algo, env_type, env_name, msg)
    with open(os.path.join(save_dir, "config.log"), 'w') as f:
        f.write(str(config))
    logger = Logger(log_dir)

    save_name_f = lambda i: os.path.join(save_dir, f"{i}.torch")

    env = Env(env_name)
    dim_s, dim_a = env.state_action_size()
    print("The dimension info of the environment: ", dim_s, dim_a)

    policy = MHAOptionPolicy(config, dim_s=dim_s, dim_a=dim_a, ori_dim_s=dim_s)
    ppo = HierarchicalPPO(config, policy, dim_s)
    # ppo = MHAOptionPPO(config, policy, dim_s)

    # load the pretrained model
    model_dict = policy.state_dict()
    pretrained_model = torch.load(config.policy_model, map_location=config.device)
    pretrained_dict = pretrained_model
    # print("pre: ", model_dict)
    new_dict = {}
    for k, v in model_dict.items():
        # print(k)
        if ('embed_option' in k) or ('act_doe' in k):
            for pk, pv in pretrained_dict.items():
                if k in pk:
                    print("Essential: ", k, pk)
                    new_dict[k] = pv
    model_dict.update(new_dict)
    policy.load_state_dict(model_dict)
    # print("post: ", policy.state_dict())

    # dict_name = list(policy.state_dict())
    # for i, p in enumerate(dict_name):
    #     print(i, p)
    # freeze the parameters related to the low-level policy
    # for i, p in enumerate(policy.parameters()):
    #     # print(p)
    #     if i == 0 or i >= 25:
    #         p.requires_grad = False

    # for i, p in enumerate(policy.parameters()):
    #     print(p)

    sampling_agent = Sampler(seed, env, policy, is_expert=False)

    for i in range(n_epoch):
        sample_sxar, sample_r = sample_batch(policy, sampling_agent, n_sample)
        lr_mult = lr_factor_func(i, n_epoch, 1., 0.)
        if sample_r > 1200 and i > 50:
            config.train_low_policy = False
        ppo.step(sample_sxar, lr_mult=lr_mult, train_policy=config.train_low_policy)
        if (i + 1) % 50 == 0:
            info_dict, cs_sample = reward_validate(sampling_agent, policy) # testing performance
            torch.save(policy.state_dict(), save_name_f(i))
            logger.log_test_info(info_dict, i)
        print(f"{i}: r-sample-avg={sample_r} ; {msg}")
        logger.log_train("r-sample-avg", sample_r, i) # a very important metric
        logger.flush()


if __name__ == "__main__":
    # learn the expert policy/option-policy based on the environment rewards using PPO
    multiprocessing.set_start_method('spawn')

    arg = ARGConfig()
    arg.add_arg("env_type", "mujoco", "Environment type, can be [mujoco, ...]")
    # [PointCell-v1, AntCell-v1, HalfCheetahVel-v0, WalkerRandParams-v0, KitchenSeqEnv-v[0~6]]
    arg.add_arg("env_name", "PointRoom-v1", "Environment name")
    arg.add_arg("algo", "hier_ppo_tran", "Environment type, can be [hier_ppo_tran]")
    arg.add_arg("device", "cuda:0", "Computing device")
    arg.add_arg("tag", "default", "Experiment tag")
    arg.add_arg("n_epoch", 2000, "Number of training epochs")
    arg.add_arg("seed", 0, "Random seed")
    arg.add_arg("train_low_policy", True, "Random seed")
    arg.parser()

    assert arg.env_type == "mujoco"
    config = mujoco_config
    config.update(arg)

    if config.env_name.startswith("Ant") or config.env_name.startswith("Walker"):
        config.hidden_policy = (128, 128)
        config.hidden_critic = (128, 128)

    elif config.env_name.startswith("Kitchen"):
        # config.n_sample = 512
        config.hidden_policy = (256, 256)
        config.hidden_critic = (256, 256)

    print(f"Training this env with larger policy network size :{config.hidden_policy}")


    config.use_option = True
    config.use_c_in_discriminator = False # in fact, there are no discriminators
    config.use_d_info_gail = False
    config.use_vae = False
    config.train_option = True
    if config.algo == 'ppo':
        config.use_option = False
        config.train_option = False
    config.policy_model = "./option_model/624.torch"
    config.critic_model = "./option_model/624_critic.torch"
    print(f">>>> Training {config.algo} on {config.env_name} environment, on {config.device}")
    learn(config, msg=config.tag)
