#!/usr/bin/env python3
import os
import torch
import random
import numpy as np
import torch.multiprocessing as multiprocessing
from utils.common_utils import get_dirs, set_seed
from utils.logger import Logger
from utils.config import ARGConfig, Config
from default_config import mujoco_config
from model.maml_learner import MAMLLearner

def preprocess(data_set, context_dim, env_name):
    # we need to replace the true task info (e.g., goal location) in the expert data with the context
    # since the learned policy will not be provided the true task info
    num_traj = 0
    num_trans = 0
    for task_idx in data_set:
        temp_context = torch.tensor(data_set[task_idx]['context'], dtype=torch.float32).unsqueeze(0)
        temp_context = torch.zeros_like(temp_context)
        temp_traj_list = data_set[task_idx]['demos']
        num_traj += len(temp_traj_list)
        for traj_idx in range(len(temp_traj_list)):
            s, a, r = temp_traj_list[traj_idx]
            s = s[:, :-context_dim] # eliminate the true task info # TODO: only do this elimination
            context_sup = temp_context.repeat(s.shape[0], 1)
            s = torch.cat([s, context_sup], dim=1)
            temp_traj_list[traj_idx] = (s, a, r)
            num_trans += int(s.shape[0])

    return float(num_trans)/float(num_traj)

def learn(config: Config, msg="default"):
    ## prepare
    env_type = config.env_type
    if env_type == "mujoco":
        from envir.mujoco_env import MujocoEnv as Env, get_demo
    else:
        raise ValueError(f"Unknown env type {env_type}")

    n_traj = config.n_traj
    n_epoch = config.n_epoch
    seed = config.seed
    env_name = config.env_name

    set_seed(seed)
    log_dir, save_dir, train_set_name, test_set_name, _ = get_dirs(seed, config.algo, env_type, env_name, msg)
    with open(os.path.join(save_dir, "config.log"), 'w') as f:
        f.write(str(config))  # important for reproducing and visualisaton
    logger = Logger(log_dir)  # tensorboard
    save_name_f = lambda i: os.path.join(save_dir, f"{i}.torch")

    env = Env(env_name)
    env.init()
    dim_s, dim_a = env.state_action_size()
    dim_cnt, cnt_limit = env.get_context_info()
    print("The dimension info of the environment: name:{}, dim_s:{}, dim_a:{}, context_dim:{}, context_limit:{}.".format(
            env_name, dim_s, dim_a, dim_cnt, cnt_limit))

    # the demonstration does contain task labels!!!
    train_set, test_set = get_demo(train_set_name, test_set_name, n_traj=n_traj, task_specific=True)
    train_average_horizon = preprocess(train_set, dim_cnt, env_name)
    test_average_horizon = preprocess(test_set, dim_cnt, env_name)
    # print(train_set[0], test_set[10])
    samples_per_update = train_average_horizon * config.task_batch_size * 2
    plot_factor = float(samples_per_update) / (config.n_sample * 2)

    # print(train_average_horizon, samples_per_update, plot_factor)

    train_task_list = list(train_set.keys())
    test_task_list = list(test_set.keys())
    # the evaluation in only on the tasks contained in the test set, which is controlled to be the same as the other algorithms

    learner = MAMLLearner(config, dim_s, dim_a)

    # the train loop
    for i in range(n_epoch*100): # more training epochs than other algorithms
        cur_task_list = random.sample(train_task_list, config.task_batch_size)
        # for each task, we sample 2 trajs: 1 for inner train and the other for inner validation
        demos = []
        for cur_task in cur_task_list:
            cur_trajs = train_set[cur_task]['demos']
            cur_demos = random.sample(cur_trajs, k=2)
            demos.extend(cur_demos)
        demos = [(s.to(learner.device), a.to(learner.device)) for s, a, r in demos]

        loss = learner.step(demos)
        print(f"{i}: training_loss={loss}")

        if (i + 1) % 200 == 0:
            assert len(test_task_list) > 8
            cur_test_task_list = random.sample(test_task_list, 8) # other algorithms also use 8 trajs for evaluation
            test_task_batch = []
            for test_task in cur_test_task_list:
                test_demos = random.sample(test_set[test_task]['demos'], k=1) # TODO: more than 1
                test_demos = [(s.to(learner.device), a.to(learner.device)) for s, a, r in test_demos]
                test_task_batch.append({'context': test_set[test_task]['context'], 'demos': test_demos})

            info_dict = learner.eval(env, test_task_batch)
            logger.log_test_info(info_dict, int(i * plot_factor))
            print(f"R: [ {info_dict['r-min']:.02f} ~ {info_dict['r-max']:.02f}, avg: {info_dict['r-avg']:.02f} ], "
                  f"L: [ {info_dict['step-min']} ~ {info_dict['step-max']} ]")

            if (i + 1) % (2000) == 0:
                torch.save(learner.policy.state_dict(), save_name_f(i))

        logger.log_train("train_loss", loss, int(i * plot_factor))
        logger.flush()


if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn')

    arg = ARGConfig()
    arg.add_arg("env_type", "mujoco", "Environment type, can be [mujoco, ...]")
    arg.add_arg("env_name", "WalkerRandParams-v0", "Environment name")
    arg.add_arg("algo", "MAML_IL", "which algorithm to use, can be [MAML_IL, ...]")
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
        config.MAML_policy_hidden_dim = 128

    elif config.env_name.startswith("Kitchen"):
        # config.n_sample = 512
        config.hidden_option = (256, 256)
        config.hidden_policy = (256, 256)
        config.hidden_critic = (256, 256)
        config.MAML_policy_hidden_dim = 256

    print(f">>>> Training {config.algo} using {config.env_name} environment on {config.device}")

    learn(config, msg=config.tag)

