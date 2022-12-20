#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import click
import glob
import pickle
import numpy as np
from envir import d4rl_env
from parse_mjl import parse_mjl_logs, viz_parsed_mjl_logs
import time as timer
import gym
import torch

# headless renderer
render_buffer = []  # rendering buffer


def viewer(env,
           mode='initialize',
           filename='video',
           frame_size=(640, 480),
           camera_id=0,
           render=None):
    if render == 'onscreen':
        env.mj_render()

    elif render == 'offscreen':
        global render_buffer
        if mode == 'initialize':
            render_buffer = []
            mode = 'render'

        if mode == 'render':
            # curr_frame = env.render(mode='rgb_array')
            curr_frame = []
            render_buffer.append(curr_frame)

        # if mode == 'save':
        #     skvideo.io.vwrite(filename, np.asarray(render_buffer))
        #     print("\noffscreen buffer saved", filename)

    elif render == 'None':
        pass

    else:
        print("unknown render: ", render)


# view demos (physics ignored)
def render_demos(env, data, filename='demo_rendering.mp4', render=None):
    FPS = 30
    render_skip = max(1, round(1. / (FPS * env.sim.model.opt.timestep * env.frame_skip)))
    t0 = timer.time()

    viewer(env, mode='initialize', render=render)
    for i_frame in range(data['ctrl'].shape[0]):
        env.sim.data.qpos[:] = data['qpos'][i_frame].copy()
        env.sim.data.qvel[:] = data['qvel'][i_frame].copy()
        env.sim.forward()
        if i_frame % render_skip == 0:
            viewer(env, mode='render', render=render)
            print(i_frame, end=', ', flush=True)

    # viewer(env, mode='save', filename=filename, render=render)
    print("time taken = %f" % (timer.time() - t0))


# playback demos and get data(physics respected)
def gather_training_data(env, data, filename='demo_playback.mp4', render=None):
    env = env.env
    FPS = 30
    render_skip = max(1, round(1. / (FPS * env.sim.model.opt.timestep * env.frame_skip)))
    t0 = timer.time()

    # initialize
    env.reset()
    init_qpos = data['qpos'][0].copy()
    init_qvel = data['qvel'][0].copy()
    act_mid = env.act_mid
    act_rng = env.act_amp

    # prepare env
    env.sim.data.qpos[:] = init_qpos
    env.sim.data.qvel[:] = init_qvel
    env.sim.forward()
    viewer(env, mode='initialize', render=render)

    # step the env and gather data
    path_obs = None
    for i_frame in range(data['ctrl'].shape[0] - 1):
        # Reset every time step
        # if i_frame % 1 == 0:
        #     qp = data['qpos'][i_frame].copy()
        #     qv = data['qvel'][i_frame].copy()
        #     env.sim.data.qpos[:] = qp
        #     env.sim.data.qvel[:] = qv
        #     env.sim.forward()

        obs = env._get_obs()

        # Construct the action
        # ctrl = (data['qpos'][i_frame + 1][:9] - obs[:9]) / (env.skip * env.model.opt.timestep)
        ctrl = (data['ctrl'][i_frame] - obs[:9])/(env.skip*env.model.opt.timestep)
        act = (ctrl - act_mid) / act_rng
        act = np.clip(act, -0.999, 0.999)
        # if i_frame % 10 == 0:
        #     print(obs[:9])
        next_obs, reward, done, env_info = env.step(act)
        if path_obs is None:
            path_obs = obs
            path_act = act
            path_rwd = reward
        else:
            path_obs = np.vstack((path_obs, obs))
            path_act = np.vstack((path_act, act))
            path_rwd = np.vstack((path_rwd, reward))

        # render when needed to maintain FPS
        if i_frame % render_skip == 0:
            viewer(env, mode='render', render=render)
            print(i_frame, end=', ', flush=True)

        if done:
            break

    # finalize
    if render:
        viewer(env, mode='save', filename=filename, render=render)

    t1 = timer.time()
    print("time taken = %f" % (t1 - t0))

    # note that <init_qpos, init_qvel> are one step away from <path_obs[0], path_act[0]>
    return path_obs, path_act, path_rwd, init_qpos, init_qvel


# MAIN =========================================================
@click.command(help="parse tele-op demos")
@click.option('--env', '-e', type=str, help='gym env name', default='demo_env-v0')
@click.option(
    '--demo_dir',
    '-d',
    type=str,
    help='directory with tele-op logs',
    default='./d4rl_env/kitchen_demos_multitask/microwave_bottom_switch_slide/')
@click.option(
    '--skip',
    '-s',
    type=int,
    help='number of frames to skip (1:no skip)',
    default=40)
@click.option('--graph', '-g', type=bool, help='plot logs', default=False)
@click.option('--save_logs', '-l', type=bool, help='save logs', default=False)
@click.option(
    '--view', '-v', type=str, help='render/playback', default='playback')
@click.option(
    '--render', '-r', type=str, help='onscreen/offscreen', default='onscreen')


def main(env, demo_dir, skip, graph, save_logs, view, render):

    gym_env = gym.make(env)

    task_seq = demo_dir.split('/')[-2]
    print(task_seq)
    gym_env.set_task_elements(task_seq)

    paths = []
    print("Scanning demo_dir: " + demo_dir + "=========")
    for ind, file in enumerate(glob.glob(demo_dir + "*.mjl")):
        if ind % 3 == 0:
            render = 'onscreen'
        else:
            render = 'offscreen'
        # process logs
        print("processing: " + file, end=': ')

        data = parse_mjl_logs(file, skip) #?
        print("log duration %0.2f" % (data['time'][-1] - data['time'][0]))

        # plot logs
        if graph:
            print("plotting: " + file)
            viz_parsed_mjl_logs(data)

        # save logs
        if save_logs:
            pickle.dump(data, open(file[:-4] + ".pkl", 'wb')) # change the naming

        # render logs to video
        if view == 'render':
            render_demos(gym_env,
                         data,
                         filename=data['logName'][:-4] + '_demo_render.mp4',
                         render=render)

        # playback logs and gather data
        elif view == 'playback':
            try:
                obs, act, rwd, init_qpos, init_qvel = gather_training_data(gym_env, data,
                                                                     filename=data['logName'][:-4]+'_playback.mp4', render=render)
            except Exception as e:
                print(e)
                continue
            path = {
                'observations': obs,
                'actions': act,
                'rewards': rwd
            } #? # include all the info in an mjl file
            # print("init obs: ", obs[0])
            # print("final obs: ", obs[-1])
            # print(obs.shape, act.shape, rwd.shape, rwd)
            # print("slide: ", obs[:, 19])
            paths.append(path)
            accept = input('accept demo?')
            if accept == 'n':
                continue
            pickle.dump(path, open(demo_dir + str(ind) + ".pkl", 'wb'))
            # test_dict = pickle.load(open(demo_dir + str(ind) + ".pkl", 'rb'))
            # print(test_dict)

def pkl_to_torch(data_dir):
    env_id = 'KitchenMetaEnv-v1'
    env = gym.make(env_id)
    folder_list = []
    for folder in os.listdir(data_dir):
        folder_list.append(folder)
    folder_list.sort()
    # folder_list = folder_list[:env.get_context_dim()]
    print(folder_list)

    train_demos = {}
    test_demos = {}
    for i in range(len(folder_list)):
        task_idx = i
        cur_context = env.convert_to_context(task_idx)
        train_demos[i] = {'context': cur_context}
        test_demos[i] = {'context': cur_context} # for more challenging tasks, the test scenarios should be different from the training ones

        cur_folder = folder_list[i]
        demo_dir = data_dir + '/' + cur_folder + '/'
        train_trajs = []
        test_trajs = []
        for ind, file in enumerate(glob.glob(demo_dir + "*.pkl")):
            # print(file)
            test_dict = pickle.load(open(file, 'rb'))
            obs = test_dict['observations']
            act = test_dict['actions']
            rwd = test_dict['rewards']
            obs = obs[:, :9]
            cur_ext = np.array([cur_context for _ in range(obs.shape[0])]) # we use the real task id as the true context
            obs = np.concatenate([obs, cur_ext], axis=-1)

            print("1: ", obs.shape, act.shape, rwd.shape, rwd.sum())
            # print("2: ", obs[0], obs[1])
            # each directory contains at least 15 trajs
            if ind < 10:
                train_trajs.append((torch.tensor(obs, dtype=torch.float32), torch.tensor(act, dtype=torch.float32), torch.tensor(rwd, dtype=torch.float32)))
            elif ind < 15:
                test_trajs.append((torch.tensor(obs, dtype=torch.float32), torch.tensor(act, dtype=torch.float32), torch.tensor(rwd, dtype=torch.float32)))
            else:
                print("Enough!")
                break

        train_demos[i]['demos'] = train_trajs
        test_demos[i]['demos'] = test_trajs

    torch.save(train_demos, env_id + '_sample_train.torch')
    torch.save(test_demos, env_id + '_sample_test.torch')

if __name__ == '__main__':
    # main()
    pkl_to_torch('./d4rl_env/kitchen_demos_multitask')