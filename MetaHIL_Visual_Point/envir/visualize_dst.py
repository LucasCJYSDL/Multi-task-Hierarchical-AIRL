import os
import torch
import numpy as np
from typing import Tuple, List, Dict
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class MazeCell(Enum):
    ROBOT = -1
    EMPTY = 0
    BLOCK = 1

E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
SCALE = 8.0
MAZE = {
        'Cell': [
                    [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
                    [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
                    [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
                    [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
                    [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
                    [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
                    [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
                    [B, E, E, E, E, E, E, R, E, E, E, E, E, E, B],
                    [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
                    [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
                    [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
                    [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
                    [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
                    [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
                    [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B]]
        }

def _get_lower_left_loc(i_idx: int, j_idx: int, origin_idx: Tuple):
    ori_i = origin_idx[0]
    ori_j = origin_idx[1]

    y = (i_idx - ori_i) * SCALE + 0.0 - SCALE / 2.0
    x = (j_idx - ori_j) * SCALE + 0.0 - SCALE / 2.0

    return (x, y)


def draw_traj(env_id: str, option_num: int, trajectory_list: Dict, unique_token: str, episode_id, time_token='0'):

    maze = MAZE['Cell']
    origin_idx = (7, 7)

    maze_size = len(maze)

    # preparation
    cmap = plt.get_cmap('viridis', option_num)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # # draw the trajectories
    for c_id in range(option_num):
        c_traj_list = trajectory_list[c_id]
        for traj_id in range(len(c_traj_list)):
            s = [10.0 for _ in range(len(c_traj_list[traj_id]['x']))]
            ax.scatter(c_traj_list[traj_id]['x'], c_traj_list[traj_id]['y'], color=cmap(c_id), alpha=0.3, s=s)
            # ax.plot(c_traj_list[traj_id]['x'][0], c_traj_list[traj_id]['y'][0], marker='o', markersize=5, color='red',
            #         zorder=11, alpha=0.5)
            # ax.plot(c_traj_list[traj_id]['x'][1:], c_traj_list[traj_id]['y'][1:], label="Option #{}".format(c_id),
            #         color=cmap(c_id), alpha=1.0, linewidth=2, zorder=10)
    #         for i in range(len(c_traj_list[traj_id]['x'])-1):
    #             if (c_traj_list[traj_id]['x'][i] - c_traj_list[traj_id]['x'][i+1])**2 + (c_traj_list[traj_id]['y'][i] - c_traj_list[traj_id]['y'][i+1])**2 <= 200:
    #                 ax.plot([c_traj_list[traj_id]['x'][i], c_traj_list[traj_id]['x'][i+1]], [c_traj_list[traj_id]['y'][i], c_traj_list[traj_id]['y'][i+1]], color=cmap(c_id), alpha=0.6, linewidth=7.0)
    # # draw the maze
    for i in range(maze_size):
        for j in range(maze_size):
            if maze[i][j] == B:
                loc = _get_lower_left_loc(i, j, origin_idx)
                ax.add_patch(Rectangle(loc, SCALE, SCALE, edgecolor='gray', facecolor='gray', fill=True, alpha=0.5))

    # eliminate the redundant parts
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for p in ["left", "right", "top", "bottom"]:
        ax.spines[p].set_visible(False)

    # plt.show()
    save_path = "./visual_result/"+time_token
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig_path = os.path.join(save_path, str(episode_id) + '.png')
    plt.savefig(fig_path)
    plt.close()


def plot(exp_path):
    samples = torch.load(exp_path)
    tot_num = 10
    cur_num = 0
    traj_array = {0: [], 1: [], 2: [], 3: [], 4: []}
    traj_id = 0
    for task_id in samples:
        traj_list = samples[task_id]
        if 'demos' not in traj_list:
            continue
        for traj_id in range(90):
            s_array, c_array, a_array, r_array = traj_list['demos'][traj_id]
            s_array = s_array.cpu().numpy()
            c_array = c_array.cpu().numpy()
            r_array = r_array.cpu().numpy()
            if np.sum(r_array) < 0:
                continue
            print(task_id, s_array.shape, c_array.shape)
            temp_array = {0: {'x': [], 'y': []}, 1: {'x': [], 'y': []}, 2: {'x': [], 'y': []}, 3: {'x': [], 'y': []}, 4: {'x': [], 'y': []}}
            for i in range(len(s_array)):
                tmp_option = int(c_array[i+1])
                tmp_x = s_array[i][0]
                tmp_y = s_array[i][1]
                temp_array[tmp_option]['x'].append(tmp_x)
                temp_array[tmp_option]['y'].append(tmp_y)

            for o_id in traj_array:
                traj_array[o_id].append(temp_array[o_id])

        cur_num += 1
        if cur_num > tot_num:
            break

    draw_traj("PointCell", option_num=4, trajectory_list=traj_array, unique_token='PointCell', episode_id='off', time_token="line")


def plot_hier_policy(samples, training_episode, time_token, unique_token):
    tot_num = 10
    cur_num = 0
    traj_array = {0: [], 1: [], 2: [], 3: [], 4: []}
    for traj in samples:

        s_array, c_array, _, _ = traj
        s_array = s_array.cpu().numpy()
        c_array = c_array.cpu().numpy()
        # print(s_array.shape, c_array.shape)
        temp_array = {0: {'x': [], 'y': []}, 1: {'x': [], 'y': []}, 2: {'x': [], 'y': []}, 3: {'x': [], 'y': []}, 4: {'x': [], 'y': []}}
        for i in range(len(s_array)):
            tmp_option = int(c_array[i + 1])
            tmp_x = s_array[i][0]
            tmp_y = s_array[i][1]
            temp_array[tmp_option]['x'].append(tmp_x / 4.0)
            temp_array[tmp_option]['y'].append(tmp_y / 4.0)

        for o_id in traj_array:
            traj_array[o_id].append(temp_array[o_id])

        cur_num += 1
        if cur_num > tot_num:
            break

    draw_traj("PointCell", option_num=4, trajectory_list=traj_array, unique_token='PointCell',
              episode_id=str(training_episode) + '_' + unique_token, time_token=time_token)


if __name__ == '__main__':
    plot('../offline_plot.torch')