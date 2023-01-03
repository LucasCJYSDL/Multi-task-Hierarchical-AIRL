import torch, os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def draw_vel_change(exp_path):
    samples = torch.load(exp_path)
    data_frame_list = []
    # visual_dict = {2: 0, 6: 9, 7: 8, 9: 8, 10: 1, 11: 5}
    visual_dict = {3: 8, 5: 7, 7: 2, 8: 3, 9: 7, 11: 9}
    for task_id in visual_dict:
        traj_list = samples[task_id]
        if 'demos' not in traj_list:
            continue
        data_frame = pd.DataFrame()
        mov_max_agent = MovAvg(mode='avg')
        traj_id = visual_dict[task_id]
        s_array, c_array, _, _ = traj_list['demos'][traj_id]
        s_array = s_array.cpu().numpy()
        c_array = c_array.cpu().numpy()
        print(s_array.shape, c_array.shape)
        # print(s_array[:, 0], c_array)
        for i in range(len(s_array)):
            cur_value = mov_max_agent.update(s_array[i][0])
            # if s_array[i][0] < 0.5 and s_array[i][0] > 0 and c_array[i+1][0] == 1:
            #     cur_value = s_array[i][0]
            data_frame = data_frame.append({'Skill': c_array[i+1][0], 'Step': i, 'Velocity': cur_value},
                                           ignore_index=True)
        # print(data_frame)
        data_frame_list.append(data_frame)

    sns.set_theme(style="darkgrid")
    # sns.set(style="white", palette="muted", color_codes=True)

    # Set up the matplotlib figure
    sns.set(font_scale=1.5)
    f, axes = plt.subplots(2, 3, figsize=(13, 5), sharey=True)
    sns.despine(left=True)

    for c_id in range(6):
        row = c_id // 3
        col = c_id % 3
        sns.scatterplot(x="Step", y="Velocity", hue='Skill', data=data_frame_list[c_id], legend=False, ax=axes[row, col])
        axes[row, col].set_title('Context #{}'.format(c_id))

    plt.setp(axes, xticks=[0, 50, 100])
    plt.tight_layout()
    # f.text(0.5, 0, 'Orientation', ha='center')
    # f.text(0.04, 0.5, 'Density', va='center', rotation='vertical')
    plt.show()


def draw_vel_change_hier_policy(samples, episode_id, unique_token, time_token):
    data_frame_list = []

    for traj in samples:

        data_frame = pd.DataFrame()
        mov_max_agent = MovAvg(mode='avg')
        s_array, c_array, _, _ = traj
        s_array = s_array.cpu().numpy()
        c_array = c_array.cpu().numpy()
        print(s_array.shape, c_array.shape)
        # print(s_array[:, 0], c_array)
        for i in range(len(s_array)):
            cur_value = mov_max_agent.update(s_array[i][0])
            # if s_array[i][0] < 0.5 and s_array[i][0] > 0 and c_array[i+1][0] == 1:
            #     cur_value = s_array[i][0]
            data_frame = data_frame.append({'Skill': c_array[i+1][0], 'Step': i, 'Velocity': cur_value},
                                           ignore_index=True)
        # print(data_frame)
        data_frame_list.append(data_frame)

    sns.set_theme(style="darkgrid")
    # sns.set(style="white", palette="muted", color_codes=True)

    # Set up the matplotlib figure
    sns.set(font_scale=1.5)
    f, axes = plt.subplots(2, 6, figsize=(25, 5), sharey=True)
    sns.despine(left=True)

    for c_id in range(12):
        row = c_id // 6
        col = c_id % 6
        sns.scatterplot(x="Step", y="Velocity", hue='Skill', data=data_frame_list[c_id+1], legend=False, ax=axes[row, col])
        axes[row, col].set_title('Context #{}'.format(c_id+1))

    plt.setp(axes, xticks=[0, 50, 100])
    plt.tight_layout()
    # f.text(0.5, 0, 'Orientation', ha='center')
    # f.text(0.04, 0.5, 'Density', va='center', rotation='vertical')
    save_path = "./visual_result/" + time_token
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig_path = os.path.join(save_path, str(episode_id) + '_' + unique_token + '.png')
    plt.savefig(fig_path)
    plt.close()


class MovAvg(object):

    def __init__(self, window_size=6, mode='avg'):
        self.window_size = window_size
        self.data_queue = []
        self.mode = mode

    def set_window_size(self, num):
        self.window_size = num

    def clear_queue(self):
        self.data_queue = []

    def update(self, data):
        if len(self.data_queue) == self.window_size:
            del self.data_queue[0]
        self.data_queue.append(data)
        if self.mode == 'avg':
            return sum(self.data_queue) / len(self.data_queue)
        else:
            return max(self.data_queue)

if __name__ == '__main__':
    draw_vel_change('offline_plot.torch')