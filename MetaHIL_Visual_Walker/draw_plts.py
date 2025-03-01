import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

def plot():
    sns.set_theme(style="darkgrid")
    
    # data files
    common_dir = './log_saved'
    file_dir = {'MH-AIRL': [17048, 17229, 17306], 'MH-AIRL-no-cnt': [16074, 16223, 16680]}

    data_frame = pd.DataFrame()
    for alg, dir_name_list in file_dir.items():
        print("Processing ", alg)
        temp_array = []
        for dir_name in dir_name_list:
            csv_file_name = str(dir_name) + '.csv'
            csv_file_dir = os.path.join(common_dir, alg, csv_file_name)
            print("Loading from: ", alg, csv_file_dir)

            temp_df = pd.read_csv(csv_file_dir)
            temp_step = np.array(temp_df['Step'])
            temp_value = np.array(temp_df['Value'])

            new_temp_step = np.zeros((len(temp_step) + 1,))
            new_temp_value = np.zeros((len(temp_value) + 1,))
            new_temp_step[1:] = temp_step
            new_temp_value[1:] = temp_value
            temp_step = new_temp_step
            temp_value = new_temp_value

            print("Average rwd across the episodes: ", np.mean(temp_value))
            temp_len = len(temp_step)
            
            mov_max_agent = MovAvg(mode='avg')
            for i in range(temp_len):
                if temp_step[i] > 7600:
                    break
                cur_value = mov_max_agent.update(temp_value[i])
                temp_value[i] = cur_value

            convergent_performance = []
            mov_avg_agent = MovAvg()
            for i in range(temp_len):
                if temp_step[i] > 7600:
                    break
                cur_value = mov_avg_agent.update(temp_value[i])
                # if temp_step[i] >= 1000: # 1800, 1000
                #     temp_array.append(cur_value)
                data_frame = data_frame.append({'algorithm': alg, 'Step': temp_step[i] * 4096, 'Reward': cur_value}, ignore_index=True)
                if temp_step[i] > 7000:
                    convergent_performance.append(cur_value)
            temp_array.append(np.mean(convergent_performance))

        print("Final results for {}: ".format(alg), np.mean(temp_array), np.std(temp_array))

    # expert value: 168.46 for room
    for i in range(temp_len):
        if temp_step[i] > 7600:
            break
        data_frame = data_frame.append({'algorithm': 'Expert', 'Step': temp_step[i] * 4096, 'Reward': 399.95}, ignore_index=True)
        # print(np.array(temp_array).mean(), np.array(temp_array).std())
    sns.set(font_scale=1.5)
    # pal = sns.xkcd_palette((['red', 'blue', 'green', 'orange', 'yellow']))
    pal = sns.xkcd_palette((['red', 'blue', 'yellow']))
    g = sns.relplot(x="Step", y="Reward", hue='algorithm', kind="line", ci="sd", data=data_frame, legend='brief', palette=pal)
    g.set(ylim=(270, 410))
    leg = g._legend
    leg.set_bbox_to_anchor([0.65, 0.38])  # coordinates of lower left of bounding box [0.75, 0.35], [0.55, 0.75], [0.4, 0.7]
    # g = sns.relplot(x="training step", y="mean reward", hue='algorithm', kind="line", data=data_frame)
    g.fig.set_size_inches(15, 6)
    plt.savefig(common_dir + '/' + 'Reward.png')


class MovAvg(object):

    def __init__(self, window_size=100, mode='avg'):
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
    plot()


