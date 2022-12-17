import matplotlib.pyplot as plt
import gym
import time
from envir import mujoco_maze, mujoco_manipulation, d4rl_env

def main():
    # env_name = 'PointCell-v1'
    # env_name = 'AntCell-v1'
    # env_name = "HalfCheetahVel-v0"
    # env_name = "WalkerRandParams-v0"
    # env_name = 'KitchenSeqEnv-v0'
    # env_name = 'KitchenSeqEnv-v5'
    # env_name = 'KitchenSeqEnv-v6'
    env_name = 'KitchenMetaEnv-v0'
    env = gym.make(env_name)
    for _ in range(5):
        cnt = env.sample_context()
        print("6: ", cnt)
        env.apply_context(cnt, is_expert=False)

    obs = env.reset()
    print("2: ", obs, obs.shape)
    action_list = []
    ii = 0
    # for i in range(1000):
    #     action = env.action_space.sample()
    #     print("7: ", action, action.shape)
    #     # action = env.get_expert_action(obs)
    #     # render_obs = env.render('rgb_array', width=1000, height=1000, camera_name='add_cam')
    #     # plt.imsave(env_name+'.png', render_obs)
    #     # break
    #     env.render()
    #     # time.sleep(0.02)
    #     action_list.append(action)
    #     ii = ii+1
    #     obs, r, done, info = env.step(action)
    #     print("3: ", obs, r, done, i)
    #     if done:
    #         break
    # print(len(action_list), ii)
    # env = gym.make(env_name)
    # for _ in range(5):
    #     cnt = env.sample_context()
    #     # print("6: ", cnt)
    #     env.apply_context(cnt, is_expert=True)

if __name__ == '__main__':
    main()