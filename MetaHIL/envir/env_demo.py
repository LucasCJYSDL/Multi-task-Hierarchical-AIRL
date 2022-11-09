import matplotlib.pyplot as plt
import gym
import time
from envir import mujoco_maze

def main():
    env_name = 'PointCell-v0'
    env = gym.make(env_name)
    for _ in range(5):
        cnt = env.sample_context()
        print("6: ", cnt)
        env.apply_context(cnt, is_expert=True)
    obs = env.reset()
    print("2: ", obs, obs.shape)
    for _ in range(1000):
        # action = env.action_space.sample()
        action = env.get_expert_action(obs)
        # render_obs = env.render('rgb_array', width=1000, height=1000, camera_name='add_cam')
        # plt.imsave(env_name+'.png', render_obs)
        # break
        env.render()
        time.sleep(0.02)
        obs, r, done, info = env.step(action)
        print("3: ", obs, r, done)
        if done:
            break

if __name__ == '__main__':
    main()