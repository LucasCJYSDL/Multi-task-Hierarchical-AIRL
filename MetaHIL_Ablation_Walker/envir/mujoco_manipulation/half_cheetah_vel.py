import random
import numpy as np
from gym.envs.mujoco import HalfCheetahEnv


class HalfCheetahVelEnv(HalfCheetahEnv):
    def __init__(self):
        self._context_dim = 1
        self._context_limit = 3.0
        self.context = np.array([0.0]) # temporary
        self.is_expert = False
        self.goal_vel = np.array([3.0]) # temporary
        self.sub_goal_vel_list = [np.array([3.0])] # temporary
        self.sub_goal_vel_idx = 0
        super(HalfCheetahVelEnv, self).__init__()

    def get_context_dim(self):
        return self._context_dim

    def get_context_limit(self):
        return self._context_limit

    def sample_context(self) -> np.ndarray: # truncated standard normal distribution
        vel_context = np.random.normal(loc=0.0, scale=1.0)
        # vel_context = self._context_limit * np.tanh(vel_context/self._context_limit) # a more gentle truncation and invertiable
        # TODO
        if vel_context < -self._context_limit:
            vel_context = -self._context_limit
        if vel_context > self._context_limit:
            vel_context = self._context_limit

        return np.array([vel_context])

    def apply_context(self, context_rv: np.ndarray, is_expert: bool):
        assert len(context_rv) == self._context_dim
        self.context = context_rv
        self.is_expert = is_expert
        # get the target velocity based on the context
        goal_vel = context_rv[0] + 5.0
        self.goal_vel = np.array([goal_vel])
        self.sub_goal_vel_list = [np.array([goal_vel/2.]), np.array([0.0]), np.array([goal_vel])]
        # print("8: ", self.sub_goal_vel_list)

    def reset(self):
        super(HalfCheetahVelEnv, self).reset()
        self.sub_goal_vel_idx = 0

        return self._get_obs()

    def _get_obs(self):
        obs = np.concatenate([self.sim.data.qpos.flat[1:], # they don't include the x location
                              self.sim.data.qvel.flat,
                              self.get_body_com("torso").flat]).astype(np.float32).flatten()

        if self.is_expert:
            obs_ext = np.concatenate([obs, self.sub_goal_vel_list[self.sub_goal_vel_idx]])
        else:
            obs_ext = np.concatenate([obs, self.context])
        return obs_ext

    def seed(self, seed_idx: int=None):
        super(HalfCheetahVelEnv, self).seed(seed_idx)
        self.action_space.np_random.seed(seed_idx)
        random.seed(seed_idx)
        np.random.seed(seed_idx)

    def viewer_setup(self):
        camera_id = self.model.camera_name2id('track')
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = camera_id
        self.viewer.cam.distance = self.model.stat.extent * 0.35
        # Hide the overlay
        self.viewer._hide_overlay = True

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        forward_vel = (xposafter - xposbefore) / self.dt

        cur_sub_goal_vel = self.sub_goal_vel_list[self.sub_goal_vel_idx][0]
        goal_bonus = 0.0
        vel_diff = abs(forward_vel - cur_sub_goal_vel)
        done = False
        if vel_diff <= 1e-1: # important parameter
            print("Achieve Target Velocity {}!".format(self.sub_goal_vel_idx))
            if self.sub_goal_vel_idx == len(self.sub_goal_vel_list) - 1:
                done = True
                goal_bonus += 100.0
                print("Great Success!!!")
            self.sub_goal_vel_idx += 1
            if self.sub_goal_vel_idx > len(self.sub_goal_vel_list) - 1:
                self.sub_goal_vel_idx = len(self.sub_goal_vel_list) - 1
            goal_bonus += 100.0

        forward_reward = -1.0 * vel_diff
        ctrl_cost = 0.05 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = 0.1 * (forward_reward - ctrl_cost) + goal_bonus

        # final_vel_diff = abs(forward_vel - self.goal_vel[0])
        # if final_vel_diff > 1e-3:
        #     done = False
        # else:
        #     done = True
        #     print("Great Success!!!")

        info = dict(reward_forward=forward_reward, reward_ctrl=-ctrl_cost, goal_bouns=goal_bonus)
        # print(info)
        return observation, reward, done, info



