import random
import numpy as np
from gym import utils
from gym.envs.mujoco import MujocoEnv


class WalkerRandParamsEnv(MujocoEnv, utils.EzPickle):
    def __init__(self):
        # achieve the target velocity
        self.goal_vel = np.array([4.0])  # TODO: important parameter
        self.sub_goal_vel_list = [np.array([3.0]), np.array([0.0]), np.array([6.0])]  # TODO: important parameter
        self.sub_goal_vel_idx = 0
        # parameter-related context
        self.rand_params = 'body_mass'  # ['body_mass', 'dof_damping'], [8, 9]
        self._context_dim = 4  # [<=8, <=9] # TODO: the most important parameter!!!
        self._context_limit = 3.0  # TODO: important parameter
        self.context = np.zeros(self._context_dim, dtype=np.float32)  # temporary
        self.is_expert = False  # temporary

        MujocoEnv.__init__(self, 'walker2d.xml', 4)
        self.save_parameters()

        utils.EzPickle.__init__(self)

    def get_context_dim(self):
        return self._context_dim

    def get_context_limit(self):
        return self._context_limit

    def sample_context(self):
        sampled_context = np.random.normal(loc=0.0, scale=1.0, size=self._context_dim)
        # sampled_context = self._context_limit * np.tanh(sampled_context / self._context_limit)
        sampled_context = sampled_context.clip(min=-self._context_limit, max=self._context_limit)

        return sampled_context

    def apply_context(self, context_rv: np.ndarray, is_expert: bool):
        assert len(context_rv) == self._context_dim
        self.context = context_rv
        self.is_expert = is_expert

        multiplyers = np.array(1.5) ** context_rv
        self.true_params = self.init_params * multiplyers
        # print("100: ", context_rv, multiplyers, self.init_params, self.true_params)
        param_variable = getattr(self.model, self.rand_params)
        # print("200: ", param_variable)
        # setattr(self.model, self.rand_params, self.true_params)
        self.set_parameters(params=self.true_params)
        # print("300: ", getattr(self.model, self.rand_params))

    def set_parameters(self, params):
        if 'body_mass' == self.rand_params:
            self.model.body_mass[:self._context_dim] = params
        # damping -> different multiplier for different dofs/joints
        elif 'dof_damping' == self.rand_params:
            self.model.dof_damping[:self._context_dim] = params

    def save_parameters(self):
        if 'body_mass' == self.rand_params:
            self.init_params = self.model.body_mass[:self._context_dim].copy()
        # damping -> different multiplier for different dofs/joints
        elif 'dof_damping' == self.rand_params:
            self.init_params = self.model.dof_damping[:self._context_dim].copy()

    def reset(self):
        super(WalkerRandParamsEnv, self).reset()
        self.sub_goal_vel_idx = 0
        return self._get_obs()

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        obs = np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()
        obs = np.concatenate([obs, self.sub_goal_vel_list[self.sub_goal_vel_idx]]) # TODO: with or without the target goal in obs

        if self.is_expert:
            obs_ext = np.concatenate([obs, self.true_params])
        else:
            obs_ext = np.concatenate([obs, self.context])
        return obs_ext

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv))
        return self._get_obs()

    def seed(self, seed_idx: int=None):
        super(WalkerRandParamsEnv, self).seed(seed_idx)
        self.action_space.np_random.seed(seed_idx)
        random.seed(seed_idx)
        np.random.seed(seed_idx)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]

        done_pre = not (height > 0.8 and height < 2.0 and ang > -1.0 and ang < 1.0)

        forward_vel = (posafter - posbefore) / self.dt
        cur_sub_goal_vel = self.sub_goal_vel_list[self.sub_goal_vel_idx][0]

        goal_bonus = 0.0
        vel_diff = abs(forward_vel - cur_sub_goal_vel)
        done = False
        if vel_diff <= 0.1:
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
        ctrl_cost = 0.05 * np.sum(np.square(a))
        reward = 0.1 * (forward_reward - ctrl_cost) + goal_bonus + 0.5 # 5.0 is the survive bonus for each time step

        obs = self._get_obs()

        # final_vel_diff = abs(forward_vel - self.goal_vel[0])
        # if final_vel_diff > 1e-3:
        #     done = False
        # else:
        #     done = True
        #     print("Great Success!!!")

        done = done or done_pre

        info = dict(reward_forward=forward_reward, reward_ctrl=-ctrl_cost, goal_bouns=goal_bonus, done_pre=done_pre)
        # print(info)
        return obs, reward, done, info


