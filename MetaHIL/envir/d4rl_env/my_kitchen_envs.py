"""Environments using kitchen and Franka robot."""
import numpy as np
import random

from d4rl.kitchen.adept_envs.utils.configurable import configurable
from d4rl.kitchen.adept_envs.franka.kitchen_multitask_v0 import KitchenTaskRelaxV1
from d4rl.offline_env import OfflineEnv
from .task_config import TASK_SET, OBS_ELEMENT_INDICES, OBS_ELEMENT_GOALS, BONUS_THRESH


@configurable(pickleable=True)
class MyKitchenBase(KitchenTaskRelaxV1, OfflineEnv):
    # A string of element names. The robot's task is then to modify each of
    # these elements appropriately.
    TASK_ELEMENTS = []
    REMOVE_TASKS_WHEN_COMPLETE = True
    TERMINATE_ON_TASK_COMPLETE = True

    def __init__(self, dataset_url=None, ref_max_score=None, ref_min_score=None, **kwargs):
        self.tasks_to_complete = self.TASK_ELEMENTS.copy()
        super(MyKitchenBase, self).__init__(**kwargs)
        OfflineEnv.__init__(
            self,
            dataset_url=dataset_url,
            ref_max_score=ref_max_score,
            ref_min_score=ref_min_score) # TODO: get rid of this

    def _get_task_goal(self): # this is part of the obs, but only executed at the beginning of the episode, i.e., reset()

        new_goal = np.zeros_like(self.goal)
        for element in self.TASK_ELEMENTS:
        # for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            element_goal = OBS_ELEMENT_GOALS[element]
            new_goal[element_idx] = element_goal

        return new_goal

    def reset_model(self):
        self.tasks_to_complete = self.TASK_ELEMENTS.copy()
        # print("3000: ", self.tasks_to_complete)
        return super(MyKitchenBase, self).reset_model()

    def _get_reward_n_score(self, obs_dict):
        reward_dict, score = super(MyKitchenBase, self)._get_reward_n_score(obs_dict)
        next_q_obs = obs_dict['qp']
        next_obj_obs = obs_dict['obj_qp']
        next_goal = obs_dict['goal']
        idx_offset = len(next_q_obs)

        bonus = 0.0
        element = self.tasks_to_complete[0] # the tasks in the list should be completed in sequence
        element_idx = OBS_ELEMENT_INDICES[element]
        distance = np.linalg.norm(next_obj_obs[..., element_idx - idx_offset] - next_goal[element_idx])
        bonus += 1.0 / max(distance, BONUS_THRESH)
        complete = distance < BONUS_THRESH
        if complete:
            bonus += 100.0 # important parameter
            print("Finish Task #{}!!!".format(len(self.TASK_ELEMENTS)-len(self.tasks_to_complete)))
            if self.REMOVE_TASKS_WHEN_COMPLETE:
                self.tasks_to_complete = self.tasks_to_complete[1:]

        reward_dict['bonus'] = bonus
        reward_dict['r_total'] = bonus # reward returned in step()
        score = bonus
        # print("200: ", reward_dict)
        return reward_dict, score

    def step(self, a, b=None):
        obs, reward, done, env_info = super(MyKitchenBase, self).step(a, b=b)
        if self.TERMINATE_ON_TASK_COMPLETE:
            done = (len(self.tasks_to_complete) == 0)
            if done:
                print("Great Success!!!")
        return obs, reward, done, env_info


class KitchenMultiTask(MyKitchenBase):
    TASK_ELEMENTS = ['bottom burner'] # temporary

    def __init__(self):
        self._context_dim = 1
        self._context_limit = 2.0
        self._context_interval = 2 * self._context_limit / float(len(TASK_SET))
        self._task_name_list = list(OBS_ELEMENT_INDICES.keys())
        self.context = np.array([0.0])  # temporary
        self.is_expert = False  # temporary

        super(KitchenMultiTask, self).__init__()

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

        task_idx = int((self.context[0] - (-self._context_limit)) // self._context_interval)
        if task_idx < 0:
            task_idx = 0
        if task_idx >= len(TASK_SET):
            task_idx = len(TASK_SET) - 1
        cur_task = TASK_SET[task_idx]
        cur_task_elements = []
        for idx in cur_task:
            cur_task_elements.append(self._task_name_list[idx])
        self.TASK_ELEMENTS = cur_task_elements # in reset, self.tasks_to_complete will be updated accordingly
        # print("2000: ", task_idx, cur_task, cur_task_elements, self.TASK_ELEMENTS)


    def _get_obs(self):
        obs = super(KitchenMultiTask, self)._get_obs()
        if not self.is_expert:
            obs = np.concatenate([obs, self.context])

        return obs

    def seed(self, seed_idx: int=None):
        super(KitchenMultiTask, self).seed(seed_idx)
        self.action_space.np_random.seed(seed_idx)
        random.seed(seed_idx)
        np.random.seed(seed_idx)


class KitchenBottomBurner(MyKitchenBase):
    TASK_ELEMENTS = ['bottom burner']

class KitchenTopBurner(MyKitchenBase):
    TASK_ELEMENTS = ['top burner']

class KitchenLightSwitch(MyKitchenBase):
    TASK_ELEMENTS = ['light switch']

class KitchenSlideCabinet(MyKitchenBase):
    TASK_ELEMENTS = ['slide cabinet']

class KitchenHingeCabinet(MyKitchenBase):
    TASK_ELEMENTS = ['hinge cabinet']

class KitchenMicrowave(MyKitchenBase):
    TASK_ELEMENTS = ['microwave']

class KitchenKettle(MyKitchenBase):
    TASK_ELEMENTS = ['kettle']

