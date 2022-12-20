"""Maze tasks that are defined by their map, termination condition, and goals.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, NamedTuple, Optional, Tuple, Type

import numpy as np
import random

from envir.mujoco_maze.maze_env_utils import MazeCell


class Rgb(NamedTuple):
    red: float
    green: float
    blue: float

    def rgba_str(self) -> str:
        return f"{self.red} {self.green} {self.blue} 1"


RED = Rgb(0.7, 0.1, 0.1)
GREEN = Rgb(0.1, 0.7, 0.1)
BLUE = Rgb(0.1, 0.1, 0.7)

class Scaling(NamedTuple):
    ant: Optional[float]
    point: Optional[float]


class MazeGoal:
    def __init__(
        self,
        pos: np.ndarray,
        reward_scale: float = 1.0,
        rgb: Rgb = RED,
        threshold: float = 1.5, # important parameter
        custom_size: Optional[float] = None,
    ) -> None:
        assert 0.0 <= reward_scale <= 1.0
        self.pos = pos
        self.dim = pos.shape[0]
        self.reward_scale = reward_scale
        self.rgb = rgb
        self.threshold = threshold
        self.custom_size = custom_size

    def neighbor(self, obs: np.ndarray) -> float: # so the first 2 dimensions have to be (x, y)
        return np.linalg.norm(obs[: self.dim] - self.pos) <= self.threshold

    def euc_dist(self, obs: np.ndarray) -> float:
        return np.sum(np.square(obs[: self.dim] - self.pos)) ** 0.5


class MazeTask(ABC):
    PENALTY: Optional[float] = None
    MAZE_SIZE_SCALING: Scaling = Scaling(ant=2.0, point=8.0)
    INNER_REWARD_SCALING: float = 0.0 # no reward from the mujoco setting

    def __init__(self, scale: float) -> None:
        self.goals = []
        self.scale = scale

    def sample_goals(self) -> bool:
        return False

    def set_goal(self, goal: np.ndarray) -> None:
        # print("The goal is set as: ", goal)
        self.goals = [MazeGoal(pos=goal)]

    def termination(self, obs: np.ndarray) -> bool:
        for goal in self.goals:
            if goal.neighbor(obs):
                print("Great Success!!!")
                # print(obs)
                return True
        return False

    def expert_action(self, cur_obs): # rule_based expert with random noise
        x = cur_obs[0]
        y = cur_obs[1]
        ori = cur_obs[2]
        goal_x = self.goals[0].pos[0]
        goal_y = self.goals[0].pos[1]
        tang = (goal_y - y) / (goal_x - x)
        target_ori = np.arctan(tang)
        if (goal_x - x) < 0:
            if (goal_y - y) < 0:
                target_ori -= np.pi
            else:
                target_ori += np.pi
        act = np.array([1.0, target_ori - ori])
        vel_noise = np.random.rand()
        ori_noise = 1.0 * np.random.rand() - 0.5
        act[0] -= vel_noise
        act[1] += ori_noise

        return act

    @abstractmethod
    def set_subgoal_list(self, subgoal_list) -> None:
        pass

    @abstractmethod
    def reward(self, obs: np.ndarray) -> float:
        pass

    @staticmethod
    @abstractmethod
    def create_maze() -> List[List[MazeCell]]:
        pass


class GoalRewardCell(MazeTask):
    PENALTY: float = 0.0
    MAZE_SIZE_SCALING: Scaling = Scaling(ant=2.0, point=8.0)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([8.0 * scale, -8.0 * scale]))]


    def reward(self, obs: np.ndarray) -> float:

        for goal in self.goals:
            if goal.neighbor(obs):
                return goal.reward_scale
        return self.PENALTY

    @staticmethod
    def create_maze() -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT

        return [
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
            [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B]
        ]



class MultiGoalRewardCell(GoalRewardCell):

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([8.0 * scale, -8.0 * scale]))]
        self.subgoal_list = None
        self.subgoal_idx = 0
        self.dist_threshold = 1.5 # important parameter

    def set_subgoal_list(self, subgoal_list) -> None:
        self.subgoal_list = subgoal_list
        self.subgoal_idx = 0

    def get_cur_subgoal(self):
        return self.subgoal_list[self.subgoal_idx]

    def get_cur_subgoal_idx(self):
        return self.subgoal_idx

    def reward(self, obs: np.ndarray, act: np.ndarray):
        xy = obs[:2]
        goal_xy = self.get_cur_subgoal()
        dist = np.linalg.norm(xy-goal_xy)
        subgoal_bonus = 0.0
        done = False
        if dist <= self.dist_threshold:
            print("Reach Goal {}: {}!".format(self.subgoal_idx, goal_xy))
            # dist = self.dist_threshold
            if self.subgoal_idx == len(self.subgoal_list) - 1:
                done = True
                print("Great Success!")
                subgoal_bonus += 1000.0
            self.subgoal_idx += 1
            if self.subgoal_idx > len(self.subgoal_list) - 1:
                self.subgoal_idx = len(self.subgoal_list) - 1

            subgoal_bonus += 100.0 # important parameter
        ctrl_cost = 0.05 * np.sum(np.square(act))
        rwd = 0.1 * (-dist - ctrl_cost) + subgoal_bonus + 1.0 # 0.5 is the survival bonus
        # print("Here: ", -dist, -ctrl_cost, subgoal_bonus, rwd)
        # ori_rwd = super(MultiGoalRewardCell, self).reward(obs)

        return rwd, done # important parameter



class DistRewardCell(GoalRewardCell):

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([6.0 * scale, 6.0 * scale]))]

    def reward(self, obs: np.ndarray) -> float:
        ori_rwd = super(DistRewardCell, self).reward(obs)
        return -self.goals[0].euc_dist(obs) / self.scale / 10.0 + ori_rwd * 100.0 # important parameter



class TaskRegistry:
    REGISTRY: Dict[str, List[Type[MazeTask]]] = {
        "Cell": [DistRewardCell, MultiGoalRewardCell]
    }

    @staticmethod
    def keys() -> List[str]:
        return list(TaskRegistry.REGISTRY.keys())

    @staticmethod
    def tasks(key: str) -> List[Type[MazeTask]]:
        return TaskRegistry.REGISTRY[key]
