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

    @abstractmethod
    def reward(self, obs: np.ndarray) -> float:
        pass

    @staticmethod
    @abstractmethod
    def create_maze() -> List[List[MazeCell]]:
        pass


class GoalRewardCell(MazeTask):
    PENALTY: float = -0.001
    MAZE_SIZE_SCALING: Scaling = Scaling(ant=2.0, point=8.0)

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.scale = scale
        self.goals = [MazeGoal(np.array([6.0 * scale, -6.0 * scale]))]
        self.subgoal_list = None
        self.subgoal_idx = 0
        self.dist_threshold = 1.5  # important parameter

    def set_subgoal_list(self) -> None:
        # self.subgoal_list = [np.array([2.0 * self.scale, 0.0]), np.array([4.0 * self.scale, 0.0]),
        #                      np.array([6.0 * self.scale, 0.0]), np.array([6.0 * self.scale, -6.0 * self.scale])]
        self.subgoal_list = [np.array([2.0 * self.scale, 0.0]), np.array([4.0 * self.scale, 0.0]),
                             np.array([4.0 * self.scale, -2.0 * self.scale]),
                             np.array([6.0 * self.scale, -2.0 * self.scale]), np.array([6.0 * self.scale, -6.0 * self.scale])]
        self.subgoal_idx = 0

    def get_cur_subgoal(self):
        return self.subgoal_list[self.subgoal_idx]

    def get_cur_subgoal_idx(self):
        return self.subgoal_idx

    def reward(self, obs: np.ndarray):
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
        rwd = subgoal_bonus - 0.1

        return rwd, done # important parameter


    def create_maze(self) -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT

        return [
            [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
            [B, E, B, B, B, E, E, E, B, B, B, B, B, E, B],
            [B, E, B, E, E, E, E, E, E, E, E, E, B, E, B],
            [B, E, B, E, B, E, E, E, B, B, B, E, B, E, B],
            [B, E, B, E, B, E, E, E, E, E, E, E, E, E, B],
            [B, E, B, E, B, E, E, E, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, R, E, E, E, E, E, E, B],
            [B, E, E, E, E, E, E, E, E, E, B, E, B, E, B],
            [B, E, E, E, E, E, E, E, E, E, B, E, B, E, B],
            [B, E, B, E, B, B, B, E, E, E, B, E, B, E, B],
            [B, E, B, E, E, E, E, E, E, E, E, E, B, E, B],
            [B, E, B, B, B, B, B, E, E, E, B, B, B, E, B],
            [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
            [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B]
        ]
        # return [
        #     [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
        #     [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
        #     [B, E, B, B, E, E, B, B, B, B, B, B, B, E, B],
        #     [B, E, B, E, E, E, E, E, E, E, E, E, B, E, B],
        #     [B, E, B, E, B, B, B, E, B, B, B, E, E, E, B],
        #     [B, E, B, E, B, E, E, E, E, E, B, E, E, E, B],
        #     [B, E, B, E, B, E, E, E, E, E, B, E, B, E, B],
        #     [B, E, B, E, E, E, E, R, E, E, E, E, B, E, B],
        #     [B, E, B, E, B, E, E, E, E, E, B, E, B, E, B],
        #     [B, E, E, E, B, E, E, E, E, E, B, E, B, E, B],
        #     [B, E, E, E, B, B, B, E, B, B, B, E, B, E, B],
        #     [B, E, B, E, E, E, E, E, E, E, E, E, B, E, B],
        #     [B, E, B, B, B, B, B, B, B, E, E, B, B, E, B],
        #     [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
        #     [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B]
        # ]

        # return [
        #     [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
        #     [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
        #     [B, E, B, B, E, E, E, B, B, B, B, B, B, E, B],
        #     [B, E, B, E, E, E, E, E, E, E, E, E, B, E, B],
        #     [B, E, B, E, B, B, E, E, E, B, B, E, E, E, B],
        #     [B, E, B, E, B, E, E, E, E, E, B, E, E, E, B],
        #     [B, E, B, E, E, E, E, E, E, E, E, E, E, E, B],
        #     [B, E, B, E, E, E, E, R, E, E, E, E, B, E, B],
        #     [B, E, E, E, E, E, E, E, E, E, E, E, B, E, B],
        #     [B, E, E, E, B, E, E, E, E, E, B, E, B, E, B],
        #     [B, E, E, E, B, B, E, E, E, B, B, E, B, E, B],
        #     [B, E, B, E, E, E, E, E, E, E, E, E, B, E, B],
        #     [B, E, B, B, B, B, B, B, E, E, E, B, B, E, B],
        #     [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
        #     [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B]
        # ]

class GoalRewardCell_1_1(GoalRewardCell):

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([6.0 * scale, -6.0 * scale]))]

    def set_subgoal_list(self) -> None:
        self.subgoal_list = [np.array([2.0 * self.scale, 0.0]),
                             np.array([6.0 * self.scale, 0.0]), np.array([6.0 * self.scale, -6.0 * self.scale])]
        # self.subgoal_list = [np.array([2.0 * self.scale, 0.0]), np.array([4.0 * self.scale, 0.0]),
        #                      np.array([4.0 * self.scale, -1.0 * self.scale]),
        #                      np.array([6.0 * self.scale, -1.0 * self.scale]), np.array([6.0 * self.scale, -6.0 * self.scale])]
        # fewer points are even faster
        # self.subgoal_list = [np.array([4.0 * self.scale, -2.0 * self.scale]), np.array([6.0 * self.scale, -6.0 * self.scale])]
        self.subgoal_idx = 0

class GoalRewardCell_1_2(GoalRewardCell):

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([6.0 * scale, 6.0 * scale]))]

    def set_subgoal_list(self) -> None:
        self.subgoal_list = [np.array([2.0 * self.scale, 0.0]),
                             np.array([6.0 * self.scale, 0.0]), np.array([6.0 * self.scale, 6.0 * self.scale])]
        # self.subgoal_list = [np.array([0.0, 2.0 * self.scale]), np.array([0.0, 4.0 * self.scale]),
        #                      np.array([2.0 * self.scale, 4.0 * self.scale]),
        #                      np.array([2.0 * self.scale, 6.0 * self.scale]), np.array([6.0 * self.scale, 6.0 * self.scale])]
        # self.subgoal_list = [np.array([2.0 * self.scale, 4.0 * self.scale]), np.array([6.0 * self.scale, 6.0 * self.scale])]
        self.subgoal_idx = 0

class GoalRewardCell_1_3(GoalRewardCell):

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([-6.0 * scale, 6.0 * scale]))]

    def set_subgoal_list(self) -> None:
        self.subgoal_list = [np.array([-2.0 * self.scale, 0.0]),
                             np.array([-6.0 * self.scale, 0.0]), np.array([-6.0 * self.scale, 6.0 * self.scale])]
        # self.subgoal_list = [np.array([-2.0 * self.scale, 0.0]), np.array([-4.0 * self.scale, 0.0]),
        #                      np.array([-4.0 * self.scale, 2.0 * self.scale]),
        #                      np.array([-6.0 * self.scale, 2.0 * self.scale]), np.array([-6.0 * self.scale, 6.0 * self.scale])]
        # self.subgoal_list = [np.array([-4.0 * self.scale, 2.0 * self.scale]), np.array([-6.0 * self.scale, 6.0 * self.scale])]
        self.subgoal_idx = 0

class GoalRewardCell_1_4(GoalRewardCell):

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([-6.0 * scale, -6.0 * scale]))]

    def set_subgoal_list(self) -> None:
        self.subgoal_list = [np.array([-2.0 * self.scale, 0.0]),
                             np.array([-6.0 * self.scale, 0.0]), np.array([-6.0 * self.scale, -6.0 * self.scale])]
        # self.subgoal_list = [np.array([0.0, -2.0 * self.scale]), np.array([0.0, -4.0 * self.scale]),
        #                      np.array([-2.0 * self.scale, -4.0 * self.scale]),
        #                      np.array([-2.0 * self.scale, -6.0 * self.scale]),
        #                      np.array([-6.0 * self.scale, -6.0 * self.scale])]
        # self.subgoal_list = [np.array([-2.0 * self.scale, -4.0 * self.scale]), np.array([-6.0 * self.scale, -6.0 * self.scale])]
        self.subgoal_idx = 0


class GoalRewardCell_2_1(GoalRewardCell):

    def __init__(self, scale: float) -> None:
        super().__init__(scale)
        self.goals = [MazeGoal(np.array([5.0 * scale, -5.0 * scale]))]

    def set_subgoal_list(self) -> None:
        self.subgoal_list = [np.array([5.0 / 3.0 * self.scale, 0.0]), np.array([10.0 / 3.0 * self.scale, 0.0]),
                             np.array([5.0 * self.scale, 0.0]), np.array([5.0 * self.scale, -5.0 * self.scale])]
        self.subgoal_idx = 0

    def create_maze(self) -> List[List[MazeCell]]:
        E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT

        return [
            [B, B, B, B, B, B, B, B, B, B, B, B, B],
            [B, E, E, E, E, E, E, B, E, E, E, E, B],
            [B, E, B, B, B, B, E, B, E, B, B, E, B],
            [B, E, B, B, B, B, E, B, E, B, B, E, B],
            [B, E, E, E, B, B, E, B, B, B, B, E, B],
            [B, B, B, B, B, E, E, E, B, B, B, E, B],
            [B, E, E, E, E, E, R, E, E, E, E, E, B],
            [B, E, B, B, B, E, E, E, B, B, B, B, B],
            [B, E, B, B, B, B, E, B, B, E, E, E, B],
            [B, E, B, B, E, B, E, B, B, B, B, E, B],
            [B, E, B, B, E, B, E, B, B, B, B, E, B],
            [B, E, E, E, E, B, E, E, E, E, E, E, B],
            [B, B, B, B, B, B, B, B, B, B, B, B, B]
        ]




class TaskRegistry:
    REGISTRY: Dict[str, List[Type[MazeTask]]] = {
        "Room": [GoalRewardCell_1_1, GoalRewardCell_1_2, GoalRewardCell_1_3, GoalRewardCell_1_4],
        "Maze": [GoalRewardCell_2_1]
    }

    @staticmethod
    def keys() -> List[str]:
        return list(TaskRegistry.REGISTRY.keys())

    @staticmethod
    def tasks(key: str) -> List[Type[MazeTask]]:
        return TaskRegistry.REGISTRY[key]
