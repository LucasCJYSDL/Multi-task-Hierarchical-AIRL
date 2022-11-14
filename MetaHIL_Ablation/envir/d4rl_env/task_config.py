import numpy as np

TASK_SET = [
    [0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5], [0, 1, 6],
    [0, 2, 3], [0, 2, 4], [0, 2, 5], [0, 2, 6],
    [0, 3, 4], [0, 3, 5], [0, 3, 6],
    [0, 4, 5], [0, 4, 6],
    [0, 5, 6],
    [1, 2, 3], [1, 2, 4], [1, 2, 5], [1, 2, 6],
    [1, 3, 4], [1, 3, 5], [1, 3, 6],
    [1, 4, 5], [1, 4, 6],
    [1, 5, 6],
    [2, 3, 4], [2, 3, 5], [2, 3, 6],
    [2, 4, 5], [2, 4, 6],
    [2, 5, 6],
    [3, 4, 5], [3, 4, 6],
    [3, 5, 6],
    [4, 5, 6]
]

OBS_ELEMENT_INDICES = {
    'bottom burner': np.array([11, 12]),
    'top burner': np.array([15, 16]),
    'light switch': np.array([17, 18]),
    'slide cabinet': np.array([19]),
    'hinge cabinet': np.array([20, 21]),
    'microwave': np.array([22]),
    'kettle': np.array([23, 24, 25, 26, 27, 28, 29]),
    }
OBS_ELEMENT_GOALS = {
    'bottom burner': np.array([-0.88, -0.01]),
    'top burner': np.array([-0.92, -0.01]),
    'light switch': np.array([-0.69, -0.05]),
    'slide cabinet': np.array([0.37]),
    'hinge cabinet': np.array([0., 1.45]),
    'microwave': np.array([-0.75]),
    'kettle': np.array([-0.23, 0.75, 1.62, 0.99, 0., 0., -0.06]),
    }
BONUS_THRESH = 0.1 # TODO: important parameter