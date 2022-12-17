import numpy as np

# the objects are sorted by their locations, since they should be operated in the order from bottom to top
TASK_SET = [
    [0, 1, 3, 4], [0, 1, 2, 3],
    [6, 0, 3, 4], [6, 0, 2, 3], [6, 0, 2, 4], [6, 0, 1, 4], [6, 0, 1, 3], [6, 0, 1, 2], [6, 2, 3, 4], [6, 1, 2, 3],
    [5, 0, 3, 4], [5, 0, 2, 3], [5, 0, 1, 2], [5, 0, 1, 3], [5, 0, 1, 4], [5, 6, 0, 4], [5, 6, 0, 3], [5, 6, 3, 4],
    [5, 6, 2, 3], [5, 6, 2, 4], [5, 6, 1, 4], [5, 6, 1, 2], [5, 2, 3, 4], [5, 1, 2, 4]
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

# BONUS_THRESH = {
#     'bottom burner': 0.3,
#     'top burner': 0.3,
#     'light switch': 0.1, # sure
#     'slide cabinet': 0.1, # sure
#     'hinge cabinet': 0.3, # sure
#     'microwave': 0.1,
#     'kettle': 0.3,
#     }

BONUS_THRESH = {
    'bottom burner': 0.3,
    'top burner': 0.3,
    'light switch': 0.3, # sure
    'slide cabinet': 0.3, # sure
    'hinge cabinet': 0.3, # sure
    'microwave': 0.3,
    'kettle': 0.3,
    }

GOALS ={
    'microwave': [-0.7257075, -1.76812719, 1.87064519, -1.743234, -0.40666457, 1.37335837, 2.31221587, 0.02517328, 0.0199741],
    'top burner': [-1.36160548, -1.28755842,  1.37085593, -2.22010132,  0.13305926,  1.92485686, 1.83141891, 0.00518498, 0.01898529],
    'light switch': [-1.25434787, -1.5240441, 1.40332382, -1.98307871, 0.25769077, 1.59573581, 1.28076917, 0.04758731, 0.00334905],
    'hinge cabinet': [-1.29686945, -1.75817476, 0.81952516, -2.23927152, 1.63736467, 1.35726842, -0.25862886, 0.01189523, 0.03942422],
    'slide cabinet': [-1.88249748, -1.29900735, 1.058495, -1.91799365, 0.25787251, 1.82972458, 1.06006236, 0.03650248, 0.00991203],
    'bottom burner': [-1.35762018, -1.39887535, 1.41825019, -2.31836902, -0.1938481, 2.02699184, 2.17750164, 0.00890941, 0.01784447],
    'kettle': [-0.9732931, -1.75514653, 1.94103677, -2.20073082, -0.13495215, 1.1877437, 1.99049029, 0.02339095, 0.01859485]
}

if __name__ == '__main__':
    task_name_list = list(BONUS_THRESH.keys())
    for tasklist in TASK_SET:
        temp_task_names = []
        for task in tasklist:
            temp_task_names.append(task_name_list[task])
        print(temp_task_names)