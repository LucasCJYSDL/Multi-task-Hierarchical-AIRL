#!/usr/bin/env python3
import random

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model.option_policy import OptionPolicy
from utils.common_utils import validate, reward_validate

def vae_loss(policy, sa_array, temperature):
    losses = []

    temp_array_list = random.sample(sa_array, 64)
    for s_array, a_array in temp_array_list:
        # print(s_array.shape, a_array.shape)
        epi_len = int(s_array.shape[0])
        ct_1 = torch.empty(1, 1, dtype=torch.long, device=policy.device).fill_(policy.dim_c)
        for t in range(epi_len):
            st = s_array[t].unsqueeze(0) # tensor on the corresponding device
            at = a_array[t].unsqueeze(0)
            losses.append(policy.vae_forward(st, ct_1, at, temperature)) # note that ct_1 is the last_step option choice
            ct = policy.sample_option(st, ct_1=ct_1, tau=temperature, fixed=False) # for training (1, 1)
            ct_1 = ct

    return sum(losses) / len(sa_array)


def pretrain(policy: OptionPolicy, sa_array, save_name_f, logger, msg, n_iter, log_interval):

    optimizer = torch.optim.Adam(policy.parameters(), weight_decay=1.e-3)

    log_test = logger.log_pretrain
    log_train = logger.log_pretrain

    anneal_rate = 0.00003
    temp_min = 0.5
    temperature = 1.0
    cool_interval = 10

    for i in tqdm(range(n_iter)):
        loss = vae_loss(policy, sa_array, temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TODO: with or without the cooling process
        if i % cool_interval == 0:
            temperature = np.maximum(temperature * np.exp(-anneal_rate * i), temp_min)

        if (i + 1) % log_interval == 0:
            torch.save(policy.state_dict(), save_name_f(i))
            print(f"pre-{i}; loss={loss.item()}; {msg}")
        else:
            print(f"pre-{i}; loss={loss.item()}; {msg}")
        log_train("loss", loss.item(), i)
        logger.flush()