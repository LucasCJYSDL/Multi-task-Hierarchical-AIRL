#!/usr/bin/env python3
import random

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model.option_policy import OptionPolicy
from utils.common_utils import validate, reward_validate

def vae_loss(policy, high_batch, low_batch, high_loss_func):
    batch_s, batch_c1, batch_c = high_batch
    pre_c_batch = policy.log_trans(batch_s, batch_c1)
    high_loss = high_loss_func(pre_c_batch, batch_c.squeeze(-1))

    batch_s, batch_c, batch_a = low_batch
    low_loss = - torch.mean(policy.log_prob_action(batch_s, batch_c, batch_a))

    # print("3: ", high_loss, low_loss)

    return high_loss, low_loss


def get_batches(high_set, low_set):
    mini_highs = random.sample(high_set, 1024)
    mini_lows = random.sample(low_set, 1024)
    # print("2: ", len(mini_highs), len(mini_lows))
    high_s, high_c1, high_c = [], [], []
    low_s, low_c, low_a = [], [], []
    for s, c1, c in mini_highs:
        high_s.append(s)
        high_c1.append(c1)
        high_c.append(c)

    for s, c, a in mini_lows:
        low_s.append(s)
        low_c.append(c)
        low_a.append(a)

    high_s_batch = torch.cat(high_s, dim=0)
    high_c1_batch = torch.cat(high_c1, dim=0)
    high_c_batch = torch.cat(high_c, dim=0)

    low_s_batch = torch.cat(low_s, dim=0)
    low_c_batch = torch.cat(low_c, dim=0)
    low_a_batch = torch.cat(low_a, dim=0)

    # print(high_s_batch.shape, high_c1_batch.shape, high_c_batch.shape, low_s_batch.shape, low_c_batch.shape, low_a_batch.shape)

    return (high_s_batch, high_c1_batch, high_c_batch), (low_s_batch, low_c_batch, low_a_batch)



def pretrain(policy: OptionPolicy, sa_array, save_name_f, logger, msg, n_iter, log_interval):
    high_loss_func = torch.nn.NLLLoss()
    high_optimizer = torch.optim.Adam(policy.get_certain_param(low_policy=False), weight_decay=1.e-3)
    low_optimizer = torch.optim.Adam(policy.get_certain_param(low_policy=True), weight_decay=1.e-3)

    log_test = logger.log_pretrain
    log_train = logger.log_pretrain

    anneal_rate = 0.00003
    temp_min = 0.5
    temperature = 1.0
    cool_interval = 10

    high_set = []
    low_set = []

    for s_array, c_array, a_array in sa_array:
        # print(s_array.shape, c_array.shape, a_array.shape)
        epi_len = int(s_array.shape[0])
        # ct_1 = torch.empty(1, 1, dtype=torch.long, device=policy.device).fill_(policy.dim_c)
        for t in range(epi_len):
            st = s_array[t].unsqueeze(0) # tensor on the corresponding device
            at = a_array[t].unsqueeze(0)
            ct_1 = c_array[t].unsqueeze(0)
            ct = c_array[t+1].unsqueeze(0)
            high_set.append((st, ct_1, ct))
            low_set.append((st, ct, at))

    print("1: ", len(high_set), len(low_set))

    for i in tqdm(range(n_iter)):

        high_batch, low_batch = get_batches(high_set, low_set)

        high_loss, low_loss = vae_loss(policy, high_batch, low_batch, high_loss_func)

        high_optimizer.zero_grad()
        high_loss.backward()
        high_optimizer.step()

        low_optimizer.zero_grad()
        low_loss.backward()
        low_optimizer.step()

        # TODO: with or without the cooling process
        if i % cool_interval == 0:
            temperature = np.maximum(temperature * np.exp(-anneal_rate * i), temp_min)

        if (i + 1) % log_interval == 0:
            torch.save(policy.state_dict(), save_name_f(i))
            print(f"pre-{i}; high_loss={high_loss.item()}; low_loss={low_loss.item()}; {msg}")
        # else:
        #     print(f"pre-{i}; high_loss={high_loss.item()}; low_loss={low_loss.item()}; {msg}")
        log_train("high_loss", high_loss.item(), i)
        log_train("low_loss", low_loss.item(), i)
        logger.flush()