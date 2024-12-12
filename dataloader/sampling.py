# -*- coding: utf-8 -*-

from utils.config_utils import get_config_option
from utils.seed_utils import setup_seed
setup_seed(int(get_config_option("model", "gnn", "seed")))
import math
import random
import numpy as np

from utils.config_utils import get_config


def address_down_sampling_mask(rs_NP_ratio, classes, train_mask):
    dataset_path = get_config('dataset')
    down_sampling = dataset_path['Elliptic++']['down_sampling'] == str(True)

    if down_sampling:
        # rs_NP_ratio = float(dataset_path['Elliptic++']['rs_NP_ratio'])
        P_num = (classes[train_mask] == 1).sum()  # the number of positive samples
        N_num = (classes[train_mask] == 0).sum()  # the number of negative samples
        if N_num <= math.floor(P_num * rs_NP_ratio):   # The number of negative samples is less than or equal to the expected negative samples
            return train_mask
        # Otherwise, under sampling
        Neg_index = set(np.where((classes == 0) & train_mask)[0])
        Neg_abandon_index = random.sample(list(Neg_index), N_num - math.floor(P_num * rs_NP_ratio))
        Neg_mask = np.full(classes.shape, True, dtype=bool)
        Neg_mask[Neg_abandon_index] = False
        train_mask = train_mask & Neg_mask
        # print(f"Random Sampling: Pos num {sum(classes[train_mask]==1)}, Neg num {N_num}, reserve Neg num {sum(classes[train_mask]==0)}")
        return train_mask
    return train_mask