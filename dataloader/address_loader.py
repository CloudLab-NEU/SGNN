# -*- coding: utf-8 -*-


import numpy as np

from utils.config_utils import get_config
from utils.dataset_utils import get_addr_class_np_list


def addr_hash_time_train_test():
    # address，Time step, label
    addr_class_data = get_addr_class_np_list()
    addr_classes_list = addr_class_data['addr_classes_list']
    train_results = []
    test_results = []
    dataset_config = get_config('dataset')
    for i in range(int(dataset_config['Elliptic++']['train_st'])-1, int(dataset_config['Elliptic++']['train_et'])):
        addr_classes_i = addr_classes_list[i]
        addr_classes_i = np.insert(addr_classes_i, 1, i+1, axis=1)
        train_results.append(addr_classes_i)
    for i in range(int(dataset_config['Elliptic++']['test_st'])-1, int(dataset_config['Elliptic++']['test_et'])):
        addr_classes_i = addr_classes_list[i]
        addr_classes_i = np.insert(addr_classes_i, 1, i+1, axis=1)
        test_results.append(addr_classes_i)
    return np.vstack(train_results), np.vstack(test_results)
