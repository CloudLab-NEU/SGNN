# -*- coding: utf-8 -*-

from utils.config_utils import get_config_option
from utils.seed_utils import setup_seed
setup_seed(int(get_config_option("model", "gnn", "seed")))
import time
import os
import sys
import random
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch_geometric.data import HeteroData
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from operator import eq
from utils.file_utils import absolute_path, create_csv, writerow_csv
from utils.config_utils import get_config
from utils.dataset_utils import get_tx_np_list, get_addr_feature_np_list, get_addr_class_np_list, get_addr_edge_np_list
from dataloader.sampling import address_down_sampling_mask

def dataloader(rs_NP_ratio, seed=int(get_config_option("model", "gnn", "seed"))):
    # setup_seed(seed)
    addr_feature_data = get_addr_feature_np_list()
    addr_class_data = get_addr_class_np_list()
    addr_edge_data = get_addr_edge_np_list()
    tx_data = get_tx_np_list()
    scaler = MinMaxScaler()
    datas = []
    addr_features_list = addr_feature_data['addr_features_train_list'].tolist() + addr_feature_data['addr_features_test_list'].tolist()
    addr_classes_list = addr_class_data['addr_classes_list']
    txs_features_list = tx_data['txs_features_list']
    txs_classes_list = tx_data['txs_classes_list']
    addrTxValue_list = addr_edge_data['addrTxValue_list']
    txAddrValue_list = addr_edge_data['txAddrValue_list']
    model_config = get_config('model')
    address_has_tx_feature_repeat = model_config['gnn']['address_has_tx_feature_repeat'] == str(True)
    address_tx_no_feature = model_config['gnn']['address_tx_no_feature'] == str(True)
    tx_no_feature = model_config['gnn']['tx_no_feature'] == str(True)
    hetero_edge_reverse = model_config['gnn']['hetero_edge_reverse'] == str(True)
    hetero_edge_forward = model_config['gnn']['hetero_edge_forward'] == str(True)

    address_feature_start = 3 if address_has_tx_feature_repeat else 2
    addrs_pd_list = []
    txs_pd_list = []
    # addr_test_all_num = 0
    batch_times = ["hetero_TG_building_time"]
    for i in tqdm(range(int(get_config('dataset')['Elliptic++']['time_steps'])), desc='Hetero Data List: '):
        start_time = time.time()

        data = HeteroData()
        addr_classes_i = addr_classes_list[i][:, 1].astype(int)
        txs_classes_i = txs_classes_list[i][:, 1].astype(int)
        txs_train_mask, txs_test_mask = get_tx_mask(txs_classes_i)

        data['address'].x = torch.Tensor(scaler.fit_transform(np.ones((len(addr_classes_list[i]), 1), dtype=float))) \
            if address_tx_no_feature \
            else torch.Tensor(scaler.fit_transform(
            addr_features_list[i][:, address_feature_start:]))  # [num_papers, num_features_address]
        data['address'].y = torch.LongTensor(addr_classes_i)
        # print("train:")
        addr_train_mask = get_addr_mask(addr_classes_list[i], txs_classes_list[i], txs_train_mask)
        # print("test:")
        addr_test_mask = get_addr_mask(addr_classes_list[i], txs_classes_list[i], txs_test_mask)
        # print(sum(addr_test_mask == 1))
        # addr_test_all_num += sum(addr_test_mask == 1)
        addr_train_mask = address_down_sampling_mask(rs_NP_ratio, addr_classes_i, addr_train_mask)
        assert all(addr_classes_i[addr_train_mask | addr_test_mask] != 3), "class should not equal 3"
        data['address'].train_mask = torch.BoolTensor(addr_train_mask)
        data['address'].test_mask = torch.BoolTensor(addr_test_mask)

        data['transaction'].x = torch.Tensor(scaler.fit_transform(np.ones((len(txs_classes_list[i]), 1), dtype=float))) \
            if address_tx_no_feature or tx_no_feature \
            else torch.Tensor(scaler.fit_transform(txs_features_list[i][:, 5:]))
        data['transaction'].y = torch.LongTensor(txs_classes_i)
        txs_train_mask = address_down_sampling_mask(rs_NP_ratio, txs_classes_i, txs_train_mask)
        data['transaction'].train_mask = torch.BoolTensor(txs_train_mask)
        data['transaction'].test_mask = torch.BoolTensor(txs_test_mask)
        data['transaction'].txId = torch.LongTensor(txs_classes_list[i][:, 0])

        if hetero_edge_reverse:
            # print(f"hetero_edge_reverse: {hetero_edge_reverse}!")
            data['address', 'flow_reverse', 'transaction'].edge_index = \
                torch.LongTensor(txAddrValue_list[i][:, [1, 0]].transpose().astype(int))  # [2, num_edges_flows]
            data['transaction', 'flow_reverse', 'address'].edge_index = \
                torch.LongTensor(addrTxValue_list[i][:, :2].transpose().astype(int))

        if hetero_edge_forward:
            data['address', 'flow', 'transaction'].edge_index = \
                torch.LongTensor(addrTxValue_list[i][:, [1, 0]].transpose().astype(int))  # [2, num_edges_flows]
            data['transaction', 'flow', 'address'].edge_index = \
                torch.LongTensor(txAddrValue_list[i][:, :2].transpose().astype(int))

        end_time = time.time()
        batch_time = end_time - start_time
        batch_times.append(batch_time * 1000)
        # batch_times.append(batch_time)

        datas.append(data)

    # print(addr_test_all_num)
    # cost_path = absolute_path(get_config_option("path", "file", "cost_model"))
    # cost_columns = ["type"]
    # cost_columns.extend([f"Time step {i + 1}" for i in range(int(get_config('dataset')['Elliptic++']['time_steps']))])
    # if not os.path.exists(cost_path):
    #     create_csv(cost_path, cost_columns)
    # writerow_csv(cost_path, batch_times)


    #     addrs_pd = pd.DataFrame(columns=['address', 'Time step', 'train_mask', 'test_mask', 'class'])
    #     txs_pd = pd.DataFrame(columns=['txId', 'Time step', 'train_mask', 'test_mask', 'class'])
    #     addrs_pd['address'] = addr_classes_list[i][:, 0]
    #     addrs_pd['Time step'] = i + 1
    #     addrs_pd['train_mask'] = addr_train_mask
    #     addrs_pd['test_mask'] = addr_test_mask
    #     addrs_pd['class'] = addr_classes_list[i][:, 1]
    #     txs_pd['txId'] = txs_classes_list[i][:, 0]
    #     txs_pd['Time step'] = i + 1
    #     txs_pd['train_mask'] = txs_train_mask
    #     txs_pd['test_mask'] = txs_test_mask
    #     txs_pd['class'] = txs_classes_list[i][:, 1]
    #     addrs_pd_list.append(addrs_pd)
    #     txs_pd_list.append(txs_pd)
    # addrs_pd = pd.concat(addrs_pd_list)
    # txs_pd = pd.concat(txs_pd_list)
    # addrs_pd.to_csv(absolute_path(
    #     get_config_option('path', 'file', 'addr_train_test').format(rs_NP_ratio=rs_NP_ratio)),
    #     index=False)
    # txs_pd.to_csv(absolute_path(
    #     get_config_option('path', 'file', 'tx_train_test').format(rs_NP_ratio=rs_NP_ratio)),
    #     index=False)
    return datas

def get_hetero_train_test_loader(rs_NP_ratio):
    hetero_loader = dataloader(rs_NP_ratio)
    dataset_config = get_config('dataset')
    hetero_train_loader = hetero_loader[
                          int(dataset_config['Elliptic++']['train_st'])-1:int(dataset_config['Elliptic++']['train_et'])]
    hetero_test_loader = hetero_loader[
                         int(dataset_config['Elliptic++']['test_st'])-1:int(dataset_config['Elliptic++']['test_et'])]
    return hetero_train_loader, hetero_test_loader


def get_tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()

def get_data_size(rs_NP_ratio):
    hetero_train_loader, _ = get_hetero_train_test_loader(rs_NP_ratio)
    address_nums = ["address_num"]
    transaction_nums = ["transaction_num"]
    AG_data_sizes = ["hetero_AG_data_sizes"]
    TG_data_sizes = ["hetero_TG_data_sizes"]
    for i in range(len(hetero_train_loader)):
        data = hetero_train_loader[i]
        address_x = data['address'].x
        address_y = data['address'].y
        transaction_x = data['transaction'].x
        transaction_y = data['transaction'].y
        address_num = data['address'].x.shape[0]
        transaction_num = data['transaction'].x.shape[0]
        edge_index1 = data['address', 'flow', 'transaction'].edge_index
        edge_index2 = data['transaction', 'flow', 'address'].edge_index
        edge_index3 = data['address', 'flow_reverse', 'transaction'].edge_index
        edge_index4 = data['transaction', 'flow_reverse', 'address'].edge_index
        AG_data_size = (get_tensor_bytes(address_x) + get_tensor_bytes(address_y) + get_tensor_bytes(transaction_x)
                     + get_tensor_bytes(transaction_y) + get_tensor_bytes(edge_index1) + get_tensor_bytes(edge_index2)
                     + get_tensor_bytes(edge_index3) + get_tensor_bytes(edge_index4)) / (1024 * 1024)
        TG_data_size = (get_tensor_bytes(transaction_x) + get_tensor_bytes(transaction_y)) / (1024 * 1024)
        address_nums.append(address_num)
        transaction_nums.append(transaction_num)
        AG_data_sizes.append(AG_data_size)
        TG_data_sizes.append(TG_data_size)
    cost_path = absolute_path(get_config_option("path", "file", "cost_model"))
    cost_columns = ["type"]
    cost_columns.extend([f"Time step {i+1}" for i in range(len(hetero_train_loader))])
    if not os.path.exists(cost_path):
        create_csv(cost_path, cost_columns)
    writerow_csv(cost_path, address_nums)
    writerow_csv(cost_path, transaction_nums)
    writerow_csv(cost_path, AG_data_sizes)
    writerow_csv(cost_path, TG_data_sizes)
        # print(f"{i} address_x: {get_tensor_bytes(address_x) / 1024}")  # KB
        # print(f"{i} address_y: {get_tensor_bytes(address_y) / 1024}")  # KB
        # print(f"{i} transaction_x: {get_tensor_bytes(transaction_x) / 1024}")  # KB
        # print(f"{i} transaction_y: {get_tensor_bytes(transaction_y) / 1024}")  # KB
        # print(f"{i} edge_index1: {get_tensor_bytes(edge_index1) / 1024}")  # KB
        # print(f"{i} edge_index2: {get_tensor_bytes(edge_index2) / 1024}")  # KB
        # print(f"{i} edge_attr1: {get_tensor_bytes(edge_attr1) / 1024}")  # KB
        # print(f"{i} edge_attr2: {get_tensor_bytes(edge_attr2) / 1024}")  # KB
        # print(f"{i} edge_index3: {get_tensor_bytes(edge_index3) / 1024}")  # KB
        # print(f"{i} edge_index4: {get_tensor_bytes(edge_index4) / 1024}")  # KB
        # print(f"{i} edge_attr3: {get_tensor_bytes(edge_attr3) / 1024}")  # KB
        # print(f"{i} edge_attr4: {get_tensor_bytes(edge_attr4) / 1024}")  # KB



######################################
def get_mask(classes):
    train_mask = classes != 3
    test_mask = classes != 3
    return train_mask, test_mask


def get_addr_mask(addr_classes, txs_classes, txs_mask):
    path_config = get_config('path')
    address_inOrOut_tx_time_class = pd.read_csv(absolute_path(path_config['file']['address_inOrOut_tx_time_class']))
    address_inOrOut_tx_time_class = address_inOrOut_tx_time_class.drop(
        index=address_inOrOut_tx_time_class[
            (address_inOrOut_tx_time_class['tx_class'] == 3)].index.tolist()).reset_index()
    txs = pd.DataFrame(txs_classes[:, 0][txs_mask], columns=['txId'])
    # print(f"tx num : {len(txs_classes[:, 0][txs_mask])}")
    addrs = pd.merge(txs, address_inOrOut_tx_time_class, on='txId')[['txId', 'address', 'addr_class']].drop_duplicates()
    addrs = addrs.loc[addrs['addr_class'] != 3, :]
    addr_index = np.where(np.isin(addr_classes[:, 0], np.array(addrs['address'])))[0]
    addr_mask = np.full(addr_classes[:, 0].shape, False, dtype=bool)
    addr_mask[addr_index] = True
    return addr_mask

def get_tx_mask(classes):
    dataset_path = get_config('dataset')
    train_set_ratio = float(dataset_path['Elliptic++']['train_set_ratio'])
    data_index = np.where(classes != 3)[0]
    train_index = random.sample(list(data_index), round(train_set_ratio * len(data_index)))
    test_index = list(set(data_index) - set(train_index))
    train_mask = np.full(classes.shape, False, dtype=bool)
    test_mask = np.full(classes.shape, False, dtype=bool)
    train_mask[train_index] = True
    test_mask[test_index] = True
    return train_mask, test_mask


if __name__ == '__main__':
    # get_data_size(rs_NP_ratio=5)
    # get_hetero_train_test_loader(rs_NP_ratio=5)
    dataloader(5)






