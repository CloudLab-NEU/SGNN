# -*- coding: utf-8 -*-

from utils.config_utils import get_config_option
from utils.seed_utils import setup_seed
setup_seed(int(get_config_option("model", "gnn", "seed")))
import time
import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data
from tqdm import tqdm

from utils.config_utils import get_config
from utils.dataset_utils import get_hyperedge_np_list, get_tx_np_list, \
    get_addr_edge_np_list
from utils.file_utils import absolute_path, create_csv, writerow_csv

def dataloader():
    tx_data = get_tx_np_list()
    hyperedge_data = get_hyperedge_np_list()
    addr_edge_data = get_addr_edge_np_list()
    scaler = MinMaxScaler()
    datas = []
    hyperedge_index_list = hyperedge_data['hyperedge_index_list']
    hyperedge_weight_list = hyperedge_data['hyperedge_weight_list']
    txs_edges_list = tx_data['txs_edges_list']
    addr_edge_index_list = addr_edge_data['addr_edge_index_list']

    batch_times = ["hyper_AG_building_time"]
    for i in tqdm(range(int(get_config('dataset')['Elliptic++']['time_steps'])), desc='Hyper Data List: '):
        start_time = time.time()

        addr_edge_index = torch.LongTensor(addr_edge_index_list[i].transpose().astype(int))
        tx_edge_index = torch.LongTensor(txs_edges_list[i].transpose().astype(int))

        hyperedge_index = get_hyperedge_index(hyperedge_index_list[i])
        hyperedge_weight = torch.Tensor(scaler.fit_transform(hyperedge_weight_list[i].reshape(-1, 1)))
        data = Data(
            addr_edge_index=addr_edge_index,
            tx_edge_index = tx_edge_index,
            hyperedge_index=hyperedge_index,
            hyperedge_weight=hyperedge_weight
        )

        end_time = time.time()
        batch_time = end_time - start_time
        batch_times.append(batch_time * 1000)

        datas.append(data)

    # cost_path = absolute_path(get_config_option("path", "file", "cost_model"))
    # cost_columns = ["type"]
    # cost_columns.extend([f"Time step {i + 1}" for i in range(int(get_config('dataset')['Elliptic++']['time_steps']))])
    # if not os.path.exists(cost_path):
    #     create_csv(cost_path, cost_columns)
    # writerow_csv(cost_path, batch_times)
    return datas


def get_tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()

def get_data_size():
    datas = dataloader()
    AG_data_sizes = ["hyper_AG_data_sizes"]
    TG_data_sizes = ["hyper_TG_data_sizes"]
    for i in range(len(datas)):
        data = datas[i]
        tx_edge_index = data.tx_edge_index
        hyperedge_index = data.hyperedge_index
        hyperedge_weight = data.hyperedge_weight
        AG_data_size = (get_tensor_bytes(hyperedge_index) + get_tensor_bytes(hyperedge_weight)) / (1024 * 1024)

        TG_data_size = (get_tensor_bytes(tx_edge_index)) / (1024 * 1024)

        AG_data_sizes.append(AG_data_size)
        TG_data_sizes.append(TG_data_size)
    cost_path = absolute_path(get_config_option("path", "file", "cost_model"))
    cost_columns = ["type"]
    cost_columns.extend([f"Time step {i + 1}" for i in range(len(datas))])
    if not os.path.exists(cost_path):
        create_csv(cost_path, cost_columns)
    writerow_csv(cost_path, AG_data_sizes)
    writerow_csv(cost_path, TG_data_sizes)

def get_hyperedge_index(hyperedge_index):
    return torch.LongTensor(
        np.vstack((hyperedge_index.row, hyperedge_index.col)).astype(np.int64))


def get_hyper_train_test_loader():
    hyper_loader = dataloader()
    dataset_config = get_config('dataset')
    hyper_train_loader = \
        hyper_loader[int(dataset_config['Elliptic++']['train_st']) - 1:int(dataset_config['Elliptic++']['train_et'])]
    hyper_test_loader = \
        hyper_loader[int(dataset_config['Elliptic++']['test_st']) - 1:int(dataset_config['Elliptic++']['test_et'])]
    return hyper_train_loader, hyper_test_loader


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.LongTensor(indices, values, shape)


if __name__ == '__main__':
    get_data_size()
