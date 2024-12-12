# -*- coding: utf-8 -*-


import os
import pandas as pd
import numpy as np
from utils.config_utils import get_config
from utils.file_utils import absolute_path
from tqdm import tqdm
import scipy.sparse as sp


def get_all_np_list():
    model_config = get_config('model')
    data_path = model_config['gnn']['data_path']
    # if data_path == 'addr_only0':
    #     return get_addr_np_list()
    address_has_tx_feature = model_config['gnn']['address_has_tx_feature'] == str(True)
    address_has_tx_feature_repeat = model_config['gnn']['address_has_tx_feature_repeat'] == str(True)
    config = get_config('path')
    # all_np_list_path = absolute_path(config[data_path]['all_np_list'])
    # addr_np_list_path = absolute_path(config[data_path]['addr_np_list'])
    addr_feature_np_list_path = absolute_path(config[data_path]['addr_feature_np_list'])
    addr_class_np_list_path = absolute_path(config[data_path]['addr_class_np_list'])
    tx_np_list_path = absolute_path(config[data_path]['tx_np_list'])
    hyperedge_np_list_path = absolute_path(config[data_path]['hyperedge_np_list'])
    addr_edge_np_list_path = absolute_path(config[data_path]['addr_edge_np_list'])
    idx_np_list_path = absolute_path(config[data_path]['idx_np_list'])
    # hyperedge_index_np_list_path = absolute_path(config[data_path]['hyperedge_index_np_list'])
    # hyperedge_weight_np_list_path = absolute_path(config[data_path]['hyperedge_weight_np_list'])

    if os.path.exists(hyperedge_np_list_path) and \
            os.path.exists(addr_feature_np_list_path) and \
            os.path.exists(addr_class_np_list_path) and \
            os.path.exists(tx_np_list_path) and \
            os.path.exists(addr_edge_np_list_path) and os.path.exists(idx_np_list_path):
        addr_feature_data = np.load(addr_feature_np_list_path, allow_pickle=True)
        addr_class_data = np.load(addr_class_np_list_path, allow_pickle=True)
        tx_data = np.load(tx_np_list_path, allow_pickle=True)
        hyperedge_data = np.load(hyperedge_np_list_path, allow_pickle=True)
        addr_edge_data = np.load(addr_edge_np_list_path, allow_pickle=True)
        idx_data = np.load(idx_np_list_path, allow_pickle=True)
        return addr_feature_data, addr_class_data, tx_data, hyperedge_data, addr_edge_data, idx_data

    if address_has_tx_feature_repeat:
        addr_features = pd.read_csv(absolute_path(config['file']['addr_feature_with_tx_feature']))
    elif address_has_tx_feature:
        addr_features = pd.read_csv(absolute_path(config['file']['addr_feature_with_tx_feature_mean']))
    else:
        addr_features = pd.read_csv(absolute_path(config['file']['addr_features_no_repeat']))

    addr_classes = pd.read_csv(absolute_path(config['file']['addr_classes']))
    addrTxValue = pd.read_csv(absolute_path(config['file']['addrTxValue']))
    txAddrValue = pd.read_csv(absolute_path(config['file']['txAddrValue']))
    addr_edges = pd.read_csv(absolute_path(config['file']['addr_edges_new']))

    txs_features = pd.read_csv(absolute_path(config['file']['txs_features_new']))
    txs_classes = pd.read_csv(absolute_path(config['file']['txs_classes_new']))
    txs_edges = pd.read_csv(absolute_path(config['file']['txs_edges_new']))

    dataset_config = get_config('dataset')
    addr_features_list, addr_classes_list, addrTxValue_list, txAddrValue_list, addr_edge_index_list, \
        txs_features_list, txs_classes_list, txs_edges_list, \
        hyperedge_index_list, hyperedge_weight_list, addr_idx_list, txs_idx_list = \
        get_all_time_list(1, int(dataset_config['Elliptic++']['time_steps']),
                          addr_features, addr_classes, addrTxValue, txAddrValue, addr_edges,
                          txs_features, txs_classes, txs_edges)

    # np.savez(all_np_list_path,
    #          addr_features_list=addr_features_list, addr_classes_list=addr_classes_list,
    #          addrTxValue_list=addrTxValue_list, txAddrValue_list=txAddrValue_list,
    #          hyperedge_index_list=hyperedge_index_list, hyperedge_weight_list=hyperedge_weight_list,
    #          txs_features_list=txs_features_list, txs_classes_list=txs_classes_list, txs_edges_list=txs_edges_list)

    # np.savez(addr_np_list_path,
    #          addr_features_list=addr_features_list, addr_classes_list=addr_classes_list)
    # np.savez(addr_feature_np_list_path, addr_features_list=addr_features_list)
    np.savez(addr_feature_np_list_path, addr_features_train_list=addr_features_list[:34],
             addr_features_test_list=addr_features_list[34:])
    np.savez(addr_class_np_list_path, addr_classes_list=addr_classes_list)
    np.savez(tx_np_list_path,
             txs_features_list=txs_features_list, txs_classes_list=txs_classes_list, txs_edges_list=txs_edges_list)
    np.savez(hyperedge_np_list_path,
             hyperedge_index_list=hyperedge_index_list, hyperedge_weight_list=hyperedge_weight_list)
    # np.savez(hyperedge_index_np_list_path, hyperedge_index_list=hyperedge_index_list)
    # np.savez(hyperedge_weight_np_list_path, hyperedge_weight_list=hyperedge_weight_list)
    np.savez(addr_edge_np_list_path, addrTxValue_list=addrTxValue_list, txAddrValue_list=txAddrValue_list,
             addr_edge_index_list=addr_edge_index_list)
    np.savez(idx_np_list_path, addr_idx_list=addr_idx_list, txs_idx_list=txs_idx_list)

    addr_feature_data = np.load(addr_feature_np_list_path, allow_pickle=True)
    addr_class_data = np.load(addr_class_np_list_path, allow_pickle=True)
    tx_data = np.load(tx_np_list_path, allow_pickle=True)
    hyperedge_data = np.load(hyperedge_np_list_path, allow_pickle=True)
    addr_edge_data = np.load(addr_edge_np_list_path, allow_pickle=True)
    idx_data = np.load(idx_np_list_path, allow_pickle=True)
    return addr_feature_data, addr_class_data, tx_data, hyperedge_data, addr_edge_data, idx_data


def get_tx_np_list():
    config = get_config('path')
    model_config = get_config('model')
    data_path = model_config['gnn']['data_path']
    tx_np_list_path = absolute_path(config[data_path]['tx_np_list'])
    if not os.path.exists(tx_np_list_path):
        get_all_np_list()
    data = np.load(tx_np_list_path, allow_pickle=True)
    return data


def get_addr_feature_np_list():
    config = get_config('path')
    model_config = get_config('model')
    data_path = model_config['gnn']['data_path']
    addr_feature_np_list_path = absolute_path(config[data_path]['addr_feature_np_list'])
    if not os.path.exists(addr_feature_np_list_path):
        get_all_np_list()
    data = np.load(addr_feature_np_list_path, allow_pickle=True)
    return data


def get_addr_class_np_list():
    config = get_config('path')
    model_config = get_config('model')
    data_path = model_config['gnn']['data_path']
    addr_class_np_list_path = absolute_path(config[data_path]['addr_class_np_list'])
    if not os.path.exists(addr_class_np_list_path):
        get_all_np_list()
    data = np.load(addr_class_np_list_path, allow_pickle=True)
    return data


def get_hyperedge_np_list():
    config = get_config('path')
    model_config = get_config('model')
    data_path = model_config['gnn']['data_path']
    hyperedge_np_list_path = absolute_path(config[data_path]['hyperedge_np_list'])
    if not os.path.exists(hyperedge_np_list_path):
        get_all_np_list()
    data = np.load(hyperedge_np_list_path, allow_pickle=True)
    return data


def get_addr_edge_np_list():
    config = get_config('path')
    model_config = get_config('model')
    data_path = model_config['gnn']['data_path']
    addr_edge_np_list_path = absolute_path(config[data_path]['addr_edge_np_list'])
    if not os.path.exists(addr_edge_np_list_path):
        get_all_np_list()
    data = np.load(addr_edge_np_list_path, allow_pickle=True)
    return data


def creat_index_all_np_list(addr_features_i, addr_classes_i, addrTxValue_i, txAddrValue_i, addr_edges_i,
                            txs_features_i, txs_classes_i, txs_edges_i):
    print('create index!')
    model_config = get_config('model')
    address_has_tx_feature_repeat = model_config['gnn']['address_has_tx_feature_repeat'] == str(True)
    assert (addr_features_i[:, 1] if address_has_tx_feature_repeat else addr_features_i[:, 0]).shape[0] == \
           np.unique(addr_features_i[:, 1] if address_has_tx_feature_repeat else addr_features_i[:, 0]).shape[0], 'address duplication'
    assert txs_features_i[:, 0].shape[0] == np.unique(txs_features_i[:, 0]).shape[0], 'tx duplication'
    addr_idx = np.array(addr_features_i[:, 1] if address_has_tx_feature_repeat else addr_features_i[:, 0], dtype=str)
    addr_idx_map = {j: i for i, j in enumerate(addr_idx)}
    txs_idx = np.array(txs_features_i[:, 0], dtype=np.int32)
    txs_idx_map = {j: i for i, j in enumerate(txs_idx)}
    # addr_classes_i[:, 0] = np.array(list(map(addr_idx_map.get, addr_classes_i[:, 0])), dtype=np.int32)
    # addr_classes_i = addr_classes_i[np.argsort(addr_classes_i[:, 0])]
    addrTxValue_i[:, 0] = np.array(list(map(txs_idx_map.get, addrTxValue_i[:, 0])), dtype=np.int32)
    addrTxValue_i[:, 1] = np.array(list(map(addr_idx_map.get, addrTxValue_i[:, 1])), dtype=np.int32)
    txAddrValue_i[:, 0] = np.array(list(map(txs_idx_map.get, txAddrValue_i[:, 0])), dtype=np.int32)
    txAddrValue_i[:, 1] = np.array(list(map(addr_idx_map.get, txAddrValue_i[:, 1])), dtype=np.int32)
    addr_edges_i[:, 0] = np.array(list(map(addr_idx_map.get, addr_edges_i[:, 0])), dtype=np.int32)
    addr_edges_i[:, 1] = np.array(list(map(addr_idx_map.get, addr_edges_i[:, 1])), dtype=np.int32)

    txs_edges_i = np.array(list(map(txs_idx_map.get, txs_edges_i.flatten())),
                           dtype=np.int32).reshape(txs_edges_i.shape)
    # txs_classes_i[:, 0] = np.array(list(map(txs_idx_map.get, txs_classes_i[:, 0])), dtype=np.int32)
    # txs_classes_i = txs_classes_i[np.argsort(txs_classes_i[:, 0])]

    return addr_features_i, addr_classes_i, addrTxValue_i, txAddrValue_i, addr_edges_i, \
        txs_features_i, txs_classes_i, txs_edges_i, addr_idx, txs_idx


def get_all_time_list(start_time, end_time, addr_features, addr_classes, addrTxValue, txAddrValue,
                      addr_edges, txs_features, txs_classes, txs_edges):
    print('Get a list of numpy data for time')

    addr_classes["class"] = addr_classes["class"].astype(str).replace("2", "0").astype("int32")
    txs_classes["class"] = txs_classes["class"].astype(str).replace("2", "0").astype("int32")

    columns = ["Time step", "addr_feature", "addr_class", "addr_tx", "tx_addr", "addr_edge",
               "tx_feature", "tx_class", "tx_edge"]
    time_step_size = pd.DataFrame(np.zeros(shape=(49, len(columns))), columns=columns)

    addr_features_list = []
    addr_classes_list = []
    addrTxValue_list = []
    txAddrValue_list = []
    txs_features_list = []
    txs_classes_list = []
    txs_edges_list = []
    txs_edges_num = 0
    hyperedge_index_list = []
    hyperedge_weight_list = []
    addr_edge_index_list = []
    addr_idx_list = []
    txs_idx_list = []
    addr_edges_num = 0
    txs_num = 0

    for i in tqdm(range(start_time, end_time + 1), desc='Time step: '):
        # txs_features.sort_values(by="txId", inplace=True, ascending=True)
        txs_features_i = txs_features.loc[txs_features['Time step'] == i, :]
        txs_i = txs_features_i[['txId']]
        txs_num += txs_i.shape[0]
        txs_classes_i = pd.merge(txs_i, txs_classes, how='left', on='txId')
        assert txs_i.shape[0] == txs_classes_i.shape[0], \
            f'The number of transactions and transaction types is not equal！txs_i:{txs_i.shape[0]}, txs_classes_i:{txs_classes_i.shape[0]}'
        assert sum(np.where(txs_features_i['txId'] == txs_features_i['txId'], 0, 1)) == 0, 'txs: features do not correspond to classes'

        txs_edges_i = txs_edges[txs_edges.apply(lambda x: x['txId1'] in txs_i.values and x['txId2'] in txs_i.values,
                                                axis=1)]
        txs_edges_num += txs_edges_i.shape[0]

        addrTxValue_i = pd.merge(txs_i, addrTxValue, how='left', on='txId')
        txAddrValue_i = pd.merge(txs_i, txAddrValue, how='left', on='txId')

        model_config = get_config('model')
        if model_config['gnn']['address_has_tx_feature_repeat'] != str(True):
            addr_i = pd.DataFrame(pd.concat(
                [addrTxValue_i['input_address'], txAddrValue_i['output_address']], axis=0).drop_duplicates(),
                                  columns=['address'])
            addr_features_i = addr_features.loc[addr_features['Time step'] == i, :]
            addr_features_i = pd.merge(addr_i, addr_features_i, how='left', on='address')
            assert addr_features_i.shape[1] == addr_features.shape[1], 'There is a problem with the merge'
            addr_classes_i = pd.merge(addr_i, addr_classes, how='left', on='address')
            assert addr_features_i.shape[0] == addr_classes_i.shape[0], 'shape error'
            assert addr_features_i.shape[0] == addr_features_i.drop_duplicates().shape[0], 'features duplication'

            hyperedge_index_i, hyperedge_weight_i = get_hyperedge_index_weight_no_tx_feature(
                addr_classes_i, addrTxValue_i, txs_classes_i)
        else:
            print("with tx features! ")
            addr_features_i = pd.merge(txs_i, addr_features, how='left', on='txId')
            assert addr_features_i.shape == addr_features.loc[addr_features['Time step'] == i, :].shape
            addr_classes_i = pd.merge(addr_features_i[['address']], addr_classes, how='left', on='address')
            assert addr_features_i.shape[0] == addr_classes_i.shape[0], 'shape error'
            assert addr_features_i.shape[0] == addr_features_i.drop_duplicates().shape[0]
            hyperedge_index_i, hyperedge_weight_i = get_hyperedge_index_weight_with_tx_feature(
                addr_features_i, addrTxValue_i, txs_classes_i)

        assert addr_features_i.isnull().sum().sum() == 0, f'i: {i}'
        assert sum(np.where(addr_features_i['address'] == addr_classes_i['address'], 0, 1)) == 0, 'addrs: features do not correspond to classes'

        addr_edges_i = pd.merge(addr_edges, txs_i, on='txId')[['input_address', 'output_address']]
        addr_edges_num += addr_edges_i.shape[0]
        addr_features_i, addr_classes_i, addrTxValue_i, txAddrValue_i, addr_edges_i, \
            txs_features_i, txs_classes_i, txs_edges_i, addr_idx, txs_idx = \
            creat_index_all_np_list(addr_features_i.to_numpy(), addr_classes_i.to_numpy(),
                                    addrTxValue_i.to_numpy(), txAddrValue_i.to_numpy(), addr_edges_i.to_numpy(),
                                    txs_features_i.to_numpy(), txs_classes_i.to_numpy(), txs_edges_i.to_numpy())

        time_step_size.loc[i, 'Time step'] = i + 1
        time_step_size.loc[i, 'addr_feature'] = addr_features_i.shape[0]
        time_step_size.loc[i, 'addr_class'] = addr_classes_i.shape[0]
        time_step_size.loc[i, 'addr_tx'] = addrTxValue_i.shape[0]
        time_step_size.loc[i, 'tx_addr'] = txAddrValue_i.shape[0]
        time_step_size.loc[i, 'addr_edge'] = addr_edges_i.shape[0]
        time_step_size.loc[i, 'tx_feature'] = txs_features_i.shape[0]
        time_step_size.loc[i, 'tx_class'] = txs_classes_i.shape[0]
        time_step_size.loc[i, 'tx_edge'] = txs_edges_i.shape[0]

        addr_features_list.append(addr_features_i)
        addr_classes_list.append(addr_classes_i)
        addrTxValue_list.append(addrTxValue_i)
        txAddrValue_list.append(txAddrValue_i)
        txs_features_list.append(txs_features_i)
        txs_classes_list.append(txs_classes_i)
        txs_edges_list.append(txs_edges_i)
        hyperedge_index_list.append(hyperedge_index_i)
        hyperedge_weight_list.append(hyperedge_weight_i)
        addr_edge_index_list.append(addr_edges_i)
        addr_idx_list.append(addr_idx)
        txs_idx_list.append(txs_idx)
    assert txs_edges_num == txs_edges.shape[0], \
        f'Transaction edge connections between different time steps！txs_edges_num:{txs_edges_num}, txs_edges:{txs_edges.shape[0]}'
    assert addr_edges_num == addr_edges.shape[0], \
        f'The different time steps are connected by transaction edges (addrAddr)！addr_edges_num:{addr_edges_num}, addr_edges:{addr_edges.shape[0]}'
    assert txs_num == txs_classes.shape[0], \
        f'Incorrect number of transactions！txs_num:{txs_num}, txs_classes:{txs_classes.shape[0]}'

    config = get_config('path')
    model_config = get_config('model')
    data_path = model_config['gnn']['data_path']
    time_step_size.to_csv(
        absolute_path(config[data_path]['time_step_size']), mode='w', header=True, index=False)
    return addr_features_list, addr_classes_list, addrTxValue_list, txAddrValue_list, addr_edge_index_list, \
        txs_features_list, txs_classes_list, txs_edges_list, hyperedge_index_list, hyperedge_weight_list, \
        addr_idx_list, txs_idx_list


def get_hyperedge_index_weight_no_tx_feature(addr_classes, addrTxValue, txs_classes):
    edge_index = np.zeros((addr_classes.shape[0], txs_classes.shape[0]), dtype=np.int32)
    edge_weight = np.zeros((txs_classes.shape[0]), dtype=np.float64)
    for i in range(addrTxValue.shape[0]):
        curr_addr = addrTxValue.loc[i, 'input_address']
        curr_tx = addrTxValue.loc[i, 'txId']
        curr_tx_index = txs_classes.loc[txs_classes['txId'] == curr_tx, :].index.tolist()[0]
        curr_addr_index = addr_classes[addr_classes['address'] == curr_addr].index.tolist()[0]
        edge_index[curr_addr_index, curr_tx_index] = 1
        edge_weight[curr_tx_index] += addrTxValue.loc[i, 'input_value']
    # for i in range(addr_classes.shape[0]):
    #     curr_addr = addr_classes.loc[i, 'address']
    #     curr_addrTxValue = addrTxValue.loc[addrTxValue['input_address'] == curr_addr, :]
    #     for j in range(curr_addrTxValue.shape[0]):
    #         curr_tx = curr_addrTxValue.loc[j, 'txId']
    #         curr_tx_index = txs_classes.loc[txs_classes['txId'] == curr_tx, :].index.tolist()[0]
    #         edge_index[i, curr_tx_index] = 1
    #         edge_weight[curr_tx_index] += curr_addrTxValue.loc[j, 'input_value']
    return sp.coo_matrix(edge_index), edge_weight


def get_hyperedge_index_weight_with_tx_feature(addr_features, addrTxValue, txs_classes):
    edge_index = np.zeros((addr_features.shape[0], txs_classes.shape[0]), dtype=np.int32)
    edge_weight = np.zeros((txs_classes.shape[0]), dtype=np.float64)
    for i in range(addrTxValue.shape[0]):
        curr_addr = addrTxValue.loc[i, 'input_address']
        curr_tx = addrTxValue.loc[i, 'txId']
        curr_tx_index = txs_classes.loc[txs_classes['txId'] == curr_tx, :].index.tolist()[0]
        curr_addr_index = addr_features[(addr_features['address'] == curr_addr) & (addr_features['txId'] == curr_tx)].index.tolist()[0]
        edge_index[curr_addr_index, curr_tx_index] = 1
        edge_weight[curr_tx_index] += addrTxValue.loc[i, 'input_value']
    return sp.coo_matrix(edge_index), edge_weight


if __name__ == '__main__':
    get_all_np_list()
