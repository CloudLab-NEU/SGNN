# -*- coding: utf-8 -*-


import pandas as pd

from utils.config_utils import get_config
from utils.file_utils import absolute_path



def get_address_feature_no_repeat():
    config = get_config('path')
    df = pd.read_csv(absolute_path(config['file']['addr_features']))
    df.drop_duplicates().to_csv(absolute_path(config['file']['addr_features_no_repeat']), index=False)


def get_address_feature_with_tx_feature_repeat():
    config = get_config('path')
    addr_features = pd.read_csv(absolute_path(config['file']['addr_features']))
    addr_features = addr_features.drop(columns=['Time step']).drop_duplicates()  # address
    tx_features = pd.read_csv(absolute_path(config['file']['txs_features']))  # txId
    addrTx = pd.read_csv(absolute_path(config['file']['addrTx']))
    txAddr = pd.read_csv(absolute_path(config['file']['txAddr']))
    addrTx.columns = ['address', 'txId']
    txAddr.columns = ['txId', 'address']
    addr_tx = pd.concat([addrTx[['address', 'txId']], txAddr[['address', 'txId']]], axis=0)
    assert addr_tx.shape[0] == addrTx.shape[0] + txAddr.shape[0]
    addr_features_all = pd.merge(addr_tx, addr_features, on='address')
    assert addr_features_all.shape[0] == addr_tx.shape[0]
    addr_features_with_tx_features = pd.merge(addr_features_all, tx_features, on='txId')
    assert addr_features_with_tx_features.shape[0] == addr_features_all.shape[0]
    print(addr_features_with_tx_features.shape)
    addr_features_with_tx_features = addr_features_with_tx_features.drop_duplicates()
    print(addr_features_with_tx_features.shape)
    df_time_step = addr_features_with_tx_features['Time step']
    addr_features_with_tx_features = addr_features_with_tx_features.drop('Time step', axis=1)
    addr_features_with_tx_features.insert(1, 'Time step', df_time_step)
    addr_features_with_tx_features.to_csv(absolute_path(config['file']['addr_features_with_tx_features']), index=False)


def get_addr_feature_with_tx_feature_mean():
    config = get_config('path')
    addr_features = pd.read_csv(absolute_path(config['file']['addr_features_with_tx_features']))
    addr_features = addr_features.drop('txId', axis=1)
    addr_feature_with_tx_feature_mean = addr_features.groupby(['address', 'Time step']).mean().reset_index()
    addr_feature_with_tx_feature_mean.to_csv(absolute_path(config['file']['addr_feature_with_tx_feature_mean']), index=False)



def get_address_edge_new():
    config = get_config('path')
    addrTx = pd.read_csv(absolute_path(config['file']['addrTx']))
    txAddr = pd.read_csv(absolute_path(config['file']['txAddr']))
    addr_edges = pd.read_csv(absolute_path(config['file']['addr_edges']))
    addr_edges_new = pd.merge(addrTx, txAddr, on='txId')
    assert addr_edges.shape[0] == addr_edges_new.shape[0], f'merge error！addr_edges:{addr_edges.shape[0]}, addr_edges_new:{addr_edges_new.shape[0]}'
    addr_edges_new.to_csv(absolute_path(config['file']['addr_edges_new']), index=False)
    print(f'addr_edges:{addr_edges.shape[0]}, addr_edges_new:{addr_edges_new.shape[0]}')



if __name__ == '__main__':
    get_addr_feature_with_tx_feature_mean()

