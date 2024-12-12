# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from utils.file_utils import absolute_path
from utils.config_utils import get_config


def get_tx_by_address():
    config = get_config('path')
    tx_class_df = pd.read_csv(absolute_path(config['file']['txs_classes']))
    addrTx_df = pd.read_csv(absolute_path(config['file']['addrTx']))
    txAddr_df = pd.read_csv(absolute_path(config['file']['txAddr']))
    tx_class_df = tx_class_df[tx_class_df.apply(lambda x: x['txId'] in addrTx_df['txId'].values
                                                          and x['txId'] in txAddr_df['txId'].values, axis=1)]
    tx_class_df.to_csv(absolute_path(config['file']['txs_classes_new']), index=False)


def get_tx_edge_feature_by_tx():
    config = get_config('path')
    tx_class_df = pd.read_csv(absolute_path(config['file']['txs_classes_new']))
    tx_edge_df = pd.read_csv(absolute_path(config['file']['txs_edges']))
    tx_feature_df = pd.read_csv(absolute_path(config['file']['txs_features']))
    tx_feature_df = tx_feature_df[tx_feature_df.apply(lambda x: x['txId'] in tx_class_df['txId'].values, axis=1)]
    tx_edge_df = tx_edge_df[tx_edge_df.apply(lambda x: x['txId1'] in tx_class_df['txId'].values and
                                                       x['txId2'] in tx_class_df['txId'].values, axis=1)]
    txId2hash = pd.read_csv(absolute_path(config['file']['txId2hash']))
    tx_time_df = pd.read_csv(absolute_path(config['file']['txs_time']))
    tx_time_df = pd.merge(txId2hash, tx_time_df, how='left', on='txhash')
    tx_feature_df = pd.merge(tx_time_df, tx_feature_df, on='txId')
    assert tx_feature_df.shape[0] == tx_time_df.shape[0], 'Some txs don not have timestamp！'
    tx_time_df.to_csv(absolute_path(config['file']['txs_time_new']), index=False)
    tx_feature_df.to_csv(absolute_path(config['file']['txs_features_new']), index=False)
    tx_edge_df.to_csv(absolute_path(config['file']['txs_edges_new']), index=False)


def tx_indegree():
    """input: addrTx or txAddr"""
    config = get_config('path')
    addrTx = pd.read_csv(absolute_path(config['file']['addrTx']))
    indegree = addrTx.groupby(['txId']).size().reset_index()
    indegree.columns = ['txId', 'indegree']
    indegree.to_csv(absolute_path(config['file']['txs_indegree']), index=False)
    txAddr = pd.read_csv(absolute_path(config['file']['txAddr']))
    outdegree = txAddr.groupby(['txId']).size().reset_index()
    outdegree.columns = ['txId', 'outdegree']
    outdegree.to_csv(absolute_path(config['file']['txs_outdegree']), index=False)
    degree = pd.merge(indegree, outdegree, on='txId')
    degree.to_csv(absolute_path(config['file']['txs_degree']), index=False)


def tx_total_values():
    config = get_config('path')
    addrTxValue = pd.read_csv(absolute_path(config['file']['addrTxValue']))
    in_total_values = addrTxValue.groupby(['txId'])['input_value'].sum().reset_index()
    in_total_values.columns = ['txId', 'in_total_values']
    in_total_values.to_csv(absolute_path(config['file']['input_total_values_new']), index=False)
    txAddrValue = pd.read_csv(absolute_path(config['file']['txAddrValue']))
    out_total_values = txAddrValue.groupby(['txId'])['output_value'].sum().reset_index()
    out_total_values.columns = ['txId', 'out_total_values']
    out_total_values.to_csv(absolute_path(config['file']['output_total_values_new']), index=False)
    total_values = pd.merge(in_total_values, out_total_values, on='txId')
    total_values['values_difference'] = total_values['in_total_values'] - total_values['out_total_values']
    total_values.to_csv(absolute_path(config['file']['txs_total_values_new']), index=False)


def txs_some_feature():
    config = get_config('path')
    txs_degree = pd.read_csv(absolute_path(config['file']['txs_degree']))
    txs_total_values = pd.read_csv(absolute_path(config['file']['txs_total_values_new']))
    txs_classes = pd.read_csv(absolute_path(config['file']['txs_classes_new']))
    txs_some_feature = pd.merge(txs_degree, txs_total_values, on='txId')
    txs_some_feature = pd.merge(txs_some_feature, txs_classes, on='txId')
    assert txs_some_feature.shape[0] == txs_classes.shape[0]
    txs_some_feature.to_csv(absolute_path(config['file']['txs_some_feature_new']), index=False)


if __name__ == '__main__':
    # get_tx_by_address()
    # get_tx_edge_feature_by_tx()
    # tx_indegree()
    # tx_total_values()
    txs_some_feature()