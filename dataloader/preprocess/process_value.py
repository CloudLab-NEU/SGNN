# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from utils.config_utils import get_config
from utils.file_utils import absolute_path


def addrTxValue():
    """sent value and time"""
    config = get_config('path')
    value_df = pd.read_csv(absolute_path(config['file']['in_value_new']))
    addTx_df = pd.read_csv(absolute_path(config['file']['addrTx']))
    txId2hash_df = pd.read_csv(absolute_path(config['file']['txId2hash']))
    id_df = pd.merge(txId2hash_df, value_df, how='left', on='txhash')
    df = pd.merge(addTx_df, id_df, how='left', on=['txId', 'input_address'])
    tx_time_df = pd.read_csv(absolute_path(config['file']['txs_time_new']),
                             usecols=['txId', 'txhash', 'block_timestamp', 'block_number'])
    df = pd.merge(df[['txId', 'input_address', 'txhash', 'input_value']], tx_time_df, how='left', on=['txId', 'txhash'])[
        ['txId', 'input_address', 'txhash', 'input_value', 'block_timestamp', 'block_number']]
    assert df.shape[0] == addTx_df.shape[0]
    df.to_csv(absolute_path(config['file']['addrTxValue']), index=False)


def txAddrValue():
    """received value and time"""
    config = get_config('path')
    value_df = pd.read_csv(absolute_path(config['file']['out_value_new']))
    txAdd_df = pd.read_csv(absolute_path(config['file']['txAddr']))
    txId2hash_df = pd.read_csv(absolute_path(config['file']['txId2hash']))
    id_df = pd.merge(txId2hash_df, value_df, how='left', on='txhash')
    df = pd.merge(txAdd_df, id_df, how='left', on=['txId', 'output_address'])
    tx_time_df = pd.read_csv(absolute_path(config['file']['txs_time_new']),
                             usecols=['txId', 'txhash', 'block_timestamp', 'block_number'])
    df = pd.merge(df[['txId', 'output_address', 'txhash', 'output_value']], tx_time_df, how='left', on=['txId', 'txhash'])[
        ['txId', 'output_address', 'txhash', 'output_value', 'block_timestamp', 'block_number']]
    assert df.shape[0] == txAdd_df.shape[0]
    df.to_csv(absolute_path(config['file']['txAddrValue']), index=False)


def process_in_out_value_repeat_address():
    config = get_config('path')
    in_value = pd.read_csv(absolute_path(config['file']['in_value']))
    out_value = pd.read_csv(absolute_path(config['file']['out_value']))
    in_value_new = in_value.groupby(by=['txhash', 'input_address'])['input_value'].sum()
    # print(in_value_new)
    # print(in_value_new.to_frame())
    in_value_new.to_frame().to_csv(absolute_path(config['file']['in_value_new']))
    out_value_new = out_value.groupby(by=['txhash', 'output_address'])['output_value'].sum()
    # print(out_value_new)
    # print(out_value_new.to_frame())
    out_value_new.to_frame().to_csv(absolute_path(config['file']['out_value_new']))


if __name__ == '__main__':
    addrTxValue()
    txAddrValue()
    # process_in_out_value_repeat_address()

