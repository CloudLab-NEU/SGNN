# -*- coding: utf-8 -*-

import os
import torch
import numpy as np

from models.MLP.models import MLP_addr
from models.SGNN.models import SGNN_addr_att, SGNN_noHY_addr_att, SGNN_noHE_addr_att, SGNN_addr_noAtt
from utils.config_utils import get_config
from utils.file_utils import absolute_path

model_config = get_config('model')

ctd = model_config['gnn']['ctd']
no_cuda = model_config['gnn']['no_cuda'] == str(True)
fastmode = model_config['gnn']['fastmode'] == str(True)
seed = int(model_config['gnn']['seed'])

address_tx_no_feature = model_config['gnn']['address_tx_no_feature'] == str(True)
tx_no_feature = model_config['gnn']['tx_no_feature'] == str(True)

address_has_tx_feature = model_config['gnn']['address_has_tx_feature'] == str(True)
address_has_tx_feature_repeat = model_config['gnn']['address_has_tx_feature_repeat'] == str(True)
if address_tx_no_feature:
    address_n_feature = int(model_config['gnn']['address_n_feature_no_feature'])
    tx_n_feature = int(model_config['gnn']['tx_n_feature_no_feature'])
else:
    address_n_feature = int(
        model_config['gnn']['address_n_feature_with_tx_feature']) if address_has_tx_feature or address_has_tx_feature_repeat else int(
        model_config['gnn']['address_n_feature'])
    tx_n_feature = int(
        model_config['gnn']['tx_n_feature_no_feature']) if tx_no_feature else int(
        model_config['gnn']['tx_n_feature'])

hetero_gnns_hidden = torch.LongTensor(list(map(int, model_config['gnn']['hetero_gnns_hidden'].split())))
hyper_gnns_hidden = torch.LongTensor(list(map(int, model_config['gnn']['hyper_gnns_hidden'].split())))

gnns_hidden = torch.LongTensor(list(map(int, model_config['gnn']['gnns_hidden'].split())))

mlp_hidden = torch.LongTensor(list(map(int, model_config['gnn']['mlp_hidden'].split())))
fcs_hidden = torch.LongTensor(list(map(int, model_config['gnn']['fcs_hidden'].split())))
n_classes = int(model_config['gnn']['n_classes'])
gnn_heads = int(model_config['gnn']['gnn_heads'])
bias = model_config['gnn']['bias'] == str(True)
hyper_use_attention = model_config['gnn']['hyper_use_attention'] == str(True)
hyper_attention_mode = model_config['gnn']['hyper_attention_mode']
dropout = float(model_config['gnn']['dropout'])
do_bn = model_config['gnn']['do_bn'] == str(True)


tfe = model_config['gnn']['tfe'] == str(True)
tfe_dim = hetero_gnns_hidden[-1].item() if tfe else 0
tfe_mlp_hidden = int(model_config['gnn']['tfe_mlp_hidden']) if tfe else 0
tfe_depth = int(model_config['gnn']['tfe_depth']) if tfe else 0
tfe_heads = int(model_config['gnn']['tfe_heads']) if tfe else 0
tfe_head_dim = int(model_config['gnn']['tfe_head_dim']) if tfe else 0
tfe_dropout = float(model_config['gnn']['dropout']) if tfe else 0
tfe_type = model_config['gnn']['tfe_type'] if tfe else 0


opt = model_config['gnn']['opt']
lr0 = float(model_config['gnn']['lr0'])
decay_rate = float(model_config['gnn']['decay_rate'])
weight_decay = float(model_config['gnn']['weight_decay'])
criterion_weight = np.array(list(map(float, model_config.get("gnn", "criterion_weight").split())))
epochs = int(model_config['gnn']['epochs'])
start_epoch = int(model_config['gnn']['start_epoch'])
min_epoch = int(model_config['gnn']['min_epoch'])


model_folder = model_config['gnn']['model_folder']
model_name = model_config['gnn']['model_name']

hetero_edge_reverse = model_config['gnn']['hetero_edge_reverse'] == str(True)
hetero_edge_forward = model_config['gnn']['hetero_edge_forward'] == str(True)

hetero_edge_type = []
if hetero_edge_reverse:
    hetero_edge_type.append(('address', 'flow_reverse', 'transaction'))
    hetero_edge_type.append(('transaction', 'flow_reverse', 'address'))
    r_metadata = (['address', 'transaction'],
            [('address', 'flow_reverse', 'transaction'), ('address', 'flow_reverse', 'transaction')],
            [address_n_feature, tx_n_feature])
if hetero_edge_forward:
    hetero_edge_type.append(('address', 'flow', 'transaction'))
    hetero_edge_type.append(('transaction', 'flow', 'address'))
    f_metadata = (['address', 'transaction'],
            [('address', 'flow', 'transaction'), ('address', 'flow', 'transaction')],
            [address_n_feature, tx_n_feature])

metadata = (['address', 'transaction'],
            hetero_edge_type,
            [address_n_feature, tx_n_feature])

path_config = get_config('path')
# result_path = absolute_path(path_config['file']['result_path'])
result_path = '../results'

dataset_path = get_config('dataset')
down_sampling = dataset_path['Elliptic++']['down_sampling'] == str(True)
rs_NP_ratio = float(dataset_path['Elliptic++']['rs_NP_ratio'])
test_st = int(dataset_path['Elliptic++']['test_st'])
test_et = int(dataset_path['Elliptic++']['test_et'])

# cuda
os.environ["CUDA_VISIBLE_DEVICES"] = ctd
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print("Cuda Available:{}, use {}!".format(use_cuda, device))


# Initialize model function


def creat_SGNN_addr_noAtt():
    return SGNN_addr_noAtt(
        n_features=address_n_feature,
        hetero_gnns_hidden=hetero_gnns_hidden,
        hyper_gnns_hidden=hyper_gnns_hidden,
        mlp_hidden=mlp_hidden,
        fcs_hidden=fcs_hidden,
        n_classes=n_classes,
        metadata=metadata,
        heads=gnn_heads,
        bias=bias,
        use_attention=hyper_use_attention,
        attention_mode=hyper_attention_mode,
        dropout=dropout,
        do_bn=do_bn,
        device=device
    )


def creat_SGNN_addr_att():
    return SGNN_addr_att(
        n_features=address_n_feature,
        hetero_gnns_hidden=hetero_gnns_hidden,
        hyper_gnns_hidden=hyper_gnns_hidden,
        mlp_hidden=mlp_hidden,
        tfe_dim=tfe_dim,
        tfe_mlp_hidden=tfe_mlp_hidden,
        tfe_depth=tfe_depth,
        tfe_heads=tfe_heads,
        tfe_head_dim=tfe_head_dim,
        tfe_dropout=tfe_dropout,
        tfe_type=tfe_type,
        fcs_hidden=fcs_hidden,
        n_classes=n_classes,
        metadata=metadata,
        heads=gnn_heads,
        bias=bias,
        use_attention=hyper_use_attention,
        attention_mode=hyper_attention_mode,
        dropout=dropout,
        do_bn=do_bn,
        device=device
    )


def creat_SGNN_noHY_addr_att():
    return SGNN_noHY_addr_att(
        n_features=address_n_feature,
        hetero_gnns_hidden=hetero_gnns_hidden,
        mlp_hidden=mlp_hidden,
        tfe_dim=tfe_dim,
        tfe_mlp_hidden=tfe_mlp_hidden,
        tfe_depth=tfe_depth,
        tfe_heads=tfe_heads,
        tfe_head_dim=tfe_head_dim,
        tfe_dropout=tfe_dropout,
        tfe_type=tfe_type,
        fcs_hidden=fcs_hidden,
        n_classes=n_classes,
        metadata=metadata,
        heads=gnn_heads,
        bias=bias,
        use_attention=hyper_use_attention,
        attention_mode=hyper_attention_mode,
        dropout=dropout,
        do_bn=do_bn,
        device=device
    )


def creat_SGNN_noHE_addr_att():
    return SGNN_noHE_addr_att(
        n_features=address_n_feature,
        hyper_gnns_hidden=hyper_gnns_hidden,
        mlp_hidden=mlp_hidden,
        tfe_dim=tfe_dim,
        tfe_mlp_hidden=tfe_mlp_hidden,
        tfe_depth=tfe_depth,
        tfe_heads=tfe_heads,
        tfe_head_dim=tfe_head_dim,
        tfe_dropout=tfe_dropout,
        tfe_type=tfe_type,
        fcs_hidden=fcs_hidden,
        n_classes=n_classes,
        metadata=metadata,
        heads=gnn_heads,
        bias=bias,
        use_attention=hyper_use_attention,
        attention_mode=hyper_attention_mode,
        dropout=dropout,
        do_bn=do_bn,
        device=device
    )


def creat_MLP_addr():
    return MLP_addr(
        n_features=address_n_feature,
        fcs_hidden=fcs_hidden,
        n_classes=n_classes,
        bias=bias,
        dropout=dropout,
        do_bn=do_bn,
        device=device
    )
