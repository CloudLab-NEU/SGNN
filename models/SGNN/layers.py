# -*- coding: utf-8 -*-

from utils.config_utils import get_config_option
from utils.seed_utils import setup_seed
setup_seed(int(get_config_option("model", "gnn", "seed")))
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import HANConv, HypergraphConv, HGTConv, HeteroConv, GATConv

"""
HANsBlock
HyperGraphsBlock
"""


class HANsBlock(nn.Module):

    def __init__(self, n_features, gnns_hidden: Tensor, metadata, heads, bias, dropout, do_bn):
        super(HANsBlock, self).__init__()
        dims = [n_features] + gnns_hidden.tolist()  # dims length = layer_num+1

        self.lin_dict = torch.nn.ModuleDict()
        for i in range(len(metadata[0])):
            node_type = metadata[0][i]
            self.lin_dict[node_type] = nn.Linear(metadata[2][i], dims[0], bias=bias)

        self.hans = nn.ModuleList([
            HANLayer(in_channels=dims[i], out_channels=dims[i + 1], metadata=metadata, heads=heads,
                     do_bn=do_bn, dropout=dropout) for i in range(len(dims) - 1)  # Build layer i+1 HAN
        ])

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()
        for han in self.hans:
            x_dict = han(x_dict, edge_index_dict)
        return x_dict['address']

class HANLayer(nn.Module):

    def __init__(self, in_channels, out_channels, metadata, heads, do_bn, dropout):
        super(HANLayer, self).__init__()
        self.han = HANConv(in_channels=in_channels, out_channels=out_channels, metadata=metadata, heads=heads)

        self.relu_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            self.relu_dict[node_type] = nn.Sequential(
                nn.BatchNorm1d(out_channels) if do_bn else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.han(x_dict, edge_index_dict)
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.relu_dict[node_type](x)
        return x_dict


class HypergraphsBlock(nn.Module):
    def __init__(self, n_features, gnns_hidden: Tensor, heads, bias, use_attention, attention_mode, dropout,
                 do_bn):
        super(HypergraphsBlock, self).__init__()
        dims = [n_features] + gnns_hidden.tolist()  # dims length = layer_num+1
        self.hgs = nn.ModuleList([
            HypergraphLayer(in_channels=dims[i], out_channels=dims[i + 1], heads=heads, bias=bias,
                            do_bn=do_bn, dropout=dropout) for i in range(len(dims) - 1)
            # Build layer i+1 HAN
        ])

    def forward(self, x, hyperedge_index, hyperedge_weight):
        for hg in self.hgs:
            x = hg(x, hyperedge_index, hyperedge_weight)
        return x


class HypergraphLayer(nn.Module):

    def __init__(self, in_channels, out_channels, heads, bias, do_bn, dropout):
        super(HypergraphLayer, self).__init__()
        self.hg = HypergraphConv(in_channels=in_channels, out_channels=out_channels, heads=heads, bias=bias)
        self.do_bn = do_bn
        if do_bn:
            self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hyperedge_index, hyperedge_weight):
        x = self.hg(x=x, hyperedge_index=hyperedge_index, hyperedge_weight=hyperedge_weight)
        if self.do_bn:
            x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x





