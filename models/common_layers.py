# -*- coding: utf-8 -*-

from utils.config_utils import get_config_option
from utils.seed_utils import setup_seed
setup_seed(int(get_config_option("model", "gnn", "seed")))
import torch
import torch.nn as nn
from torch import Tensor


class FCsBlock(nn.Module):
    """Linear part"""

    def __init__(self, n_features, fcs_hidden: Tensor, n_classes, bias, do_bn, dropout):
        super(FCsBlock, self).__init__()
        dims = [n_features] + fcs_hidden.tolist()  # dims length = layer_num+1
        self.fcs = nn.ModuleList([
            FCLayer(dims[i], dims[i + 1], do_bn, dropout, bias) for i in range(len(dims) - 1)  # Build layer i+1 Linear
        ])
        self.out = nn.Linear(dims[-1], n_classes, bias=bias)  # (batch, n_classes)

    def forward(self, x):
        for lin in self.fcs:
            x = lin(x)  # (batch, linears_hidden[i])
        x = self.out(x)  # (batch, n_classes)
        return x


class FCLayer(nn.Module):
    def __init__(self, in_features, out_features, do_bn, dropout, bias=True):
        super(FCLayer, self).__init__()
        self.lin = nn.Linear(in_features, out_features, bias=bias)
        self.do_bn = do_bn
        if do_bn:
            self.bn = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.lin(x)
        if self.do_bn and x.shape[0] > 1:
            h = self.bn(h)
        h = self.relu(h)
        h = self.dropout(h)
        return h


class MLP(nn.Module):
    """Linear part"""

    def __init__(self, n_features, mlp_hidden: Tensor, bias, do_bn, dropout):
        super(MLP, self).__init__()
        dims = [n_features] + mlp_hidden.tolist()  # dims length = layer_num+1
        self.mlp = nn.ModuleList([
            FCLayer(dims[i], dims[i + 1], do_bn, dropout, bias) for i in range(len(dims) - 1)  # Build layer i+1 Linear
        ])

    def forward(self, x):
        for lin in self.mlp:
            x = lin(x)  # (batch, linears_hidden[i])
        return x


class Project(nn.Module):

    def __init__(self, in_features, out_features, bias):
        super(Project, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(in_features, out_features, bias=bias),
            nn.ReLU()
        )

    def forward(self, x):
        return self.seq(x)
