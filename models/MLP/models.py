# -*- coding: utf-8 -*-

from utils.config_utils import get_config_option
from utils.seed_utils import setup_seed
setup_seed(int(get_config_option("model", "gnn", "seed")))
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from models.common_layers import FCsBlock

class MLP_addr(nn.Module):

    def __init__(self, n_features, fcs_hidden: Tensor, n_classes, bias, dropout, do_bn, device):
        super(MLP_addr, self).__init__()
        self.device = device
        self.fcs = FCsBlock(n_features, fcs_hidden, n_classes, bias, do_bn, dropout)

    def forward(self, hetero_data, hyper_data):
        # FCs
        tx_x = self.fcs(hetero_data['address'].x)
        return tx_x