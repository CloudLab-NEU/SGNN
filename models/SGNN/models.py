# -*- coding: utf-8 -*-

from utils.config_utils import get_config_option
from utils.seed_utils import setup_seed
setup_seed(int(get_config_option("model", "gnn", "seed")))
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from error.NoAttentionTypeError import NoAttentionTypeError
from models.SGNN.attention_layers import TransformerEncoder
from models.SGNN.layers import HANsBlock, HypergraphsBlock
from models.common_layers import MLP, FCsBlock, Project


class SGNN_addr_noAtt(nn.Module):

    def __init__(self, n_features, hetero_gnns_hidden: Tensor, hyper_gnns_hidden: Tensor,
                 mlp_hidden: Tensor, fcs_hidden: Tensor, n_classes,
                 metadata, heads, bias, use_attention, attention_mode, dropout, do_bn, device):
        super(SGNN_addr_noAtt, self).__init__()
        self.device = device
        self.hans = HANsBlock(n_features, hetero_gnns_hidden, metadata, heads, bias, dropout, do_bn)
        self.hgs = HypergraphsBlock(n_features, hyper_gnns_hidden, heads, bias,
                                    use_attention, attention_mode, dropout, do_bn)
        self.fcs = FCsBlock(n_features+hetero_gnns_hidden[-1]+hyper_gnns_hidden[-1], fcs_hidden, n_classes, bias, do_bn, dropout)


    def forward(self, hetero_data, hyper_data):
        # address
        addr_hetero_x = self.hans(hetero_data.x_dict, hetero_data.edge_index_dict)  # (addr_num, addr_feature)
        addr_hyper_x = self.hgs(hetero_data['address'].x,
                                hyper_data.hyperedge_index,
                                hyper_data.hyperedge_weight)  # (addr_num, addr_feature)
        # FCs
        addr_x = self.fcs(torch.cat((hetero_data['address'].x, addr_hetero_x, addr_hyper_x), 1))
        return addr_x



class SGNN_addr_att(nn.Module):

    def __init__(self, n_features, hetero_gnns_hidden: Tensor, hyper_gnns_hidden: Tensor, mlp_hidden: Tensor,
                 tfe_dim, tfe_mlp_hidden, tfe_depth, tfe_heads, tfe_head_dim, tfe_dropout, tfe_type,
                 fcs_hidden: Tensor, n_classes,
                 metadata, heads, bias, use_attention, attention_mode, dropout, do_bn, device):
        super(SGNN_addr_att, self).__init__()
        self.device = device
        self.hans = HANsBlock(n_features, hetero_gnns_hidden, metadata, heads, bias, dropout, do_bn)
        self.hgs = HypergraphsBlock(n_features, hyper_gnns_hidden, heads, bias,
                                    use_attention, attention_mode, dropout, do_bn)
        self.tfe_type = tfe_type
        if self.tfe_type in ["1st", "avg3"]:
            self.proj = Project(n_features, tfe_dim, bias)
        self.tfe = TransformerEncoder(tfe_dim, tfe_depth, tfe_heads, tfe_head_dim, tfe_mlp_hidden, tfe_type, tfe_dropout)
        self.fcs = FCsBlock(tfe_dim, fcs_hidden, n_classes, bias, do_bn, dropout)


    def forward(self, hetero_data, hyper_data):
        # address
        addr_hetero_x = self.hans(hetero_data.x_dict, hetero_data.edge_index_dict)  # (addr_num, addr_feature)
        addr_hyper_x = self.hgs(hetero_data['address'].x,
                                hyper_data.hyperedge_index,
                                hyper_data.hyperedge_weight)  # (addr_num, addr_feature)
        # atten
        if self.tfe_type in ["1st", "avg3"]:
            addr_x = self.proj(hetero_data['address'].x)
            addr_x = self.tfe(torch.stack((addr_x, addr_hetero_x, addr_hyper_x), dim=1))  # (addr_num, 3, tfe_dim)
        elif self.tfe_type == "avg2":
            addr_x = self.tfe(torch.stack((addr_hetero_x, addr_hyper_x), dim=1))  # (addr_num, 2, tfe_dim)
        else:
            raise NoAttentionTypeError("The tfe_type is error during training.")
        # FCs
        addr_x = self.fcs(addr_x)
        return addr_x


class SGNN_noHY_addr_att(nn.Module):

    def __init__(self, n_features, hetero_gnns_hidden: Tensor, mlp_hidden: Tensor,
                 tfe_dim, tfe_mlp_hidden, tfe_depth, tfe_heads, tfe_head_dim, tfe_dropout, tfe_type,
                 fcs_hidden: Tensor, n_classes,
                 metadata, heads, bias, use_attention, attention_mode, dropout, do_bn, device):
        super(SGNN_noHY_addr_att, self).__init__()
        self.device = device
        self.hans = HANsBlock(n_features, hetero_gnns_hidden, metadata, heads, bias, dropout, do_bn)
        self.tfe_type = tfe_type
        if self.tfe_type in ["1st", "avg3"]:
            self.proj = Project(n_features, tfe_dim, bias)
        self.tfe = TransformerEncoder(tfe_dim, tfe_depth, tfe_heads, tfe_head_dim, tfe_mlp_hidden, tfe_type, tfe_dropout)
        self.fcs = FCsBlock(tfe_dim, fcs_hidden, n_classes, bias, do_bn, dropout)


    def forward(self, hetero_data, hyper_data):
        # address
        addr_hetero_x = self.hans(hetero_data.x_dict, hetero_data.edge_index_dict)  # (addr_num, addr_feature)
        # atten
        if self.tfe_type in ["1st", "avg3"]:
            addr_x = self.proj(hetero_data['address'].x)
            addr_x = self.tfe(torch.stack((addr_x, addr_hetero_x), dim=1))  # (addr_num, 2, tfe_dim)
        else:
            raise NoAttentionTypeError("The tfe_type is error during training.")
        # FCs
        addr_x = self.fcs(addr_x)
        return addr_x
    

class SGNN_noHE_addr_att(nn.Module):

    def __init__(self, n_features, hyper_gnns_hidden: Tensor, mlp_hidden: Tensor,
                 tfe_dim, tfe_mlp_hidden, tfe_depth, tfe_heads, tfe_head_dim, tfe_dropout, tfe_type,
                 fcs_hidden: Tensor, n_classes,
                 metadata, heads, bias, use_attention, attention_mode, dropout, do_bn, device):
        super(SGNN_noHE_addr_att, self).__init__()
        self.device = device
        self.hgs = HypergraphsBlock(n_features, hyper_gnns_hidden, heads, bias,
                                    use_attention, attention_mode, dropout, do_bn)
        self.tfe_type = tfe_type
        if self.tfe_type in ["1st", "avg3"]:
            self.proj = Project(n_features, tfe_dim, bias)
        self.tfe = TransformerEncoder(tfe_dim, tfe_depth, tfe_heads, tfe_head_dim, tfe_mlp_hidden, tfe_type, tfe_dropout)
        self.fcs = FCsBlock(tfe_dim, fcs_hidden, n_classes, bias, do_bn, dropout)


    def forward(self, hetero_data, hyper_data):
        # address
        addr_hyper_x = self.hgs(hetero_data['address'].x,
                                hyper_data.hyperedge_index,
                                hyper_data.hyperedge_weight)  # (addr_num, addr_feature)
        # atten
        if self.tfe_type in ["1st", "avg3"]:
            addr_x = self.proj(hetero_data['address'].x)
            addr_x = self.tfe(torch.stack((addr_x, addr_hyper_x), dim=1))  # (addr_num, 2, tfe_dim)
        else:
            raise NoAttentionTypeError("The tfe_type is error during training.")
        # FCs
        addr_x = self.fcs(addr_x)
        return addr_x


def hyperedge_ma_sum(hyperedge_index, row, col, device):
    values = torch.tensor(np.ones(hyperedge_index.shape[1]), dtype=torch.float32).to(device)
    shape = torch.Size((row, col))
    return torch.sparse.FloatTensor(hyperedge_index, values, shape)


def hyperedge_ma_avg(hyperedge_index, row, col, device):
    values = torch.tensor(np.ones(hyperedge_index.shape[1]), dtype=torch.float64).to(device)
    hyperedge_ma = torch.sparse.FloatTensor(
        hyperedge_index,
        values,
        torch.Size((row, col))
    )
    degree = torch.sparse.sum(hyperedge_ma, dim=[0]).values()
    values_avg = values / degree[hyperedge_index[1].cpu().numpy()]
    hyperedge_ma_average = torch.sparse.FloatTensor(
        hyperedge_index,
        values_avg.float(),
        torch.Size((row, col))
    )
    return hyperedge_ma_average



