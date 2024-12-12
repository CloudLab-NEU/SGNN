# -*- coding: utf-8 -*-


import os
import torch
import torch.backends.cudnn
from torch.nn import Module
from utils.config_utils import get_config_option


def get_device():
    """cuda device"""
    ctd = get_config_option("LB-GLAT", "training", "ctd")
    os.environ["CUDA_VISIBLE_DEVICES"] = ctd
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print("Cuda Available:{}, use {}!".format(use_cuda, device))
    return device


def get_paras_num(model: Module, name):
    total_num = sum(p.numel() for p in model.parameters())
    train_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {name}, Total paras number: {total_num}, Train paras number: {train_num}")
    return {'Total': total_num, 'Train': train_num}


def print_model_state_dict(model: Module):
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())


def print_optimizer_state_dict(optimizer):
    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])



