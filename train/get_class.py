# -*- coding: utf-8 -*-

from utils.config_utils import get_config_option
from utils.seed_utils import setup_seed

setup_seed(int(get_config_option("model", "gnn", "seed")))

import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils.criterion_utils import get_indicators
from dataloader.address_loader import addr_hash_time_train_test
from dataloader.hetero_loader import get_hetero_train_test_loader
from dataloader.hyper_loader import get_hyper_train_test_loader
from utils.config_utils import get_config
from utils.file_utils import absolute_path
from train.train_common import address_model_criterion


#######################################

model_config = get_config('model')
ctd = model_config['gnn']['ctd']
no_cuda = model_config['gnn']['no_cuda'] == str(True)
seed = int(model_config['gnn']['seed'])
# setup_seed(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = ctd
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


#######################################


def addr_get(model, criterion, model_path, hetero_train_loader, hetero_test_loader, hyper_train_loader,
             hyper_test_loader):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    model_path = model_path.replace('.pth', '')
    addr_train_path = rf"{model_path}/addr_analysis_train.csv"
    addr_test_path = rf"{model_path}/addr_analysis_test.csv"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    train_loss, train_y, train_y_pred, train_y_mask = addr_get_train(model, criterion, hetero_train_loader,
                                                                     hyper_train_loader)
    test_loss, test_y, test_y_pred, test_y_mask = addr_get_test(model, criterion, hetero_test_loader,
                                                                hyper_test_loader)

    train_data, test_data = addr_hash_time_train_test()  # [hash, Time step, label]
    train_data_mask = train_data[train_y_mask.astype(bool)]
    test_data_mask = test_data[test_y_mask.astype(bool)]

    columns = ['address', 'Time step', 'class', 'class_pred']
    train_data_mask = pd.DataFrame(np.hstack([train_data_mask, train_y_pred[:, np.newaxis]]), columns=columns)
    test_data_mask = pd.DataFrame(np.hstack([test_data_mask, test_y_pred[:, np.newaxis]]), columns=columns)
    train_data_mask.to_csv(addr_train_path, index=False)
    test_data_mask.to_csv(addr_test_path, index=False)

    test_acc, test_precision, test_recall, test_F1, test_AUC = \
        get_indicators(np.array(test_y), np.array(test_y_pred))
    print(f'Accuracy: {test_acc}, '
          f'Precision P: {test_precision[1]} N: {test_precision[0]}, '  # Positive sample, negative sample
          f'Recall P: {test_recall[1]} N: {test_recall[0]}, '
          f'F1-score P: {test_F1[1]} N: {test_F1[0]}, AUC {test_AUC}')
    print(f"[addr_analysis]  train_loss: {train_loss}, test_loss: {test_loss}")


#######################################


def addr_get_test(model, criterion, hetero_test_loader, hyper_test_loader):
    test_loss = 0
    samples_num = 0
    test_y = torch.Tensor()
    test_y_pred = torch.Tensor()
    test_y_mask = torch.Tensor()
    for i in tqdm(range(len(hetero_test_loader)), desc="Test Data: "):
        hetero_data = hetero_test_loader[i].to(device)
        hyper_data = hyper_test_loader[i].to(device)

        test_mask = hetero_data['address'].test_mask
        output_mask = model(hetero_data=hetero_data, hyper_data=hyper_data)[test_mask]
        y_mask = hetero_data['address'].y[test_mask]
        loss = criterion(output_mask, y_mask)
        test_loss += loss.item() * y_mask.size()[0]
        samples_num += y_mask.size()[0]

        y_pred_mask = output_mask.argmax(dim=1).detach().cpu()  # Use the class with highest probability.
        test_y = torch.cat((test_y, y_mask.detach().cpu()), 0)
        test_y_pred = torch.cat((test_y_pred, y_pred_mask), 0)
        test_y_mask = torch.cat((test_y_mask, test_mask.detach().cpu()), 0)  # mask

    test_loss = test_loss / samples_num
    # test_acc, test_precision, test_recall, test_F1, test_AUC = \
    #     get_indicators(test_y.numpy(), test_y_pred.numpy())
    return test_loss, test_y.numpy().astype(int), test_y_pred.numpy().astype(int), test_y_mask.numpy().astype(int)


def addr_get_train(model, criterion, hetero_train_loader, hyper_train_loader):
    train_loss = 0
    samples_num = 0
    train_y = torch.Tensor()
    train_y_pred = torch.Tensor()
    train_y_mask = torch.Tensor()
    for i in tqdm(range(len(hetero_train_loader)), desc="Test Data: "):
        hetero_data = hetero_train_loader[i].to(device)
        hyper_data = hyper_train_loader[i].to(device)

        train_mask = hetero_data['address'].train_mask
        output_mask = model(hetero_data=hetero_data, hyper_data=hyper_data)[train_mask]
        y_mask = hetero_data['address'].y[train_mask]
        loss = criterion(output_mask, y_mask)
        train_loss += loss.item() * y_mask.size()[0]
        samples_num += y_mask.size()[0]

        y_pred_mask = output_mask.argmax(dim=1).detach().cpu()  # Use the class with highest probability.
        train_y = torch.cat((train_y, y_mask.detach().cpu()), 0)
        train_y_pred = torch.cat((train_y_pred, y_pred_mask), 0)
        train_y_mask = torch.cat((train_y_mask, train_mask.detach().cpu()), 0)  # mask

    train_loss = train_loss / samples_num
    # train_acc, train_precision, train_recall, train_F1, train_AUC = \
    #     get_indicators(train_y.numpy(), train_y_pred.numpy())
    return train_loss, train_y.numpy().astype(int), train_y_pred.numpy().astype(int), train_y_mask.numpy().astype(int)


#######################################

def test_addr_get(model_name):
    model_config = get_config('model')
    ctd = model_config['gnn']['ctd']
    no_cuda = model_config['gnn']['no_cuda'] == str(True)
    os.environ["CUDA_VISIBLE_DEVICES"] = ctd
    use_cuda = not no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model_path = r"./results/addr/SGNN_addr_att/paras/now/SGNN.pth"
    model_path = absolute_path(model_path)
    model, criterion = address_model_criterion(model_name)
    model.to(device)
    criterion.to(device)

    hetero_train_loader, hetero_test_loader = get_hetero_train_test_loader(5)
    hyper_train_loader, hyper_test_loader = get_hyper_train_test_loader()  # train_mask val_mask test_mask

    addr_get(model, criterion, model_path, hetero_train_loader, hetero_test_loader, hyper_train_loader,
             hyper_test_loader)


if __name__ == '__main__':
    test_addr_get(model_name="SGNN_addr_att")