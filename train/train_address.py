# -*- coding: utf-8 -*-

from utils.config_utils import get_config_option
from utils.seed_utils import setup_seed

setup_seed(int(get_config_option("model", "gnn", "seed")))
import copy
import time
import torch
from torch import optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import pandas as pd

from dataloader.hetero_loader import get_hetero_train_test_loader
from dataloader.hyper_loader import get_hyper_train_test_loader
from error.NoModelError import NoModelError
from error.NoOptimError import NoOptimError
from train.get_class import addr_get
from train.train_config import *
from utils.common_utils import get_paras_num, print_model_state_dict, print_optimizer_state_dict
from utils.criterion_utils import get_indicators
from utils.file_utils import create_csv, writerow_csv


###########################################
# setup_seed(seed)
result_path = '../results'
result_path = rf'{result_path}/addr'

##########################################
# Load data
if model_name in ['SGNN_noHE_addr_att', 'SGNN_noHY_addr_att',
                  'SGNN_addr_att', 'SGNN_addr_noAtt', 'MLP_addr']:
    hetero_train_loader, hetero_test_loader = get_hetero_train_test_loader(rs_NP_ratio)
    hyper_train_loader, hyper_test_loader = get_hyper_train_test_loader()  # train_mask val_mask test_mask
else:
    hetero_train_loader = []
    hetero_test_loader = []
    hyper_train_loader = []
    hyper_test_loader = []

###########################################
# Model and optimizer

if model_name in ['SGNN_addr_att']:
    model = creat_SGNN_addr_att()
elif model_name in ['SGNN_addr_noAtt']:
    model = creat_SGNN_addr_noAtt()
elif model_name in ['SGNN_noHY_addr_att']:
    model = creat_SGNN_noHY_addr_att()
elif model_name in ['SGNN_noHE_addr_att']:
    model = creat_SGNN_noHE_addr_att()
elif model_name in ['MLP_addr']:
    model = creat_MLP_addr()
else:
    raise NoModelError("No model is specified during training.")
paras_num = get_paras_num(model, model_name)

optimizer = None
if opt == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=lr0, weight_decay=weight_decay)
elif opt == "AdamW":
    optimizer = optim.AdamW(model.parameters(), lr=lr0, weight_decay=weight_decay)
elif opt == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=lr0, weight_decay=weight_decay)
elif opt == "RMSprop":
    optimizer = optim.RMSprop(model.parameters(), lr=lr0, weight_decay=weight_decay)
else:
    raise NoOptimError("No optim is specified during training.")

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (1 + decay_rate * epoch),
                                        last_epoch=start_epoch - 1)
criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(criterion_weight))
# Cuda
model.to(device)
criterion.to(device)

###########################################
# tensorboard
gnns_hidden_str = "".join([str(i) for i in gnns_hidden.numpy()])
hetero_gnns_hidden_str = "".join([str(i) for i in hetero_gnns_hidden.numpy()])
hyper_gnns_hidden_str = "".join([str(i) for i in hyper_gnns_hidden.numpy()])
gnns_hidden_str = gnns_hidden_str + "_" + hetero_gnns_hidden_str + "_" + hyper_gnns_hidden_str
mlp_hidden_str = "".join([str(i) for i in mlp_hidden.numpy()])
fcs_hidden_str = "".join([str(i) for i in fcs_hidden.numpy()])
criterion_weight_str = "".join([str(cw) for cw in criterion_weight])

model_subfolder = gnns_hidden_str + "_" + mlp_hidden_str + "_" + fcs_hidden_str + "_t" + str(test_st) + str(
    test_et) + "_s" + str(down_sampling) + str(rs_NP_ratio) + f"_f{str(int(hetero_edge_forward))}r{str(int(hetero_edge_reverse))}"


model_result_filename = f'd{dropout}b{int(do_bn)}' \
                        f'_t{int(address_has_tx_feature)}{int(address_has_tx_feature_repeat)}' \
                        f'{int(tx_no_feature)}{int(address_tx_no_feature)}' \
                        f'_{int(tfe)}{tfe_dim}{tfe_mlp_hidden}{tfe_depth}{tfe_heads}{tfe_head_dim}{tfe_type}' \
                        f'_{int(hyper_use_attention)}{hyper_attention_mode}' \
                        f'_cw{criterion_weight_str}l{lr0}{decay_rate}e{epochs}'


logs_path = rf'{result_path}/{model_folder}/logs/{model_subfolder}/{model_result_filename}'
if not os.path.exists(logs_path):
    os.makedirs(logs_path)
    
###########################################
# total results
addr_total_results_columns = ["model_name",
                              "gnns_hidden_str", "mlp_hidden_str", "fcs_hidden_str",
                              "paras_num", "address_n_feature", "tx_n_feature",
                              "data_path", "address_has_tx_feature", "address_has_tx_feature_repeat",
                              "tx_no_feature", "address_tx_no_feature", "bias",
                              "dropout", "do_bn", "criterion_weight_str",
                              "lr0", "decay_rate", "opt", "epochs", "best_epoch",
                              "train_loss", "train_acc", "train_precision_pos", "train_precision_neg",
                              "train_recall_pos", "train_recall_neg", "train_F1_pos", "train_F1_neg", "train_AUC",
                              "test_loss", "test_acc", "test_precision_pos", "test_precision_neg",
                              "test_recall_pos", "test_recall_neg", "test_F1_pos", "test_F1_neg", "test_AUC",
                              "tfe", "tfe_dim", "tfe_mlp_hidden", "tfe_depth", "tfe_heads", "tfe_head_dim",
                              "tfe_dropout", "tfe_type", "test_st", "test_et",
                              "hyper_use_attention", "hyper_attention_mode", "down_sampling", "rs_NP_ratio", "hetero_edge_forward", "hetero_edge_reverse"]

addr_total_results_path = absolute_path(path_config['file']['addr_total_results'])
if not os.path.exists(addr_total_results_path):
    create_csv(addr_total_results_path, addr_total_results_columns)

result = [model_name, gnns_hidden_str, mlp_hidden_str, fcs_hidden_str, paras_num.get("Total"),
          address_n_feature, tx_n_feature, model_config['gnn']['data_path'],
          address_has_tx_feature, address_has_tx_feature_repeat, tx_no_feature, address_tx_no_feature,
          bias, dropout, do_bn, criterion_weight_str, lr0, decay_rate, opt, epochs]
result_end = [tfe, tfe_dim, tfe_mlp_hidden, tfe_depth, tfe_heads, tfe_head_dim, tfe_dropout, tfe_type,
              test_st, test_et, hyper_use_attention, hyper_attention_mode, down_sampling, rs_NP_ratio, hetero_edge_forward, hetero_edge_reverse]

###########################################
# Train
def train_epoch(epoch, batch_train_times, batch_inference_times):
    model.train()
    print("=" * 30 + f"{model_name} Train Epoch {epoch}" + "=" * 30)
    print(f"Learning Rate: {scheduler.get_lr()}")  # Display the current learning rate.
    train_loss = 0
    samples_num = 0
    train_y = torch.Tensor()
    train_y_pred = torch.Tensor()
    for i in tqdm(range(len(hetero_train_loader)), desc=f"Train Epoch {epoch}: "):
        hetero_data = hetero_train_loader[i].to(device)
        hyper_data = hyper_train_loader[i].to(device)

        start_train = torch.cuda.Event(enable_timing=True)
        end_train = torch.cuda.Event(enable_timing=True)
        start_train.record()

        optimizer.zero_grad()
        output_mask = model(hetero_data=hetero_data, hyper_data=hyper_data)[hetero_data['address'].train_mask]
        y_mask = hetero_data['address'].y[hetero_data['address'].train_mask]
        loss = criterion(output_mask, y_mask)
        train_loss += loss.item() * y_mask.size()[0]
        samples_num += y_mask.size()[0]
        loss.backward()  # Derive gradients.
        optimizer.step()

        end_train.record()
        torch.cuda.synchronize()
        train_time = start_train.elapsed_time(end_train)
        batch_train_times[i] += train_time


        y_pred_mask = output_mask.argmax(dim=1).detach().cpu()  # Use the class with highest probability.
        train_y = torch.cat((train_y, y_mask.detach().cpu()), 0)
        train_y_pred = torch.cat((train_y_pred, y_pred_mask), 0)

    scheduler.step()  # Update Learning Rate
    train_loss = train_loss / samples_num
    train_acc, train_precision, train_recall, train_F1, train_AUC = get_indicators(train_y.numpy(),
                                                                                   train_y_pred.numpy())

    if not fastmode:
        test_loss, test_acc, test_precision, test_recall, test_F1, test_AUC, batch_inference_times = test(batch_inference_times)
        print_epoch(epoch, train_loss, train_acc, train_precision, train_recall, train_F1, train_AUC,
                    test_loss, test_acc, test_precision, test_recall, test_F1, test_AUC)

        print(f"batch_train_times: {batch_train_times}ms")
        print(f"batch_inference_times: {batch_inference_times}ms")
        return (train_loss, train_acc, train_precision[1], train_precision[0],
            train_recall[1], train_recall[0], train_F1[1], train_F1[0], train_AUC,
            test_loss, test_acc, test_precision[1], test_precision[0],
            test_recall[1], test_recall[0], test_F1[1], test_F1[0], test_AUC), batch_train_times, batch_inference_times
    else:
        print_epoch(epoch, train_loss, train_acc, train_precision, train_recall, train_F1, train_AUC)
        return (train_loss, train_acc, train_precision[1], train_precision[0],
            train_recall[1], train_recall[0], train_F1[1], train_F1[0], train_AUC,
            0, 0, 0, 0, 0, 0, 0, 0, 0), batch_train_times, batch_inference_times

def test(batch_inference_times):
    model.eval()
    test_loss = 0
    samples_num = 0
    test_y = torch.Tensor()
    test_y_pred = torch.Tensor()
    for i in tqdm(range(len(hetero_test_loader)), desc="Test Data: "):
        hetero_data = hetero_test_loader[i].to(device)
        hyper_data = hyper_test_loader[i].to(device)

        start_infer = torch.cuda.Event(enable_timing=True)
        end_infer = torch.cuda.Event(enable_timing=True)
        start_infer.record()

        output_mask = model(hetero_data=hetero_data, hyper_data=hyper_data)[hetero_data['address'].test_mask]

        end_infer.record()
        torch.cuda.synchronize()
        inference_time = start_infer.elapsed_time(end_infer)
        batch_inference_times[i] += inference_time

        y_mask = hetero_data['address'].y[hetero_data['address'].test_mask]
        loss = criterion(output_mask, y_mask)
        test_loss += loss.item() * y_mask.size()[0]
        samples_num += y_mask.size()[0]

        y_pred_mask = output_mask.argmax(dim=1).detach().cpu()  # Use the class with highest probability.
        test_y = torch.cat((test_y, y_mask.detach().cpu()), 0)
        test_y_pred = torch.cat((test_y_pred, y_pred_mask), 0)

    test_loss = test_loss / samples_num
    test_acc, test_precision, test_recall, test_F1, test_AUC = \
        get_indicators(test_y.numpy(), test_y_pred.numpy())
    return test_loss, test_acc, test_precision, test_recall, test_F1, test_AUC, batch_inference_times

def print_epoch(epoch, train_loss, train_acc, train_precision, train_recall, train_F1, train_AUC,
                test_loss=0.0, test_acc=0.0, test_precision=None, test_recall=None, test_F1=None, test_AUC=0.0):
    if not fastmode:
        print(f"[Epoch: {epoch}]: \n"
              f'[Train] Loss: {train_loss}, Accuracy: {train_acc}, '
              f'Precision P: {train_precision[1]} N: {train_precision[0]}, '  # Positive sample, negative sample
              f'Recall P: {train_recall[1]} N: {train_recall[0]}, '
              f'F1-score P: {train_F1[1]} N: {train_F1[0]}, AUC {train_AUC} \n'
              f'[Test] Loss: {test_loss}, Accuracy: {test_acc}, '
              f'Precision P: {test_precision[1]} N: {test_precision[0]}, '  # Positive sample, negative sample
              f'Recall P: {test_recall[1]} N: {test_recall[0]}, '
              f'F1-score P: {test_F1[1]} N: {test_F1[0]}, AUC {test_AUC}')
    else:
        print(f"[Epoch: {epoch}]: \n"
              f'[Train] Loss: {train_loss}, Accuracy: {train_acc}, '
              f'Precision P: {train_precision[1]} N: {train_precision[0]}, '  # Positive sample, negative sample
              f'Recall P: {train_recall[1]} N: {train_recall[0]}, '
              f'F1-score P: {train_F1[1]} N: {train_F1[0]}, AUC {train_AUC}')

def train(epochs):
    print("=" * 30 + model_name + "=" * 30)
    print(model)
    print_model_state_dict(model)
    print_optimizer_state_dict(optimizer)
    columns = ["epoch", "train_loss", "train_acc", "train_precision_pos", "train_precision_neg",
               "train_recall_pos", "train_recall_neg", "train_F1_pos", "train_F1_neg", "train_AUC",
               "test_loss", "test_acc", "test_precision_pos", "test_precision_neg",
               "test_recall_pos", "test_recall_neg", "test_F1_pos", "test_F1_neg", "test_AUC"]
    results = pd.DataFrame(np.zeros(shape=(epochs, len(columns))), columns=columns)

    results_dir = f"{result_path}/{model_folder}/results/{model_subfolder}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    model_dir = f"{result_path}/{model_folder}/paras/{model_subfolder}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    t_start = time.time()
    best_epoch = -1
    best_model = None
    min_test_loss = float('inf')

    batch_train_times = [0] * len(hetero_train_loader)
    batch_inference_times = [0] * len(hetero_test_loader)
    for epoch in range(epochs):
        evals, batch_train_times, batch_inference_times = train_epoch(epoch, batch_train_times, batch_inference_times)
        results.iloc[epoch, 0] = epoch + 1
        results.iloc[epoch, 1:] = evals

        # save best_model
        if epoch + 1 > min_epoch and evals[9] < min_test_loss:
            best_epoch = epoch + 1
            min_test_loss = evals[9]
            best_model = copy.deepcopy(model)

    # average_train_times = [train_time / epochs for train_time in batch_train_times]
    # average_inference_times = [inference_time / epochs for inference_time in batch_inference_times]
    # average_train_times_type = ["average_train_times"]
    # average_train_times_type.extend(average_train_times)
    # average_inference_times_type = ["average_inference_times"]
    # average_inference_times_type.extend(average_inference_times)
    # if model_name in ["SGNN_addr_att"]:
    #     cost_path = absolute_path(get_config_option("path", "file", "cost_model"))
    #     cost_columns = ["type"]
    #     cost_columns.extend([f"Time step {i + 1}" for i in range(len(hetero_train_loader))])
    #     if not os.path.exists(cost_path):
    #         create_csv(cost_path, cost_columns)
    #     writerow_csv(cost_path, average_train_times_type)
    #     writerow_csv(cost_path, average_inference_times_type)

    # print("Optimization Finished!")
    t_total = time.time() - t_start
    # print("Total time elapsed: {:.4f}s".format(t_total))
    results["epoch"] = results["epoch"].astype(int)

    best_result = results[results["epoch"] == best_epoch].to_numpy()[0][1:].tolist()
    result.append(best_epoch)
    result.extend(best_result)
    result.extend(result_end)
    result.append(t_total)
    writerow_csv(addr_total_results_path, result)

    results.to_csv(
        f'{results_dir}/{model_result_filename}_{best_epoch}.csv',
        mode='w', header=True, index=False)

    best_state = {'model': best_model.state_dict()}
    model_path = f'{model_dir}/{model_result_filename}_{best_epoch}.pth'
    torch.save(
        best_state,
        model_path)
    print(results)
    print(
        f"best_epoch: {best_epoch}, min_test_loss: {min_test_loss}, best_result: {' '.join([str(br) for br in best_result])}")

    # addr_get(model, criterion, model_path, hetero_train_loader, hetero_test_loader, hyper_train_loader,
    #          hyper_test_loader)

train(epochs)

