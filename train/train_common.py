from utils.config_utils import get_config_option
from utils.seed_utils import setup_seed

setup_seed(int(get_config_option("model", "gnn", "seed")))

from torch import nn
from error.NoModelError import NoModelError
from train.train_config import *


def address_model_criterion(model_name):
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

    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(criterion_weight))

    return model, criterion