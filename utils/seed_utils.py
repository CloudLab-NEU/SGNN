import random

import numpy as np
import torch
import torch_geometric


def setup_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch_geometric.seed_everything(seed)
    # torch.set_default_dtype(torch.float64)


# setup_seed(int(get_config_option("model", "gnn", "seed")))