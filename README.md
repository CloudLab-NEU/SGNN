# Illicit Social Accounts? Anti-money Laundering for Transactional Blockchains

This is the source code of our paper titled "Illicit Social Accounts? Anti-money Laundering for Transactional Blockchains".
We present the PyTorch implementation of SGNN and other ablation models.


## Model
![SGNN](figures%2FModel.jpg)

## Overview
The repository is organised as follows:

- `.config/` contains the configuration files for dataset and models.
- `.data/` contains the data.
- `.dataloader/` contains the preprocessing files for the data and the data loading files for models.
- `.error/` contains the custom error class files.
- `.figures/` contains the figures of the readme.md file.
- `.models/` contains the implementation of SGNN and other ablation models.
- `.result/` contains the result of SGNN.
- `.train/` contains the training startup files of the model in the experiments.
- `.utils/` contains the tool files composed of frequently used functions.

## Environment

- `python`==3.10.10 
- `torch`==2.0.0+cu118 (CUDA 11.8)
- `pandas`==2.0.3 
- `numpy`==1.25.1
- `scikit-learn`==1.3.0

## Data

We used the Ellipse++ dataset, available at https://github.com/git-disl/EllipticPlusPlus. 
Due to space limitations, we can only provide a small portion of the data in `.data/` folder.
Please refer to the [data.md](data%2Fdata.md) file under `.data/` folder for more information about the data.


## Run

Read the [data.md](data%2Fdata.md) file before running. 
Run the following code to train the current model:
```
python train\train_address.py 
```
If you want to modify the model parameters or run other ablation models, the relevant parameters in `.config/` need to be modified.

