# How to get data
The paths to all data in the code are documented in [path.conf](..%2Fconfig%2Fpath.conf).
We only uploaded some of the data because the files were too large to upload to github.
The .zip files need to be unzipped. 

## txId2hash.csv:
Alexander de-anonymized 99.5% transactions in 2019: https://habr.com/ru/articles/479178/ .
The data is available at [Kaggle](https://www.kaggle.com/datasets/alexbenzik/deanonymized-995-pct-of-elliptic-transactions).
Download it under `.data/` and name it txId2hash.csv.

## Files in .json/
The input_sent_value.json and output_recevied_value.json are available from [BigQuery](https://cloud.google.com/bigquery/public-data) based on txId2hash.csv.
Run [json2csv.py](..%2Fdataloader%2Fpreprocess%2Fjson2csv.py) to get the input_sent_value.csv and output_recevied_value.csv.

## Other Data
If you need additional data, you can download the Elliptic++ dataset on [Github](https://github.com/git-disl/EllipticPlusPlus) to the data folder (check [path.conf](..%2Fconfig%2Fpath.conf) for the exact path) and get the desired data by running the functions in the files of [preprocess/](..%2Fdataloader%2Fpreprocess) folder and [dataset_utils.py](..%2Futils%2Fdataset_utils.py).

## Run
Running [train_address.py](..%2Ftrain%2Ftrain_address.py) requires the 6 .npz files for [addr0_tx_mean] recorded in [path.conf](..%2Fconfig%2Fpath.conf) and address_inOrOut_tx_time_class.csv.
The .npz files are available by running get_all_np_list() in [dataset_utils.py](..%2Futils%2Fdataset_utils.py).
