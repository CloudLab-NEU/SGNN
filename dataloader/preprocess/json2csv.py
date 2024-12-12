# -*- coding: utf-8 -*-


import json
import csv
import pandas as pd

"""
Convert data from json to csv file
"""

def json2csv(json_path, csv_path):
    json_data = pd.read_json(json_path, lines=True)
    print(json_data.head())
    json_data.to_csv(csv_path, index=False)


def get_received_value():
    df = pd.read_csv('../../data/process/output_recevied_value.csv')
    df['output_address'] = df['output_address'].str.replace("[\\[\\'\\]]", '')
    df.to_csv('../../data/process/output_recevied_value.csv', index=False)


if __name__ == "__main__":
    json2csv('../../data/json/input_sent_value.json', '../../data/process/input_sent_value.csv')
    json2csv('../../data/json/output_recevied_value.json', '../../data/process/output_recevied_value.csv')
    get_received_value()
