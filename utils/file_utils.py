# -*- coding: utf-8 -*-
import csv
import os
from pathlib import Path

def absolute_path(filepath):
    """
    :param filepath: other absolute path parts of the file relative to the root directory
    :return: the path of config file
    """
    cur_path = Path(os.path.abspath(os.path.dirname(__file__)))
    file_path = Path(cur_path.parent, filepath)
    return str(file_path)


def config_path(filename):
    """
    :param filename: config file name
    :return: the path of config file
    """
    cur_path = Path(os.path.abspath(os.path.dirname(__file__)))
    config_path = str(Path(cur_path.parent, "config", filename)) + ".conf"
    return config_path


def create_csv(filepath, columns):
    with open(filepath, mode='w', newline='', encoding='utf8') as cf:
        wf = csv.writer(cf)
        wf.writerow(columns)


def writerow_csv(filepath, data):
    with open(filepath, mode='a', newline='', encoding='utf8') as cf:
        wf = csv.writer(cf)
        wf.writerow(data)


if __name__ == '__main__':
    # get_config_path("dataset")
    absolute_path("\\data")
