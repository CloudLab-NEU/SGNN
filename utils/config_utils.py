# -*- coding: utf-8 -*-

import os
import configparser
from utils.file_utils import config_path

"""
the config file utils
"""

def get_config_option(filename, section, option):
    """
    :param filename: config file name
    :param section: config section name
    :param option: config option
    :return: the option value
    """
    conf = configparser.ConfigParser()
    conf.read(config_path(filename), encoding='UTF-8')
    config_option = conf.get(section, option)
    return config_option


def get_config(filename):
    """
    :param filename: config file name
    :return: the option value
    """
    config = configparser.ConfigParser()
    config.read(config_path(filename), encoding='UTF-8')
    return config


if __name__ == '__main__':
    # get_config_allOptions("a")
    classes_name = get_config_option("dataset", "Elliptic", "classes_name")
    print(classes_name.split())



