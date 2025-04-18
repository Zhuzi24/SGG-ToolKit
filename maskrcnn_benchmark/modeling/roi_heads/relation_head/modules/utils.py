# -#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Author :   Ch
# File    :   utils.py
# @Time   :   2021/7/8 15:00
import json
import os

import torch


def load_data(data_path):
    if os.path.isfile(data_path):
        file_name = os.path.basename(data_path)
        if file_name.endswith('json'):
            weight = json.load(open(data_path, 'r'))
            return torch.tensor(weight)
        else:
            return torch.load(data_path)
    raise 'invalid file path {}'.format(data_path)
