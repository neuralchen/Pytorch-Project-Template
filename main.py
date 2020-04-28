#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: main.py
# Created Date: Monday April 6th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 7th April 2020 5:49:02 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################
import  os
import argparse

def getParameters():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--root', type=str, default="D:\\PatchFace\\PleaseWork\\multi-style-gan")
    return parser.parse_args()


dir_list = [
    "data_tools",
    "env",
    "train_configs",
    "train_scripts",
    "test_scripts",
    "utilities",
    "components"
]

scripts_list = [
    "main.py",
    "parameters.py",
    "env/config.json"
]

if __name__ == "__main__":
    config = getParameters()

    root_dir = config.root
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    for item in dir_list:
        if not os.path.exists(os.path.join(root_dir,item)):
            os.makedirs(os.path.join(root_dir,item))
    for item in scripts_list:
        if not os.path.exists(os.path.join(root_dir,item)):
            script_file = open(os.path.join(root_dir,item), "w", encoding="utf-8")
            script_file.close()