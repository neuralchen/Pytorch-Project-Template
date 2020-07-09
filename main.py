#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: main.py
# Created Date: Monday April 6th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 9th July 2020 10:00:29 am
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################
import os
import shutil
import argparse

def getParameters():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--root', type=str, default="D:\\PatchFace\\PleaseWork\\wocao")
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
    "update.py",
    # "parameters.py",
    "env/config.json",
    "utilities/reporter.py",
    "utilities/yaml_config.py",
    "utilities/json_config.py",
    "utilities/sshupload.py"
]

if __name__ == "__main__":
    config = getParameters()

    root_dir = config.root
    template_root = "./related_scripts"

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    for item in dir_list:
        if not os.path.exists(os.path.join(root_dir,item)):
            os.makedirs(os.path.join(root_dir,item))
    for item in scripts_list:
        project_file = os.path.join(root_dir,item)
        if not os.path.exists(project_file):
            _,fullfilename = os.path.split(item)
            template_file = os.path.join(template_root,fullfilename)
            if os.path.exists(template_file):
                shutil.copyfile(template_file,project_file)
            else:
                script_file = open(project_file, "w", encoding="utf-8")
                script_file.close()