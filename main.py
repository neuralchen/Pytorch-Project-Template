#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: main.py
# Created Date: Monday April 6th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Thursday, 9th July 2020 10:32:22 am
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

scripts_list = [ # [source filename in related_scripts]>[target file path]
    "main.py>main.py",
    "update.py>update.py",
    "update.ico>update.ico",
    "data_loader.py>data_tools/data_loader.py",
    "config.json>env/config.json",
    "reporter.py>utilities/reporter.py",
    "yaml_config.py>utilities/yaml_config.py",
    "json_config.py>utilities/json_config.py",
    "sshupload.py>utilities/sshupload.py"
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
        source,target=item.split(">")
        source = source.strip()
        target = target.strip()
        project_file = os.path.join(root_dir,target)
        if not os.path.exists(project_file):
            _,fullfilename = os.path.split(source)
            template_file = os.path.join(template_root,fullfilename)
            if os.path.exists(template_file):
                shutil.copyfile(template_file,project_file)
            else:
                script_file = open(project_file, "w", encoding="utf-8")
                script_file.close()