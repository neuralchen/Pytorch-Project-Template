#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: main.py
# Created Date: Monday April 6th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Friday, 15th January 2021 12:26:24 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################
import os
import shutil
import argparse

def getParameters():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--root', type=str, default="H:\\TEST2\\Code")
    return parser.parse_args()


dir_list = [
    "data_tools",
    "env",
    "train_yamls",
    "train_scripts",
    "test_scripts",
    "utilities",
    "components",
    "losses"
]

scripts_list = [ # [source filename in related_scripts]>[target file path]
    "main.py>main.py",
    "UpdateGUI.py>UpdateGUI.py",
    "update.ico>update.ico",
    "data_loader.py>data_tools/data_loader.py",
    "reporter.py>utilities/reporter.py",
    "yaml_config.py>utilities/yaml_config.py",
    "json_config.py>utilities/json_config.py",
    "sshupload.py>utilities/sshupload.py",
    "checkpoint_manager.py>utilities/checkpoint_manager.py",
    "figure.py>utilities/figure.py",
    "README.md>README.md",
    "trainer_script_template.py>train_scripts/trainer_script_template.py",
    "tester_script_template.py>test_scripts/tester_script_template.py",
    "learningrate_scheduler.py>utilities/learningrate_scheduler.py",
    "env.json>env/env.json",
    "utilities.py>utilities/utilities.py",
    "save_heatmap.py>utilities/save_heatmap.py",
    "vgg_arch.py>losses/vgg_arch.py",
    "PerceptualLoss.py>losses/PerceptualLoss.py",
    "dataloader_hdf5.py>data_tools/dataloader_hdf5.py",
    "dataloader_lmdb.py>data_tools/dataloader_lmdb.py",
    "eval_dataloader_hdf5.py>data_tools/eval_dataloader_hdf5.py",
    "test_dataloader_hdf5.py>data_tools/test_dataloader_hdf5.py"
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