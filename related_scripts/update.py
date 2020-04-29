#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: update_remote_project.py
# Created Date: Wednesday February 26th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Monday, 20th April 2020 12:57:02 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################
import os
from pathlib import Path
import json
from utilities.sshupload import fileUploaderClass

ssh_ip = "192.168.101.57"
ssh_username = "gdp"
ssh_passwd = "glass123456"
ssh_port = 22
root_path = "/home/gdp/CXH/StyleTransfer"

scan_config={
    "white_list":["py","yaml"],
    # "ignore_dir":["test_logs","train_logs",".vscode"],
    # "ignore_file":[".gitignore","LICENSE","README.md","*.jpg","*.png","*.JPG","*.JPEG","*.PNG","*.pyc"],
}

if __name__ == "__main__":

    path = "./file_sync/filestate.json"
    files_dict = {}
    last_state = {}
    changed_files = []
    if not Path("./file_sync/").exists():
        Path("./file_sync/").mkdir(parents=True)
    else:
        if Path(path).exists():
            with open(path,'r') as cf:
                nodelocaltionstr = cf.read()
                last_state = json.loads(nodelocaltionstr)
    all_files = []
    # scan files
    for item in scan_config["white_list"]:
        files = Path('.').glob('*.%s'%item) # ./*
        for one_file in files:
            all_files.append(one_file)
        files = Path('.').glob('*/*.%s'%item) # ./*/*
        for one_file in files:
            all_files.append(one_file)

    # check updated files
    for item in all_files:
        temp = item.stat().st_mtime
        if item._str in last_state:
            last_mtime = last_state[item._str]
            if last_mtime != temp:
                changed_files.append(item._str)
                last_state[item._str] = temp
        else:
            changed_files.append(item._str)
            last_state[item._str] = temp
    
    with open(path, 'w') as cf:
        configjson  = json.dumps(last_state, indent=4)
        cf.writelines(configjson)
    
    print(changed_files)
    

    remotemachine = fileUploaderClass(ssh_ip,ssh_username,ssh_passwd,ssh_port)
    for item in changed_files:
        localfile = item
        print("here %s"%item)
        remotefile = Path(root_path,item).as_posix()
        remotemachine.sshScpPut(localfile,remotefile)