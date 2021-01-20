#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: main.py
# Created Date: Tuesday April 28th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Wednesday, 20th January 2021 2:09:59 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################


import  os
import  json
import  shutil
import  argparse
import  platform
from    torch.backends import cudnn
from    utilities.json_config import readConfig, writeConfig
from    utilities.reporter import Reporter
from    utilities.yaml_config import getConfigYaml
from    utilities.sshupload import fileUploaderClass


def str2bool(v):
    return v.lower() in ('true')

####################################################################################
# To configure the seting of training\finetune\test
#
####################################################################################
def getParameters():
    parser = argparse.ArgumentParser()
    
    # general settings
    parser.add_argument('--version', type=str, default='version_name', help="version name for train, test, finetune")
    parser.add_argument('--phase', type=str, default="train", 
                            choices=['train', 'finetune','test','debug'], help="The phase of current project")
    parser.add_argument('--cuda', type=int, default=0) # >0 if it is set as -1, program will use CPU
    parser.add_argument('--dataloader_workers', type=int, default=1)
    parser.add_argument('--checkpoint_epoch', type=int, default=65, help="checkpoint epoch for test phase or finetune phase")
    
    # training
    parser.add_argument('--experiment_description', type=str, default="加上perceptual loss")
    parser.add_argument('--train_yaml', type=str, default="train_rrdbnet.yaml")

    # test
    parser.add_argument('--test_script_name', type=str, default='rrdbnet')
    parser.add_argument('--test_batch_size', type=int, default=20)
    parser.add_argument('--node_name', type=str, default='localhost', 
                            choices=['localhost', '4card','8card','new4card'])
    
    parser.add_argument('--save_test_result', type=str2bool, default='True')

    parser.add_argument('--test_dataset_path', type=str, default='E:\\RainNet_Dataset\\RainNet_Evaluation.hdf5')
    parser.add_argument('--test_dataloader', type=str, default='hdf5')
    
    parser.add_argument('--use_specified_data', type=str2bool, default='False')
    parser.add_argument('--specified_data_paths', type=str, nargs='+', default=[""], help='paths to specified files')
    
    # logs (does not to be changed in most time)
    parser.add_argument('--use_tensorboard', type=str2bool, default='True',
                            choices=['True', 'False'], help='enable the tensorboard')
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--sample_step', type=int, default=100)
    
    # # template (onece editing finished, it should be deleted)
    # parser.add_argument('--str_parameter', type=str, default="default", help='str parameter')
    # parser.add_argument('--str_parameter_choices', type=str, default="default", choices=['choice1', 'choice2','choice3'], help='str parameter with choices list')
    # parser.add_argument('--int_parameter', type=int, default=0, help='int parameter')
    # parser.add_argument('--float_parameter', type=float, default=0.0, help='float parameter')
    # parser.add_argument('--bool_parameter', type=str2bool, default='True', choices=['True', 'False'], help='bool parameter')
    # parser.add_argument('--list_str_parameter', type=str, nargs='+', default=["element1","element2"], help='str list parameter')
    # parser.add_argument('--list_int_parameter', type=int, nargs='+', default=[0,1], help='int list parameter')
    return parser.parse_args()

####################################################################################
# This function will create the related directories before the 
# training\fintune\test starts
# Your_log_root (version name)
#   |---summary/...
#   |---samples/... (save evaluated images)
#   |---checkpoints/...
#   |---scripts/...
#
####################################################################################
def createDirs(sys_state):
    # the base dir
    if not os.path.exists(sys_state["log_root_path"]):
        os.makedirs(sys_state["log_root_path"])

    # create dirs
    sys_state["project_root"]        = os.path.join(sys_state["log_root_path"], sys_state["version"])
    project_root                     = sys_state["project_root"]
    if not os.path.exists(project_root):
        os.makedirs(project_root)
    
    sys_state["project_summary"]     = os.path.join(project_root, "summary")
    if not os.path.exists(sys_state["project_summary"]):
        os.makedirs(sys_state["project_summary"])

    sys_state["project_checkpoints"] = os.path.join(project_root, "checkpoints")
    if not os.path.exists(sys_state["project_checkpoints"]):
        os.makedirs(sys_state["project_checkpoints"])

    sys_state["project_samples"]     = os.path.join(project_root, "samples")
    if not os.path.exists(sys_state["project_samples"]):
        os.makedirs(sys_state["project_samples"])

    sys_state["project_scripts"]     = os.path.join(project_root, "scripts")
    if not os.path.exists(sys_state["project_scripts"]):
        os.makedirs(sys_state["project_scripts"])
    
    sys_state["reporter_path"] = os.path.join(project_root,sys_state["version"]+"_report")

def main():
    config = getParameters()
    # speed up the program
    cudnn.benchmark = True

    ignoreKey = [
        "dataloader_workers","log_root_path",
        "project_root","project_summary","project_checkpoints",
        "project_samples","project_scripts","reporter_path",
        "use_specified_data","dataset_path", "cuda"
    ]

    sys_state = {}
    # set the thread number of data loading task
    sys_state["dataloader_workers"] = config.dataloader_workers

    # set the GPU number
    if config.cuda >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda)

    # read system environment paths
    env_config = readConfig('env/env.json')
    env_config = env_config["path"]

    sys_state["cuda"] = config.cuda

    # obtain all configurations in argparse
    config_dic = vars(config)
    for config_key in config_dic.keys():
        sys_state[config_key] = config_dic[config_key]
    
    #=======================Train Phase=========================#
    if config.phase == "train":
        # read training configurations from yaml file
        ymal_config = getConfigYaml(os.path.join(env_config["train_config_path"], config.train_yaml))
        for item in ymal_config.items():
            sys_state[item[0]] = item[1]

        # create related dirs
        sys_state["log_root_path"] = env_config["train_log_root"]
        createDirs(sys_state)
        
        # create reporter file
        reporter = Reporter(sys_state["reporter_path"])

        # save the config json
        config_json = os.path.join(sys_state["project_root"], env_config["config_json_name"])
        writeConfig(config_json, sys_state)

        # save the dependent scripts 
        # TODO and copy the scripts to the project dir
        
        # save the trainer script into [train_logs_root]\[version name]\scripts\
        file1       = os.path.join(env_config["trainScriptsPath"], "trainer_%s.py"%sys_state["trainScriptName"])
        tgtfile1    = os.path.join(sys_state["project_scripts"], "trainer_%s.py"%sys_state["trainScriptName"])
        shutil.copyfile(file1,tgtfile1)

        # TODO replace below lines, here to save the critical scripts

    #=====================Finetune Phase=====================#
    elif config.phase == "finetune":
        sys_state["log_root_path"]    = env_config["train_log_root"]
        sys_state["project_root"]    = os.path.join(sys_state["log_root_path"], sys_state["version"])

        config_json                 = os.path.join(sys_state["project_root"], env_config["config_json_name"])
        train_config                = readConfig(config_json)
        for item in train_config.items():
            if item[0] in ignoreKey:
                pass
            else:
                sys_state[item[0]]  = item[1]
        
        createDirs(sys_state)
        reporter = Reporter(sys_state["reporter_path"])
        sys_state["com_base"]       = "train_logs.%s.scripts."%sys_state["version"]
    
    #=======================Test Phase=========================#
    elif config.phase == "test":

        # TODO modify below lines to obtain the configuration
        sys_state["log_root_path"]        = env_config["train_log_root"]
        
        sys_state["test_samples_path"]        = os.path.join(env_config["test_log_root"], 
                                            sys_state["version"] , "samples")

        if not os.path.exists(sys_state["test_samples_path"]):
            os.makedirs(sys_state["test_samples_path"])
        
        # Create dirs
        createDirs(sys_state)
        config_json = os.path.join(sys_state["project_root"], env_config["config_json_name"])
        
        # Read model_config.json from remote machine
        if sys_state["node_name"]!="localhost":
            print("ready to fetch the %s from the server!"%config_json)
            nodeinf     = readConfig(env_config["remoteNodeInfo"])
            nodeinf     = nodeinf[sys_state["node_name"]]
            uploader    = fileUploaderClass(nodeinf["ip"],nodeinf["user"],nodeinf["passwd"])
            
            if config.train_logs_root=="":
                remotebase  = os.path.join(nodeinf['basePath'],"train_logs",sys_state["version"]).replace('\\','/')
            else:
                remotebase  = os.path.join(config.train_logs_root).replace('\\','/')
            # Get the config.json
            print("ready to get the config.json...")
            remoteFile  = os.path.join(remotebase, env_config["config_json_name"]).replace('\\','/')
            localFile   = config_json
            
            ssh_state = uploader.sshScpGet(remoteFile,localFile)
            if not ssh_state:
                raise Exception(print("Get file %s failed! Program exists!"%remoteFile))
            print("success get the config file from server %s"%nodeinf['ip'])

        # Read model_config.json
        json_obj    = readConfig(config_json)
        for item in json_obj.items():
            if item[0] in ignoreKey:
                pass
            else:
                sys_state[item[0]] = item[1]
            
        # Read scripts from remote machine
        if sys_state["node_name"]!="localhost":
            # Get scripts
            remoteFile  = os.path.join(remotebase, "scripts", sys_state["gScriptName"]+".py").replace('\\','/')
            localFile   = os.path.join(sys_state["project_scripts"], sys_state["gScriptName"]+".py")
            ssh_state = uploader.sshScpGet(remoteFile, localFile)
            if not ssh_state:
                raise Exception(print("Get file %s failed! Program exists!"%remoteFile))
            print("Get the scripts:%s.py successfully"%sys_state["gScriptName"])
            # Get checkpoint of generator
            localFile   = os.path.join(sys_state["project_checkpoints"], "%d_Generator.pth"%sys_state["checkpoint_epoch"])
            if not os.path.exists(localFile):
                remoteFile  = os.path.join(remotebase, "checkpoints", "%d_Generator.pth"%sys_state["checkpoint_epoch"]).replace('\\','/')
                ssh_state = uploader.sshScpGet(remoteFile, localFile, True)
                if not ssh_state:
                    raise Exception(print("Get file %s failed! Program exists!"%remoteFile))
                print("Get the %s file successfully"%("%d_Generator.pth"%sys_state["checkpoint_epoch"]))
            else:
                print("%s file exists"%("%d_Generator.pth"%sys_state["checkpoint_epoch"]))

        # TODO get the checkpoint file path
        sys_state["ckp_name"]       = os.path.join(sys_state["project_checkpoints"],
                                        "epoch%d_%s.pth"%(sys_state["checkpoint_epoch"],
                                            sys_state["check_point_names"]["generator_name"]))

        # Get the test configurations
        sys_state["com_base"]       = "train_logs.%s.scripts."%sys_state["version"]

        # make a reporter
        reporter = Reporter(sys_state["reporter_path"])
        
        # Display the test information
        # TODO modify below lines to display your configuration information
        moduleName  = "test_scripts.tester_" + sys_state["test_script_name"]
        print("Start to run test script: {}".format(moduleName))
        print("Test version: %s"%sys_state["version"])
        print("Test Script Name: %s"%sys_state["test_script_name"])

        package     = __import__(moduleName, fromlist=True)
        testerClass = getattr(package, 'Tester')
        tester      = testerClass(sys_state,reporter)
        tester.test()
    
    if config.phase == "train" or config.phase == "finetune":
        
        # get the dataset path
        for data_key in env_config["dataset_paths"].keys():
            sys_state["dataset_paths"][data_key] = env_config["dataset_paths"][data_key]

        # display the training information
        moduleName  = "train_scripts.trainer_" + sys_state["train_script_name"]
        if config.phase == "finetune":
            moduleName  = sys_state["com_base"] + "trainer_" + sys_state["train_script_name"]
        
        # print some important information
        # TODO
        print("Start to run training script: {}".format(moduleName))
        print("Traning version: %s"%sys_state["version"])
        print("Dataloader Name: %s"%sys_state["dataloader"])
        # print("Image Size: %d"%sys_state["imsize"])
        print("Batch size %d"%(sys_state["batch_size"]))
        
        # Load the training script and start to train
        reporter.writeConfig(sys_state)

        package     = __import__(moduleName, fromlist=True)
        trainerClass= getattr(package, 'Trainer')
        trainer     = trainerClass(sys_state, reporter)
        trainer.train()


if __name__ == '__main__':
    main()