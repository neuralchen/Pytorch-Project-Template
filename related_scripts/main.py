#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: main.py
# Created Date: Tuesday April 28th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Monday, 27th July 2020 11:32:18 pm
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
    parser.add_argument('--mode', type=str, default="test", choices=['train', 'finetune','test','debug'])
    parser.add_argument('--cuda', type=int, default=-1) # >0 if it is set as -1, program will use CPU
    parser.add_argument('--dataloader_workers', type=int, default=1)
    parser.add_argument('--checkpoint', type=int, default=0, help="checkpoint step for test mode or finetune mode")
    
    # training
    parser.add_argument('--version', type=str, default='version', help="version name for train, test, finetune")
    parser.add_argument('--experiment_description', type=str, default="description")
    parser.add_argument('--train_yaml', type=str, default="train.yaml")

    # test
    parser.add_argument('--test_script_name', type=str, default='tester script name')
    
    # logs (does not to be changed in most time)
    parser.add_argument('--use_tensorboard', type=str2bool, default='True', choices=['True', 'False'], help='enable the tensorboard')
    parser.add_argument('--log_step', type=int, default=0)
    parser.add_argument('--sample_step', type=int, default=0)
    parser.add_argument('--model_save_step', type=int, default=0)
    parser.add_argument('--train_logs_root', type=str, default="")
    

    # template (onece editing finished, it should be deleted)
    parser.add_argument('--str_parameter', type=str, default="default", help='str parameter')
    parser.add_argument('--str_parameter_choices', type=str, default="default", choices=['choice1', 'choice2','choice3'], help='str parameter with choices list')
    parser.add_argument('--int_parameter', type=int, default=0, help='int parameter')
    parser.add_argument('--float_parameter', type=float, default=0.0, help='float parameter')
    parser.add_argument('--bool_parameter', type=str2bool, default='True', choices=['True', 'False'], help='bool parameter')
    parser.add_argument('--list_str_parameter', type=str, nargs='+', default=["element1","element2"], help='str list parameter')
    parser.add_argument('--list_int_parameter', type=int, nargs='+', default=[0,1], help='int list parameter')
    return parser.parse_args()

####################################################################################
# This function will create the related directories before the 
# training\fintune\test starts
# Your_log_root (version name)
#   |---summary/...
#   |---samples/...
#   |---checkpoints/...
#   |---scripts/...
#
####################################################################################
def createDirs(sys_state):
    # the base dir
    if not os.path.exists(sys_state["logRootPath"]):
        os.makedirs(sys_state["logRootPath"])

    # create dirs
    sys_state["projectRoot"]        = os.path.join(sys_state["logRootPath"], sys_state["version"])
    projectRoot                     = sys_state["projectRoot"]
    if not os.path.exists(projectRoot):
        os.makedirs(projectRoot)
    
    sys_state["projectSummary"]     = os.path.join(projectRoot, "summary")
    if not os.path.exists(sys_state["projectSummary"]):
        os.makedirs(sys_state["projectSummary"])

    sys_state["projectCheckpoints"] = os.path.join(projectRoot, "checkpoints")
    if not os.path.exists(sys_state["projectCheckpoints"]):
        os.makedirs(sys_state["projectCheckpoints"])

    sys_state["projectSamples"]     = os.path.join(projectRoot, "samples")
    if not os.path.exists(sys_state["projectSamples"]):
        os.makedirs(sys_state["projectSamples"])

    sys_state["projectScripts"]     = os.path.join(projectRoot, "scripts")
    if not os.path.exists(sys_state["projectScripts"]):
        os.makedirs(sys_state["projectScripts"])
    
    sys_state["reporterPath"] = os.path.join(projectRoot,sys_state["version"]+"_report")

def main():
    config = getParameters()
    # speed up the program
    cudnn.benchmark = True

    ignoreKey = [
        "dataloader_workers","logRootPath",
        "projectRoot","projectSummary","projectCheckpoints",
        "projectSamples","projectScripts","reporterPath",
        "useSpecifiedImg","dataset_path", "cuda"
    ]

    sys_state = {}
    # set the thread number of data loading task
    sys_state["dataloader_workers"] = config.dataloader_workers

    # set the GPU number
    if config.cuda >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda)

    # read system environment paths
    env_config = readConfig('env/config.json')
    env_config = env_config["path"]

    sys_state["cuda"]   = config.cuda
    
    # Train mode
    if config.mode == "train":
        
        sys_state["version"]                = config.version
        sys_state["experiment_description"] = config.experiment_description
        sys_state["mode"]                   = config.mode

        # read training configurations
        ymal_config = getConfigYaml(os.path.join(env_config["trainConfigPath"], config.train_yaml))
        for item in ymal_config.items():
            sys_state[item[0]] = item[1]

        # create related dirs
        sys_state["logRootPath"] = config.train_logs_root
        createDirs(sys_state)
        
        # create reporter file
        reporter = Reporter(sys_state["reporterPath"])

        # save the config json
        config_json = os.path.join(sys_state["projectRoot"], env_config["configJsonName"])
        writeConfig(config_json, sys_state)

        # save the dependent scripts 
        # and copy the scripts to the project dir
        
        file1       = os.path.join(env_config["trainScriptsPath"], "trainer_%s.py"%sys_state["trainScriptName"])
        tgtfile1    = os.path.join(sys_state["projectScripts"], "trainer_%s.py"%sys_state["trainScriptName"])
        shutil.copyfile(file1,tgtfile1)

        file2       = os.path.join("./components", "%s.py"%sys_state["gScriptName"])
        tgtfile2    = os.path.join(sys_state["projectScripts"], "%s.py"%sys_state["gScriptName"])
        shutil.copyfile(file2,tgtfile2)

        file3       = os.path.join("./components", "%s.py"%sys_state["dScriptName"])
        tgtfile3    = os.path.join(sys_state["projectScripts"], "%s.py"%sys_state["dScriptName"])
        shutil.copyfile(file3,tgtfile3)

    elif config.mode == "finetune":
        sys_state["logRootPath"]    = env_config["trainLogRoot"]
        sys_state["version"]        = config.version
        sys_state["projectRoot"]    = os.path.join(sys_state["logRootPath"], sys_state["version"])
        sys_state["checkpointStep"] = config.checkpoint

        config_json                 = os.path.join(sys_state["projectRoot"], env_config["configJsonName"])
        train_config                = readConfig(config_json)
        for item in train_config.items():
            if item[0] in ignoreKey:
                pass
            else:
                sys_state[item[0]] = item[1]
        
        sys_state["mode"]           = config.mode
        createDirs(sys_state)
        reporter = Reporter(sys_state["reporterPath"])
        sys_state["com_base"]       = "train_logs.%s.scripts."%sys_state["version"]
        
    elif config.mode == "test":
        sys_state["version"]        = config.version
        sys_state["logRootPath"]    = env_config["trainLogRoot"]
        sys_state["nodeName"]       = config.nodeName
        sys_state["totalImg"]       = config.totalImg
        sys_state["useSpecifiedImg"]= config.useSpecifiedImg
        sys_state["checkpointStep"] = config.checkpoint
        sys_state["testImgRoot"]    = config.testImgRoot

        sys_state["testSamples"]    = os.path.join(env_config["testLogRoot"], sys_state["version"] , "samples")
        if not os.path.exists(sys_state["testSamples"]):
            os.makedirs(sys_state["testSamples"])
        
        if config.useSpecifiedImg:
            sys_state["useSpecifiedImg"]   = config.useSpecifiedImg
        # Create dirs
        createDirs(sys_state)
        config_json = os.path.join(sys_state["projectRoot"], env_config["configJsonName"])
        
        # Read model_config.json from remote machine
        if sys_state["nodeName"]!="localhost":
            print("ready to fetch the %s from the server!"%config_json)
            nodeinf     = readConfig(env_config["remoteNodeInfo"])
            nodeinf     = nodeinf[sys_state["nodeName"]]
            uploader    = fileUploaderClass(nodeinf["ip"],nodeinf["user"],nodeinf["passwd"])
            if config.train_logs_root=="":
                remotebase  = os.path.join(nodeinf['basePath'],"train_logs",sys_state["version"]).replace('\\','/')
            else:
                remotebase  = os.path.join(config.train_logs_root).replace('\\','/')
            # Get the config.json
            print("ready to get the config.json...")
            remoteFile  = os.path.join(remotebase, env_config["configJsonName"]).replace('\\','/')
            localFile   = config_json
            
            state = uploader.sshScpGet(remoteFile,localFile)
            if not state:
                raise Exception(print("Get file %s failed! Program exists!"%remoteFile))
            print("success get the config file from server %s"%nodeinf['ip'])

        # Read model_config.json
        json_obj    = readConfig(config_json)
        for item in json_obj.items():
            if item[0] in ignoreKey:
                pass
            else:
                sys_state[item[0]] = item[1]
        
        # get the dataset path
        sys_state["content"]= env_config["datasetPath"]["Place365_big"]
        sys_state["style"]  = env_config["datasetPath"]["WikiArt"]
            
        # Read scripts from remote machine
        if sys_state["nodeName"]!="localhost":
            # Get scripts
            remoteFile  = os.path.join(remotebase, "scripts", sys_state["gScriptName"]+".py").replace('\\','/')
            localFile   = os.path.join(sys_state["projectScripts"], sys_state["gScriptName"]+".py")
            state = uploader.sshScpGet(remoteFile, localFile)
            if not state:
                raise Exception(print("Get file %s failed! Program exists!"%remoteFile))
            print("Get the scripts:%s.py successfully"%sys_state["gScriptName"])
            # Get checkpoint of generator
            localFile   = os.path.join(sys_state["projectCheckpoints"], "%d_Generator.pth"%sys_state["checkpointStep"])
            if not os.path.exists(localFile):
                remoteFile  = os.path.join(remotebase, "checkpoints", "%d_Generator.pth"%sys_state["checkpointStep"]).replace('\\','/')
                state = uploader.sshScpGet(remoteFile, localFile, True)
                if not state:
                    raise Exception(print("Get file %s failed! Program exists!"%remoteFile))
                print("Get the %s file successfully"%("%d_Generator.pth"%sys_state["checkpointStep"]))
            else:
                print("%s file exists"%("%d_Generator.pth"%sys_state["checkpointStep"]))
        sys_state["ckp_name"]       = os.path.join(sys_state["projectCheckpoints"], "%d_Generator.pth"%sys_state["checkpointStep"])    
        # Get the test configurations
        sys_state["testScriptName"] = config.testScriptName
        sys_state["batchSize"]      = config.testBatchSize
        sys_state["totalImg"]       = config.totalImg
        sys_state["saveTestImg"]    = config.saveTestImg
        sys_state["com_base"]       = "train_logs.%s.scripts."%sys_state["version"]
        reporter = Reporter(sys_state["reporterPath"])
        
        # Display the test information
        moduleName  = "test_scripts.tester_" + sys_state["testScriptName"]
        print("Start to run test script: {}".format(moduleName))
        print("Test version: %s"%sys_state["version"])
        print("Test Script Name: %s"%sys_state["testScriptName"])
        print("Generator Script Name: %s"%sys_state["gScriptName"])
        # print("Discriminator Script Name: %s"%sys_state["stuScriptName"])
        print("Image Crop Size: %d"%sys_state["imCropSize"])
        package     = __import__(moduleName, fromlist=True)
        testerClass = getattr(package, 'Tester')
        tester      = testerClass(sys_state,reporter)
        tester.test()
    
    if config.mode == "train" or config.mode == "finetune":
        
        # get the dataset path
        sys_state["content"]= env_config["datasetPath"]["Place365_big"]
        sys_state["style"]  = env_config["datasetPath"]["WikiArt"]

        # display the training information
        moduleName  = "train_scripts.trainer_" + sys_state["trainScriptName"]
        if config.mode == "finetune":
            moduleName  = sys_state["com_base"] + "trainer_" + sys_state["trainScriptName"]
        
        # print some important information
        print("Start to run training script: {}".format(moduleName))
        print("Traning version: %s"%sys_state["version"])
        print("Training Script Name: %s"%sys_state["trainScriptName"])
        print("Generator Script Name: %s"%sys_state["gScriptName"])
        print("Discriminator Script Name: %s"%sys_state["dScriptName"])
        # print("Image Size: %d"%sys_state["imsize"])
        print("Image Crop Size: %d"%sys_state["imCropSize"])
        print("D : G = %d : %d"%(sys_state["dStep"],sys_state["gStep"]))
        print("Batch size %d"%(sys_state["batchSize"]))
        print("Resblock number %d"%(sys_state["resNum"]))
        
        # Load the training script and start to train
        reporter.writeConfig(sys_state)

        print("prepare the dataloader...")
        total_loader  = getLoader(sys_state["style"], sys_state["content"],
                            sys_state["selectedStyleDir"],sys_state["selectedContentDir"],
                            sys_state["imCropSize"], sys_state["batchSize"],sys_state["dataloader_workers"])

        package     = __import__(moduleName, fromlist=True)
        trainerClass= getattr(package, 'Trainer')
        trainer     = trainerClass(sys_state, total_loader,reporter)
        trainer.train()


if __name__ == '__main__':
    main()