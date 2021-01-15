#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: test_script_template.py
# Created Date: Friday December 25th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Friday, 15th January 2021 10:30:27 am
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################


import os
import time
import datetime
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

# from utilities.Reporter import Reporter
from tqdm import tqdm

class Tester(object):
    def __init__(self, config, reporter):
        
        self.config     = config
        # logger
        self.reporter   = reporter

        #============build evaluation dataloader==============#
        print("Prepare the test dataloader...")
        dlModulename    = config["testDataloader"]
        package         = __import__("data_tools.test_dataloader_%s"%dlModulename, fromlist=True)
        dataloaderClass = getattr(package, 'TestDataset')
        dataloader      = dataloaderClass(config["testDataRoot"],
                                        config["batchSize"])
        self.test_loader= dataloader

        self.test_iter  = len(dataloader)//config["batchSize"]
        if len(dataloader)%config["batchSize"]>0:
            self.test_iter+=1
        
    
    def __init_framework__(self):
        '''
            This function is designed to define the framework,
            and print the framework information into the log file
        '''
        #===============build models================#
        print("build models...")
        # TODO [import models here]
        from components.RRDBNet import RRDBNet_Implicit

        # print and recorde model structure
        self.reporter.writeInfo("Model structure:")

        # TODO replace below lines to define the model framework
        self.network = RRDBNet_Implicit(5,1)
        
        # train in GPU
        if self.config["cuda"] >=0:
            self.network = self.network.cuda()
            
        self.network.load_state_dict(torch.load(self.config["ckp_name"]))
        print('loaded trained backbone model epoch {}...!'.format(self.config["checkpointStep"]))

    def test(self):
        
        save_result = self.config["saveTestResult"]
        save_dir    = self.config["testSamples"]
                            
        # models
        self.__init_framework__()
        
        # Start time
        import datetime
        print("Start to test at %s"%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        print('Start ===================================  test...')
        start_time = time.time()
        self.network.eval()
        with torch.no_grad():
            for iii in tqdm(range(self.test_iter)):
                lr,hr = self.test_loader()
                if self.config["cuda"] >=0:
                    hr = hr.cuda()
                    lr = lr.cuda()
                res = self.network(lr)
                print("Save test results......")
                # tensordata = res[0,:,:,:].data.cpu().numpy()
                # np.savez(os.path.join(save_dir, '{}_step{}_v_{}.npz'.format(img_name,self.config["checkpointStep"],self.config["version"])),tensordata)
                dataset_size = res.shape[0]
                res = res.cpu().numpy()
                hr = hr.cpu().numpy()
                for t in range(dataset_size):
                    sns.heatmap(res[t,0,:,:],vmin=0,vmax=res[t,0,:,:].max(),cbar=False)
                    plt.savefig(os.path.join(save_dir,'{}_epoch{}_v_{}_batch_{}.png'.format(iii,
                        self.config["checkpointStep"],self.config["version"], t)), dpi=400)
                    
                    sns.heatmap(hr[t,0,:,:],vmin=0,vmax=hr[t,0,:,:].max(),cbar=False)
                    plt.savefig(os.path.join(save_dir,'{}_epoch{}_v_{}_batch_{}_GT.png'.format(iii,
                        self.config["checkpointStep"],self.config["version"], t)), dpi=400)
                                
        elapsed = time.time() - start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed [{}]".format(elapsed))