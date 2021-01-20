#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: trainer_script_template.py
# Created Date: Friday December 25th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Wednesday, 20th January 2021 2:16:45 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################


import os
import time
import torch
import torch.nn as nn
from utilities.save_heatmap import SaveHeatmap

# modify this template to derive your train class

class Trainer(object):
    def __init__(self, config, reporter):

        self.config     = config
        # logger
        self.reporter   = reporter
        
        # Data loader
        #============build train dataloader==============#
        # TODO to modify the key: "your_train_dataset" to get your train dataset path
        train_dataset = config["dataset_paths"]["your_train_dataset"]

        #================================================#
        print("Prepare the train dataloader...")
        dlModulename    = config["dataloader"]
        package         = __import__("data_tools.dataloader_%s"%dlModulename, fromlist=True)
        dataloaderClass = getattr(package, 'GetLoader')
        dataloader      = dataloaderClass(train_dataset,
                                        config["batchSize"],
                                        config["randomSeed"])
        self.train_loader= dataloader

        #========build evaluation dataloader=============#
        # TODO to modify the key: "your_eval_dataset" to get your evaluation dataset path
        eval_dataset = config["dataset_paths"]["your_eval_dataset"]

        #================================================#
        print("Prepare the evaluation dataloader...")
        dlModulename    = config["evalDataloader"]
        package         = __import__("data_tools.eval_dataloader_%s"%dlModulename, fromlist=True)
        dataloaderClass = getattr(package, 'EvalDataset')
        dataloader      = dataloaderClass(eval_dataset,config["evalBatchSize"])
        self.eval_loader= dataloader

        self.eval_iter  = len(dataloader)//config["evalBatchSize"]
        if len(dataloader)%config["evalBatchSize"]>0:
            self.eval_iter+=1

        #==============build tensorboard=================#
        if self.config["useTensorboard"]:
            from utilities.utilities import build_tensorboard
            self.tensorboard_writer = build_tensorboard(self.config["projectSummary"])


    # TODO modify this function to build your models
    def __init_framework__(self):
        '''
            This function is designed to define the framework,
            and print the framework information into the log file
        '''
        #===============build models================#
        print("build models...")
        # TODO [import models here]
        from components.RRDBNet import RRDBNet

        # print and recorde model structure
        self.reporter.writeInfo("Model structure:")

        # TODO replace below lines to define the model framework
        self.network = RRDBNet(5,1)
        self.reporter.writeModel(self.network.__str__())
        
        # train in GPU
        if self.config["cuda"] >=0:
            self.network = self.network.cuda()

        # if in finetune phase, load the pretrained checkpoint
        if self.config["phase"] == "finetune":
            model_path = os.path.join(self.config["projectCheckpoints"],
                                        "epoch%d_%s.pth"%(self.config["checkpointStep"],
                                        self.config["checkPointNames"]["GeneratorName"]))
            self.network.load_state_dict(torch.load(model_path))
            print('loaded trained backbone model epoch {}...!'.format(self.config["projectCheckpoints"]))
    

    # TODO modify this function to evaluate your model
    def __evaluation__(self, epoch, step = 0):
        # Evaluate the checkpoint
        self.network.eval()
        with torch.no_grad():
            lr,hr = self.eval_loader()
            if self.config["cuda"] >=0:
                hr = hr.cuda()
                lr = lr.cuda()
            res = self.network(lr)
            print("Save test results......")
            res = res.cpu().numpy()
            hr = hr.cpu().numpy()
            generated_path = os.path.join(self.config["projectSamples"],
                                'epoch{}_v_{}_batch.png'.format(epoch,
                                        self.config["version"]))
            hr_path        = os.path.join(self.config["projectSamples"],
                                'epoch{}_v_{}_batch_GT.png'.format(epoch,
                                        self.config["version"]))
            SaveHeatmap(res,generated_path,2)
            SaveHeatmap(hr,hr_path,2)

    def train(self):
        
        # general configurations 
        ckpt_dir    = self.config["projectCheckpoints"]
        log_frep    = self.config["logStep"]
        model_freq  = self.config["modelSaveEpoch"]
        lr_base     = self.config["lr"]
        total_epoch = self.config["totalEpoch"]
        beta1       = self.config["beta1"]
        beta2       = self.config["beta2"]
        l1_W        = self.config["l1Weight"]
        # lrDecayStep = self.config["lrDecayStep"]
        # TODO [more configurations here]

        #===============build framework================#
        self.__init_framework__()

        # set the start point for training loop
        if self.config["phase"] == "finetune":
            start = self.config["checkpointStep"] - 1
        else:
            start = 0
        

        #===============build optimizer================#
        print("build the optimizer...")
        # Optimizer
        # TODO replace below lines to build your optimizer
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, 
                                    self.network.parameters()), lr_base, [beta1, beta2])

        #===============build losses===================#
        # TODO replace below lines to build your losses
        l1 = nn.L1Loss() # [replace this]

        from losses.PerceptualLoss import PerceptualLoss
        perceptual_config = self.config["perceptual"]
        ploss = PerceptualLoss(
                        perceptual_config["layer_weights"],
                        perceptual_config["vgg_type"],
                        perceptual_config["use_input_norm"],
                        perceptual_config["perceptual_weight"],
                        perceptual_config["criterion"]
                    )
        if self.config["cuda"] >=0:
            ploss = ploss.cuda()
        
        # Caculate the epoch number
        step_epoch  = len(self.train_loader)
        print("Total step = %d in each epoch"%step_epoch)

        # Start time
        import datetime
        print("Start to train at %s"%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        print('Start   ===========================  training...')
        start_time = time.time()
        for epoch in range(start, total_epoch):
            for step in range(step_epoch):
                # Set the networks to train mode
                self.network.train()
                # TODO [add more code here]
                
                # TODO read the training data
                lr, hr  = self.train_loader.next()
                
                # TODO [inference code here]
                generated_hr = self.network(lr)
                
                # TODO [caculate losses]
                # l1 loss
                loss_l1 = l1(generated_hr,hr)
                
                # perceptual loss
                loss_per = ploss(generated_hr,hr)
                # loss_per= perceptual_loss(generated_hr.repeat_interleave(3,1),hr.repeat_interleave(3,1))
                
                # total loss
                loss_curr= l1_W * loss_l1 + loss_per
                # loss_curr = loss_l1
                
                # clear cumulative gradient
                optimizer.zero_grad()
                # caculate gradients
                loss_curr.backward()
                # update weights
                optimizer.step()
                
                # Print out log info
                if (step + 1) % log_frep == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))

                    # cumulative steps
                    cum_step = (step_epoch * epoch + step + 1)
                    
                    #==================Print log info======================#
                    print("[{}], Elapsed [{}], Epoch [{}/{}], Step [{}/{}], loss: {:.4f}, l1: {:.4f}, perceptual: {:.4f}".
                        format(self.config["version"], elapsed, epoch + 1, total_epoch, step + 1, step_epoch, 
                                loss_curr.item(),loss_l1.item(), loss_per.item()))
                    
                    #===================Write log info into log file=======#
                    self.reporter.writeTrainLog(epoch+1,step+1,
                                "loss: {:.4f}, l1: {:.4f}, perceptual: {:.4f}".format(loss_curr.item(),
                                                                        loss_l1.item(), loss_per.item() ))

                    #==================Tensorboard=========================#
                    # write training information into tensorboard log files
                    if self.config["useTensorboard"]:

                        # TODO replace  below lines to record the losses or metrics 
                        self.tensorboard_writer.add_scalar('data/loss', loss_curr.item(), cum_step)
            
            #===============adjust learning rate============#
            if (epoch + 1) in self.config["lrDecayStep"]:
                print("Learning rate decay")
                for p in optimizer.param_groups:
                    p['lr'] *= self.config["lrDecay"]
                    print("Current learning rate is %f"%p['lr'])

            #===============save checkpoints================#
            if (epoch+1) % model_freq==0:
                print("Save epoch %d model checkpoint!"%(epoch+1))
                torch.save(self.network.state_dict(),
                        os.path.join(ckpt_dir, 'epoch{}_{}.pth'.format(epoch + 1, 
                                    self.config["checkPointNames"]["GeneratorName"])))
                del hr
                del lr
                del generated_hr
                
                self.__evaluation__(epoch+1)