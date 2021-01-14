#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: trainer_script_template.py
# Created Date: Friday December 25th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 12th January 2021 10:25:48 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################


import os
import time

import torch
import torch.nn as nn
from tqdm import tqdm

# modify this template to derive your train class

class Trainer(object):
    def __init__(self, config, reporter):

        self.config     = config
        # logger
        self.reporter   = reporter
        
        # Data loader
        print("Prepare the dataloader...")
        dlModulename    = config["dataloader"]
        package         = __import__(dlModulename, fromlist=True)
        dataloaderClass = getattr(package, 'GetLoader')
        
        # TODO replace below lines to config your dataloader
        dataloader      = dataloaderClass(config["dataset_path"],
                                        config["batchSize"],
                                        config["randomSeed"])
        self.train_loader= dataloader
        
        # self.eval_loader    = dataloaders_list[1]

    def train(self):
        
        # general configurations 
        ckpt_dir    = self.config["projectCheckpoints"]
        log_frep    = self.config["logStep"]
        model_freq  = self.config["modelSaveEpoch"]
        lr_base     = self.config["lr"]
        batch_size  = self.config["batchSize"]
        test_bs     = self.config["testBatchSize"]
        total_epoch = self.config["totalEpoch"]
        # lrDecayStep = self.config["lrDecayStep"]
        # [more configurations here]
        
        # get the dataloaders
        train_loader= self.train_loader
        test_loader = self.test_loader

        total_test= len(test_loader)

        # use the tensorboard
        if self.config["useTensorboard"]:
            from utilities.utilities import build_tensorboard
            tensorboard_writer = build_tensorboard(self.config["projectSummary"])
        
        print("build models...")
        # TODO [import models here]

        # print and recorde model structure
        self.reporter.writeInfo("Model structure:")
        
        # TODO [replace this]
        network = model()
        self.reporter.writeModel(model.__str__())
        
        # train in GPU
        if self.config["cuda"] >=0:
            network = network.cuda()

        start = 0

        if self.config["phase"] == "finetune":
            model_epoch = self.config["checkpointEpoch"]
            model_path = os.path.join(ckpt_dir, "epoch%d_resnet50.pth"%model_epoch)
            network.load_state_dict(torch.load(model_path))
            print('loaded trained backbone model epoch {}...!'.format(model_epoch))
            start = model_epoch
        
        print("build the optimizer...")

        # Optimizer
        # TODO replace below lines
        optimizer = torch.optim.SGD(class_net.parameters(), 
                            lr=lr_base, momentum=momentum, weight_decay=5e-4) # [replace this]

        # Loss
        # TODO replace below lines
        criterion = nn.CrossEntropyLoss() # [replace this]
        
        # Caculate the epoch number
        print("prepare the dataloaders...")
        total_len   = self.config["dataLen"]
        step_epoch  = total_len//batch_size     # steps in each epoch
        total_step  = total_step * batch_size
        print("Total step=%d in each epoch"%step_epoch)

        # Start time
        import datetime
        print("Start to train at %s"%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        
        print('Start   ======  training...')
        start_time = time.time()
        for epoch in range(start, total_epoch):
            for step in range(step_epoch):
                # Set the networks to train mode
                network.train()
                # [add more code here]
                
                # read the training data
                content_images, label  = train_loader.next()# [replace this]
                label           = label.long()              # [replace this]
                
                # [inference code here]
                # [caculate losses]

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
                    cum_step = (total_step1 * epoch + step + 1)
                    
                    print("[{}], Elapsed [{}], Elapsed [{}/{}], Step [{}/{}], loss: {:.4f}, acc: {:.3f}%".
                        format(self.config["version"], elapsed, epoch, total_epoch, step + 1, total_step, 
                                loss_curr.item(), 100. * correct / total_label))
                    self.reporter.writeTrainLog(epoch+1,step+1,
                                "loss: {:.4f}, acc: {:.3f}%".format(loss_curr.item(), 100. * correct / total_label))
                    
                    # write training information into tensorboard log files
                    if self.config["useTensorboard"]:
                        tensorboard_writer.add_scalar('data/loss', loss_curr.item(), cum_step) # [replace this]
            
            # adjust the learning rate
            if epoch == 40 or epoch ==75:
                print("Learning rate decay")
                for p in optimizer.param_groups:
                    p['lr'] *= 0.1
                    print("Current learning rate is %f"%p['lr'])

            # save the checkpoint
            if (epoch+1) % model_freq==0:
                print("Save epoch %d model checkpoint!"%(epoch+1))
                torch.save(class_net.state_dict(),
                        os.path.join(ckpt_dir, 'epoch{}_resnet50.pth'.format(epoch + 1)))

                # test the checkpoint
                class_net.eval()
                total_label = 0
                correct     = 0
                with torch.no_grad():
                    for _ in tqdm(range(total_test//test_batch_size)):
                        content, label = test_loader()
                        if self.config["cuda"] >=0:
                            content = content.cuda() 
                            label   = label.cuda()      
                        predict = class_net(content)
                        _, predicted = predict.max(1)
                        total_label += label.size(0)
                        correct += predicted.eq(label).sum().item()
                    test_acc = 100. * correct / total_label
                    print("Test Acc: {:.3f}%".format(test_acc))
                    self.reporter.writeTrainLog(epoch+1,step+1,
                                "Test Acc: {:.3f}%".format(test_acc))
                    if self.config["useTensorboard"]:
                        tensorboard_writer.add_scalar('data/acc', test_acc, (total_step1 * (epoch+1)))