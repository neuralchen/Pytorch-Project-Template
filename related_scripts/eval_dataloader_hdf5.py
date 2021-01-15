#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: eval_dataloader_hdf5.py
# Created Date: Tuesday January 12th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Friday, 15th January 2021 11:34:33 am
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################

import os
import glob
import h5py
import torch
import random

from pathlib import Path


class EvalDataset:
    def __init__(self, 
                h5_path,
                batch_size=16):
        """Initialize and preprocess the CelebA dataset."""
        self.batch_size = batch_size
        self.pointer    = 0

        self.h5_path= h5_path
        self.h5file = h5py.File(h5_path,'r')
        self.keys   = self.h5file["__len__"][()] #86366
        self.length = self.keys
        self.keys = [str(k) for k in range(self.keys)]

    def __call__(self):
        """Return one batch images."""
        
        # if self.pointer>=self.length:
        #     self.pointer = 0

        if self.pointer>=self.length:
            self.pointer = 0
            a = "The end of the story!"
            raise StopIteration(print(a))
        elif (self.pointer+self.batch_size) > self.length:
            end = self.length
        else:
            end = self.pointer+self.batch_size
        for i in range(self.pointer, end):
            iii = self.keys[i]
            hr  = torch.from_numpy(self.h5file[iii+"hr"][()])
            lr  = torch.from_numpy(self.h5file[iii+"lr"][()])
            
            if (i-self.pointer) == 0:
                hr_ls   = hr.unsqueeze(0)
                lr_ls   = lr.unsqueeze(0)
            else:
                hr_ls   = torch.cat((hr_ls,hr.unsqueeze(0)),0)
                lr_ls   = torch.cat((lr_ls,lr.unsqueeze(0)),0)
        self.pointer = end
        return lr_ls, hr_ls
    
    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.h5_path + ')'