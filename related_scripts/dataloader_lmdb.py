
# -*- coding:utf-8 -*-
#############################################################
# File: dataloader_lmdb.py
# Created Date: Sunday January 10th 2021
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 12th January 2021 1:19:58 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2021 Shanghai Jiao Tong University
#############################################################


import os
import glob
import torch
import random
from PIL import Image
from pathlib import Path
from torch.utils import data
import torchvision.datasets as dsets
from torchvision import transforms as T
import torchvision.transforms.functional as F
import lmdb
import pyarrow as pa
import numpy as np
torch.multiprocessing.set_start_method('spawn')


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = loader
        self.dataiter = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.__preload__()

    def __preload__(self):
        try:
            self.content, self.style, self.label = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            self.content, self.style, self.label = next(self.dataiter)
            
        with torch.cuda.stream(self.stream):
            self.content= self.content.cuda(non_blocking=True)
            self.style  = self.style.cuda(non_blocking=True)
            self.label  = self.label.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            # self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        content = self.content
        style   = self.style
        label   = self.label 
        self.__preload__()
        return content, style, label
    
    def __len__(self):
        """Return the number of images."""
        return len(self.loader)

class LmdbDataset(data.Dataset):
    """Dataset class for the Artworks dataset and content dataset."""
    def __init__(self, 
                db_path,
                data_transform=None):
        """Initialize and preprocess the lmdb dataset."""
        self.db_path= db_path
        self.env    = lmdb.open(db_path,
                            subdir=os.path.isdir(db_path),
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
                            
        with self.env.begin(write=False) as txn:
            self.length = pa.deserialize(txn.get(b'__len__'))
            self.keys   = pa.deserialize(txn.get(b'__keys__'))
        self.data_transform = data_transform

    def __getitem__(self, index):
        """Return low-resolution frames and its corresponding high-resolution."""
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = pa.deserialize(byteflow)

        # load image
        hr = unpacked[1]
        lr = unpacked[0]
        
        if self.data_transform is not None:
            hr = self.data_transform(hr)
            lr = self.data_transform(lr)

        return lr, hr

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def GetLoader(s_image_dir,c_image_dir, 
                style_selected_dir, content_selected_dir,
                crop_size=178, batch_size=16, num_workers=8, 
                colorJitterEnable=True, colorConfig={"brightness":0.05,"contrast":0.05,"saturation":0.05,"hue":0.05}):
    """Build and return a data loader."""
    
    s_transforms = []
    c_transforms = []
    
    s_transforms.append(T.Resize(768))
    # s_transforms.append(T.Resize(900))
    c_transforms.append(T.Resize(768))

    s_transforms.append(T.RandomCrop(crop_size,pad_if_needed=True,padding_mode='reflect'))
    c_transforms.append(T.RandomCrop(crop_size))

    s_transforms.append(T.RandomHorizontalFlip())
    c_transforms.append(T.RandomHorizontalFlip())
    
    s_transforms.append(T.RandomVerticalFlip())
    c_transforms.append(T.RandomVerticalFlip())

    if colorJitterEnable:
        if colorConfig is not None:
            print("Enable color jitter!")
            colorBrightness = colorConfig["brightness"]
            colorContrast   = colorConfig["contrast"]
            colorSaturation = colorConfig["saturation"]
            colorHue        = (-colorConfig["hue"],colorConfig["hue"])
            s_transforms.append(T.ColorJitter(brightness=colorBrightness,\
                                contrast=colorContrast,saturation=colorSaturation, hue=colorHue))
            c_transforms.append(T.ColorJitter(brightness=colorBrightness,\
                                contrast=colorContrast,saturation=colorSaturation, hue=colorHue))
    s_transforms.append(T.ToTensor())
    c_transforms.append(T.ToTensor())

    s_transforms.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    c_transforms.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    
    s_transforms = T.Compose(s_transforms)
    c_transforms = T.Compose(c_transforms)

    content_dataset = TotalDataset(c_image_dir,s_image_dir, content_selected_dir, style_selected_dir
                        , c_transforms,s_transforms)
    content_data_loader = data.DataLoader(dataset=content_dataset,batch_size=batch_size,
                    drop_last=True,shuffle=True,num_workers=num_workers,pin_memory=True)
    prefetcher = DataPrefetcher(content_data_loader)
    return prefetcher

def GetValiDataTensors(
                image_dir=None,
                selected_imgs=[],
                crop_size=178,
                mean = (0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
            ):
            
    transforms = []
    
    transforms.append(T.Resize(768))

    transforms.append(T.RandomCrop(crop_size,pad_if_needed=True,padding_mode='reflect'))

    transforms.append(T.ToTensor())

    transforms.append(T.Normalize(mean=mean, std=std))
    
    transforms = T.Compose(transforms)

    result_img   = []
    print("Start to read validation data......")
    if len(selected_imgs) != 0:
        for s_img in selected_imgs:
            if image_dir == None:
                temp_img = s_img
            else:
                temp_img = os.path.join(image_dir, s_img)
            temp_img = Image.open(temp_img)
            temp_img = transforms(temp_img).cuda().unsqueeze(0)
            result_img.append(temp_img)
    else:
        s_imgs = glob.glob(os.path.join(image_dir, '*.jpg'))
        s_imgs = s_imgs + glob.glob(os.path.join(image_dir, '*.png'))
        for s_img in s_imgs:
            temp_img = os.path.join(image_dir, s_img)
            temp_img = Image.open(temp_img)
            temp_img = transforms(temp_img).cuda().unsqueeze(0)
            result_img.append(temp_img)
    print("Finish to read validation data......")
    print("Total validation images: %d"%len(result_img))
    return result_img

if __name__ == "__main__":
    

    db_path = "G:\\RainNet\\Lmdb\\RainNet.lmdb"
    s_transforms = []

    s_transforms.append(T.RandomHorizontalFlip())
    
    s_transforms.append(T.RandomVerticalFlip())
    s_transforms = T.Compose(s_transforms)
    s_transforms = None

    lmdb = LmdbDataset(db_path,s_transforms)
    lmdb_dataloader = data.DataLoader(dataset=lmdb,batch_size=64,
                    drop_last=True,shuffle=True,num_workers=0,pin_memory=True)
    dataiter = iter(lmdb_dataloader)
    lr,hr = next(dataiter)
    import time
    import datetime
    start_time = time.time()
    for i in range(10):
        lr,hr = next(dataiter)
        print(hr.shape)
    elapsed = time.time() - start_time
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Elapsed [{}]".format(elapsed))