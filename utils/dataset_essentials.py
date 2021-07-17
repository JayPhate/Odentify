# -*- coding: utf-8 -*-
"""
Created on Wed May 19 11:02:12 2021

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import pdb
import random

#from augmentation import *
from .iaa_tranformation import *

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import logging
logger = logging.getLogger(__name__)

class MyDataset(Dataset):
    
    def __init__(self, train_data_path, transforms=None): 
        
        
        self.train_data_path = pd.read_csv(train_data_path, sep="\n")
        
        #self.train_data_path = self.train_data_path.iloc[9824:,:]
        print(f"train_data_path shape: {self.train_data_path.shape}")
        
        self.train_imgs_paths = self.train_data_path.iloc[:,0].apply(lambda x : x.split("|")[0]).tolist()
        self.train_lbls_paths = self.train_data_path.iloc[:,0].apply(lambda x : x.split("|")[1]).tolist()
        
        #self.root_dir = root_dir
        self.transform = transforms
        self.batch_count = 0
        self.img_size = 416
        self.multiscale = True
        
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        
    def __len__(self):
        return( len(self.train_imgs_paths))
    
    def __getitem__(self, index):
        import pdb
        #label_file = np.loadtxt(os.path.join(self.root_dir, self.train_lbls_paths[index]))
        
        label_file = np.loadtxt(self.train_lbls_paths[index])
        
        if label_file.ndim == 1:
            label_file = label_file.reshape(1,-1)
        
        #pdb.set_trace()
        #img_p = os.path.join(self.root_dir, self.train_imgs_paths[index])
        img_p = self.train_imgs_paths[index]
        
        img_file = np.array(Image.open(img_p).convert("RGB"), dtype=np.uint8)
        
        
        if self.transform:
            #img_file, label_file = self.transform((img_file, label_file))
            try:
                img_file, label_file = self.transform((img_file, label_file))
            except:
                #pdb.set_trace()
                img_file, label_file = None, None
                print("Could not apply transform.")
                logger.error(f"Failed to transform => {img_p} ")
                
                #return
        
        #if isinstance(img_file, (np.ndarray, np.generic) ):
        #if isinstance(img_file, torch.Tensor)  :
        #    print("Transformed failed..")
        #    img_file, label_file = None, None
            #pdb.set_trace()
            
        return img_p, img_file, label_file
    
    def collate_fn(self, batch):
        import pdb
        self.batch_count += 1

        # Drop invalid images
        #pdb.set_trace()
        batch = [data for data in batch if data[1] is not None]

        paths, imgs, bb_targets = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        
        #pdb.set_trace()
        # Resize images to input shape
        #F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # Add sample index to targets
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
            
        bb_targets = torch.cat(bb_targets, 0)

        return paths, imgs, bb_targets


def collate(batch): 
       # Drop invalid images
       batch = [data for data in batch if data is not None]

       paths, imgs, bb_targets = list(zip(*batch))
       return paths, imgs, bb_targets
   

#%%
