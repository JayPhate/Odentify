# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 10:26:16 2021

@author: v-japha
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:23:55 2021
# tensorboard --logdir=runs

"""

from utils.iaa_tranformation import *
from utils.dataset_essentials import *
from torch.utils.tensorboard import SummaryWriter
from dnm_model import *
import validation
import utils.model_essentials as mod_essen
import utils.cost_essentials as cost_essen

import torch
import pandas as pd
import numpy as np
import os
np.set_printoptions(suppress=True)

import warnings
warnings.filterwarnings("ignore")
import pdb
import logging
import logging.handlers
import datetime
import tqdm
import time


#logging.basicConfig(filename='weights/myapp.log', level=logging.DEBUG)#INFO

dateTag = datetime.datetime.now().strftime("%Y-%b-%d_%H-%M-%S")
logging.basicConfig(filename="weights/training_%s.log" % dateTag, level=logging.DEBUG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%%

def timer(start,end):
   hours, rem = divmod(end-start, 3600)
   minutes, seconds = divmod(rem, 60)
   return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds)

def get_loader(Xy_data_path, bs=2, shuffle=False ):
    
    logging.info('Preparing the data loader object for training...')
    my_nw_trans = transforms.Compose([MyIaaAugmenters()])
    
    #Xy_data_path = "data\\coco\\train_data_path.txt"
    #root_path = "C:\\ML\\pyTorch\\ObjectDetection\\PyTorch-YOLOv3\\"
    #root_path = os.getcwd()
    
    my_nw_dataset = MyDataset(train_data_path=Xy_data_path,
                          # root_dir=root_path,  
                           transforms=my_nw_trans )
                  
    my_nw_dataset_loader = DataLoader(my_nw_dataset, 
                                      batch_size=bs, 
                                      shuffle=shuffle,                               
                                      collate_fn=my_nw_dataset.collate_fn) #collate
    logging.info('Data loader object is prepared for training...')
    return my_nw_dataset_loader

#%%

def log_tensorboard_writer(writer, trainLoss,valPrecision, valRecall, valAP, valF1, epoch):
    writer.add_scalar('Loss/train', trainLoss, epoch)
    writer.add_scalar('test/Precision', valPrecision, epoch)
    writer.add_scalar('test/Recall', valRecall, epoch)
    writer.add_scalar('test/AP', valAP, epoch)
    writer.add_scalar('test/f1', valF1, epoch)  

#%%

def main(train_data_path, val_data_path, epochs=1, bs=2, epochs_checkpoint=10, display_thr=50):
    
    logging.info('Started')
    writer = SummaryWriter()
    dataLoader = get_loader(train_data_path, bs, shuffle=False)
    
    #dnm_yl = DNM().to(device)
    #optimizer = torch.optim.Adam(dnm_yl.parameters(),  lr=0.001)
    #dnm_yl = mod_essen.load_dark_net_weights(dnm_yl,'weights/darknet53.conv.74')

    train_loss = []
    val_precision, val_recall, val_ap, val_f1 = [],[],[],[]
    
    start = int(np.loadtxt("weights/currentEpoch.txt")) if os.path.exists("weights/currentEpoch.txt") else 0
   
    
    print("Training the model....")
    for epoch in range(start, epochs):
        
        #print(epoch, file=open("weights/currentEpoch.txt", "w"))
        #dnm_yl.train()
        
        start_time = time.time()
        for b, (paths, imgs, bb_targets) in enumerate(tqdm.tqdm(dataLoader,desc=f"Training Epoch {epoch+1}")):
            
            imgs, bb_targets = imgs.to(device), bb_targets.to(device)
            
            if (b+1) % display_thr == 0:
                logging.info(f"E:{epoch+1}, B:{b+1} => Total paths:{len(paths)}, \
Total Images: {len(imgs)}, \
Total Bbox: {len(bb_targets)}")
                for p in paths:
                    logging.info(f"############# ===> {p}")

        logging.info(f"Epoch {epoch+1} time: {timer(start_time, time.time())}")
        
#%%

## save the model
#model_name = "yl_model_" + "1" + ".pt" 
#torch.save(dnm_yl.state_dict(), os.path.join("weights", model_name))
# tensorboard --logdir=runs

if __name__ == '__main__':
    #train_data_path = "data\\coco\\train_data_path.txt"
    #val_data_path = "data\\coco\\train_data_path.txt"
    
    train_data_path = "C:\\Users\\v-japha\\Desktop\\pyPractice\\prepareData\\train_paths.txt"
    val_data_path = "C:\\Users\\v-japha\\Desktop\\pyPractice\\prepareData\\validation_paths.txt"
    
    main(train_data_path, val_data_path, 
         epochs=2, bs=32, 
         epochs_checkpoint=10, display_thr=1)






