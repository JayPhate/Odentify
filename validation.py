# -*- coding: utf-8 -*-
"""
Created on Thu May 20 13:14:29 2021

"""
from utils.iaa_tranformation import MyIaaAugmenters
from utils.dataset_essentials import MyDataset

from torch.utils.data import DataLoader
from torchvision import transforms

import dnm_model as model
import utils.cost_essentials as cost_essen
import torch
import os


#%%

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y
  
def model_eval(yl_model, data_path, batch_size = 4):
    
    yl_model.eval() 
    my_nw_trans = transforms.Compose([MyIaaAugmenters()])
    
    #Xy_data_path = "data\\coco\\train_data_path.txt"
    #root_path = os.getcwd()
    
    my_nw_dataset = MyDataset(train_data_path=data_path,  
                           transforms=my_nw_trans )
                  
    my_nw_dataset_loader = DataLoader(my_nw_dataset, 
                                      batch_size=batch_size, 
                                      shuffle=False,                               
                                      collate_fn=my_nw_dataset.collate_fn) #collate
   
    labels, res_tp = [], torch.zeros((0,4)).to(device)
    numOfSampleProcessed = 0
    for b, (paths, imgs, bb_targets) in enumerate(my_nw_dataset_loader):
        
        labels += bb_targets[:, 1].tolist()
        imgs, bb_targets = imgs.to(device), bb_targets.to(device)
        
        with torch.no_grad():
            X, X_forwards, yl_outs = yl_model(imgs)
            preds = torch.cat(yl_outs,1)
            results = cost_essen.yl_nms(preds)       
        
        bb_targets = xywh2xyxy(bb_targets[:,2:])
        yl_s1 = cost_essen.isPredictionTP(results, bb_targets, numOfSampleProcessed)
        
        res_tp = torch.cat( (res_tp, yl_s1 ),0)
        numOfSampleProcessed += batch_size
    
    if not res_tp.shape[0]:
        return [0],[0],[0],[0],None
    
    precision, recall, AP, f1, ap_class = cost_essen.get_ap_n_f1(res_tp, labels) 
    return precision, recall, AP, f1, ap_class
   
#%%
if __name__== "__main__":
    print("Testing the model!")
    
    path = "data\\coco\\coco.names"
    fp = open(path, "r")
    class_names = fp.read().split("\n")[:-1]
    fp.close()
    
    # Load the model
    model_path = 'weights\\yl_model_1.pt'
    dnm_yl = model.DNM().to(device)
    dnm_yl.load_state_dict(torch.load(model_path))
    
    precision, recall, AP, f1, ap_class = model_eval(dnm_yl)
    
    if len(ap_class) > 0:
        for i, c in enumerate(ap_class):
            print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")
        print(f"mAP: {AP.mean()}")
    else:
        print(f"mAP: {0.0}")
    
    
    
    

    
    
    

    


