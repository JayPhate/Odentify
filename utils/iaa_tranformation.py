# -*- coding: utf-8 -*-
"""
Created on Sat May  8 11:44:11 2021

"""
#%%

from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from torchvision import transforms

import imgaug.augmenters as iaa
import torch
import torch.nn.functional as F
import numpy as np


np.set_printoptions(precision=10, suppress=True)
torch.set_printoptions(sci_mode=False)

import pdb

#%%

def get_absoluteLabels(data):
    img, boxes = data
    w, h, _ = img.shape
    boxes[:,[1,3]] *= h
    boxes[:,[2,4]] *= w
    return img, boxes

def get_relativeLabels(data):
    img, boxes = data
    w, h, _ = img.shape
    boxes[:,[1,3]] /= h
    boxes[:,[2,4]] /= w
    return img, boxes

def xywh2xyxy_np(x):
    y = np.zeros_like(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def xyxy2_xywh_np(x):
    y = np.zeros_like(x)
    y[...,0] = (x[...,0] + x[...,2]) / 2 # x = (x1 + x1) /2
    y[...,1] = (x[...,1] + x[...,3]) / 2 # y
    y[...,2] = (x[...,2] - x[...,0])  # w
    y[...,3] = (x[...,3] - x[...,1])  # h
    return y

def resize(image, size):
    #pdb.set_trace()
    #try: 
    #    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    #except:
    #    pdb.set_trace()
    #    raise SystemExit('error in code want to exit')
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

#%%
class MyIaaAugmenters():
    
    def __init__(self):
        self.transformer = iaa.Sequential([
            iaa.Dropout([0.0, 0.01]),
            iaa.Sharpen((0.0, 0.1)),
            iaa.Affine(rotate=(-10, 10), translate_percent=(-0.1,0.1)),  # rotate by -45 to 45 degrees (affects segmaps)
            iaa.AddToBrightness((-10, 10)), 
            iaa.AddToHue((-5, 5)), 
            iaa.Fliplr(0.5),
            iaa.PadToAspectRatio(1.0, position="center-center")
            ])
        
        self.transformer = self.transformer.to_deterministic()
    
    def __call__(self, data):
        #pdb.set_trace()
        img, boxes = get_absoluteLabels(data)
        
        boxes[...,1:] = xywh2xyxy_np(boxes[...,1:])
        
        bounding_boxes = BoundingBoxesOnImage(
            [BoundingBox(*bbox[1:], label = bbox[0]) for bbox in boxes ],
                                              shape=img.shape)
        
        img, bounding_boxes = self.transformer(
            image=img, 
            bounding_boxes=bounding_boxes)
        
        # Clip out of image boxes
        bounding_boxes = bounding_boxes.clip_out_of_image()
        
        
        boxes = [ [box.label, box.x1, box.y1, box.x2, box.y2]  for bid, box in enumerate(bounding_boxes)]
        boxes = np.array(boxes)
        
        boxes[...,1:] = xyxy2_xywh_np(boxes[...,1:])
        
        img, boxes = get_relativeLabels( (img, boxes))       
        
        #Convert to tensors
        img = transforms.ToTensor()(img)
        
        #label, x1.y1,x2,y2
        bbx_targets = torch.zeros((len(boxes), 6))
        bbx_targets[:, 1:] = transforms.ToTensor()(boxes)
        #pdb.set_trace()
        return img, bbx_targets
        
 #%%
     

      
        
        
        
        
        
        

