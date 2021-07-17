# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 16:51:25 2021
Checking colab access

"""

#%%

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import model_essentials as mod_essen
import pdb

#%%

class DNM(nn.Module):
    """
    Darkent Model with YOLO layer
    
    """
    def __init__(self):
        super().__init__()
        self.three_to_64 = mod_essen.chn_3_to_64()
        self.sixtyFour_to_128, self.res_block_pos_128 = mod_essen.chn_sixtyFour_to_128()
        self.conv_128_to_256, self.res_block_pos_256 = mod_essen.chn_128_256()
        self.conv_256_512, self.res_block_pos_512 = mod_essen.chn_256_512()
        self.conv_512_1024, self.res_block_pos_1024 = mod_essen.chn_512_1024()
        self.conv_1024_255 = mod_essen.chn_1024_255()         
        
        self.conv_1024_256 = mod_essen.chn_1024_256()
        self.conv_768_255 = mod_essen.chn_768_255()
        self.conv_512_128 = mod_essen.chn_512_128()
        self.conv_384_255 = mod_essen.chn_384_255()
        
        #anchors
        self.a_13_x_13 = [[116, 90], [156, 198], [373, 326]]
        self.a_26_x_26 = [[30, 61], [62, 45], [59, 119]]
        self.a_52_x_52 = [[10, 13], [16, 30], [33, 23]]
        
        self.detect_13_x_13 = mod_essen.DetectionLayer(self.a_13_x_13)
        self.detect_26_x_26 = mod_essen.DetectionLayer(self.a_26_x_26)
        self.detect_52_x_52 = mod_essen.DetectionLayer(self.a_52_x_52)
              
    def forward(self, X):
        frwd_operations, detection_output = [], []
        
        for one_conv in self.three_to_64:
            X = one_conv(X)
            frwd_operations.append(X) 
        # residual block
        X = frwd_operations[-1] + frwd_operations[-3] 
        frwd_operations.append(X) 
        
        # From 64 channel to 128, two residual blocks
        for i in self.res_block_pos_128: 
            if i == -1:
                X = frwd_operations[-1] + frwd_operations[-3]
            else:
                one_conv = self.sixtyFour_to_128[i]
                X = one_conv(X)    
            frwd_operations.append(X)
        
        # From 128 channel to 256, eight residual blocks
        for i in self.res_block_pos_256: 
            if i == -1:
                X = frwd_operations[-1] + frwd_operations[-3]
            else:
                one_conv = self.conv_128_to_256[i]
                X = one_conv(X)    
            frwd_operations.append(X)
            
       # From 256 channel to 512, eight residual blocks
        for i in self.res_block_pos_512: 
            if i == -1:
                X = frwd_operations[-1] + frwd_operations[-3]
            else:
                one_conv = self.conv_256_512[i]
                X = one_conv(X)    
            frwd_operations.append(X)     
        
        # From 512 channel to 1024, four residual blocks
        for i in self.res_block_pos_1024: 
            if i == -1:
                X = frwd_operations[-1] + frwd_operations[-3]
            else:
                one_conv = self.conv_512_1024[i]
                X = one_conv(X)    
            frwd_operations.append(X)     
        
        # From 1024 to 255 channel, no residual block
        for one_conv in self.conv_1024_255:
            X = one_conv(X)
            frwd_operations.append(X) 
        
        X = self.detect_13_x_13(X) # yl_13_x_13
        frwd_operations.append(X)
        detection_output.append(X)
        
        # get yl layer for 26 x 26
        up_sample1 = F.interpolate(self.conv_1024_256(frwd_operations[80]),
                                   scale_factor=2)
                                     
        X = torch.cat([frwd_operations[61],up_sample1],1)
        #pdb.set_trace()
        # chn 768 to 255               
        for one_conv in self.conv_768_255:
            X = one_conv(X)
            frwd_operations.append(X)
        
        X = self.detect_26_x_26(X) # yl_26_x_26
        frwd_operations.append(X)
        detection_output.append(X)
        
        up_sample2 = F.interpolate(self.conv_512_128(frwd_operations[88]),
                                   scale_factor=2)
        X = torch.cat([frwd_operations[36],up_sample2],1)
        
        # conv_384_255
        for one_conv in self.conv_384_255:
            X = one_conv(X)
            frwd_operations.append(X)
        
        X = self.detect_52_x_52(X) # yl_52_x_52
        frwd_operations.append(X)
        detection_output.append(X)
        
        return X, frwd_operations, detection_output
    
#%%

