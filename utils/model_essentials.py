# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 16:57:10 2021
"""

#%%

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


#%%

"""
Process 3 channels to 64 channels

64 channels to 128 channels==> repeat this 2x

128 channels to 256 channels ==> repeat this 2x

256channels  to 512 channels ==> repeat this 8x

"""
def get_conv_operation(inp_ch, out_ch, f, s=1, p=0): 
    module = nn.Sequential()
    module.add_module("c1", 
            nn.Conv2d(inp_ch, out_ch,f, s, p, bias=False)) # R: 32 x 416 x 416
    module.add_module("b1", 
                      nn.BatchNorm2d(out_ch, eps=1e-05))
    module.add_module("r1", nn.LeakyReLU(negative_slope=0.1))
    return module 

def get_conv_255(inp_ch, out_ch, f, s=1, p=0):
    module = nn.Sequential()
    module.add_module("c_255", nn.Conv2d(inp_ch, out_ch,f, s, p))
    return module
    
def chn_3_to_64():
    
    """
    Input : 3 x 416 x 416 (ch x h x w )
    Convultion operations:
            ....      
    return: 64 x h x w       
    """
    module_list = nn.ModuleList()
    
    c1 = get_conv_operation(3, 32, 3, 1, 1)  # R: 32 x 416 x 416
    
    c2 = get_conv_operation(32, 64, 3, 2, 1) # R: 64 x 208 x 208 ==> a
    c3 = get_conv_operation(64, 32, 1, 1)    # R: 32 x 208 x 208
    c4 = get_conv_operation(32, 64, 3, 1, 1) # R: 64 x 208 x 208 ==> b
    
    # add a and b in forward call
    module_list.extend([c1,c2,c3,c4])
    
    return module_list

def chn_sixtyFour_to_128():
    """
    Input: 64 x 208 x 208
    
    return: 128 x 104 x 104
    """
    module_list = nn.ModuleList()
    
    # Iteratative twice as we hv two reidual block
    # when convolving from 64 channel to 128 channel
    res_block_pos,pos = [],0
    
    c1 = get_conv_operation(64, 128, 3, 2, 1) #R: 128 x 104 x 104 ==> a
    module_list.append(c1)
    res_block_pos.append(pos)  
    for x in range(1,3):
        c2 = get_conv_operation(128, 64, 1, 1)    #R: 64 x 104 x 104 
        c3 = get_conv_operation(64, 128, 3, 1, 1) #R: 128 x 104 x 104 ==> b
        module_list.extend([c2,c3])
        res_block_pos.extend([pos+1, pos+2, -1])
        pos = pos + 2
        
           
# =============================================================================
#     c1 = get_conv_operation(64, 128, 3, 2, 1) #R: 128 x 104 x 104 ==> a
#     c2 = get_conv_operation(128, 64, 1, 1)    #R: 64 x 104 x 104 
#     c3 = get_conv_operation(64, 128, 3, 1, 1) #R: 128 x 104 x 104 ==> b
#     #c4 = get_conv_operation(128, 64, 1, 1)   #R: 64 x 104 x 104 
#     
#     # add a and b in forward call
#     module_list.extend([c1,c2,c3])
#     
# =============================================================================
    return module_list, res_block_pos
    

def chn_128_256():
    """
    Input: 128, 104, 104

    Returns: 256, 52, 52
    -------
    None.

    """
    module_list = nn.ModuleList()
    res_block_pos,pos = [],0
    
    c1 = get_conv_operation(128, 256, 3, 2, 1) #R: 256 x 52 x 52 ==> a
    module_list.append(c1)
    res_block_pos.append(pos)
    
    for x in range(1,9):
        c2 = get_conv_operation(256, 128, 1, 1)    #R: 128 x 52 x 52 
        c3 = get_conv_operation(128, 256, 3, 1, 1) #R: 256 x 52 x 52 ==> b
        module_list.extend([c2,c3])
        res_block_pos.extend([pos+1, pos+2, -1])
        pos = pos + 2
    return module_list, res_block_pos
        

def chn_256_512():
    """
    Input: 256 x 52 x 52

    Returns : 512 x 52 x 52
    -------
    None.

    """ 
    module_list = nn.ModuleList()
    res_block_pos,pos = [],0
    
    c1 = get_conv_operation(256, 512, 3, 2, 1) #R: 256 x 52 x 52 ==> a
    module_list.append(c1)
    res_block_pos.append(pos)
    
    for x in range(1,9):
        c2 = get_conv_operation(512, 256, 1, 1)    #R: 128 x 52 x 52 
        c3 = get_conv_operation(256, 512, 3, 1, 1) #R: 256 x 52 x 52 ==> b
        module_list.extend([c2,c3])
        res_block_pos.extend([pos+1, pos+2, -1])
        pos = pos + 2
    return module_list, res_block_pos
    
def chn_512_1024():
    """
    Input: 512 x 26 x 26

    Returns: 1024 x 13 x 13
    -------
    None.

    """ 
    
    module_list = nn.ModuleList()
    res_block_pos,pos = [],0
    
    c1 = get_conv_operation(512, 1024, 3, 2, 1) #R: 256 x 52 x 52 ==> a
    module_list.append(c1)
    res_block_pos.append(pos)
    
    for x in range(1,5):
        c2 = get_conv_operation(1024, 512, 1, 1)    #R: 128 x 52 x 52 
        c3 = get_conv_operation(512, 1024, 3, 1, 1) #R: 256 x 52 x 52 ==> b
        module_list.extend([c2,c3])
        res_block_pos.extend([pos+1, pos+2, -1])
        pos = pos + 2
    return module_list, res_block_pos
 
    
def chn_1024_255():
    """
    Input: 1024 x 13 x 13

    Returns: 255 x 13 x 13
    -------
    None.

    """    
    module_list = nn.ModuleList()
    for x in range(1,4):
        c2 = get_conv_operation(1024, 512, 1, 1)
        c3 = get_conv_operation(512, 1024, 3, 1, 1)
        module_list.extend([c2,c3])
    
    #final =  get_conv_operation(1024, 255, 1, 1)   
    #module_list.append(final)
    final = get_conv_255(1024, 255, 1, 1)
    module_list.append(final)
    
    return module_list

def chn_1024_256():
    """
    Input: 1024 x 13 x 13

    Returns: 256 x 13 x 13
    -------
    None.

    """
    return get_conv_operation(1024, 256, 1, 1)  

def chn_512_128():
    """
    Input: 512 x 26 x 26

    Returns: 128 x 26 x 26
    -------
    None.

    """
    return get_conv_operation(512, 128, 1, 1)  


def chn_768_255():
    """
    Input: 768 x 26 x 26

    Returns: 255 x 26 x 26
    -------
    None.
    """
    module_list = nn.ModuleList()
    c1 = get_conv_operation(768, 256, 1, 1)
    module_list.append(c1)
    
    for x in range(1,3):     
        c2 = get_conv_operation(256, 512, 3, 1, 1)
        c3 = get_conv_operation(512, 256, 1, 1)
        module_list.extend([c2,c3])  
    last = get_conv_operation(256, 512, 3, 1, 1)
    module_list.append(last)
    
    #final =  get_conv_operation(512, 255, 1, 1)   
    #module_list.append(final)
    
    final = get_conv_255(512, 255, 1, 1)
    module_list.append(final)
    
    return module_list
 

def chn_384_255():
    """
    Input: 768 x 26 x 26

    Returns: 255 x 26 x 26
    -------
    None.

    """
    module_list = nn.ModuleList()
    c1 = get_conv_operation(384, 128, 1, 1)
    module_list.append(c1)
    for x in range(1,3):     
        c2 = get_conv_operation(128, 256, 3, 1, 1)
        c3 = get_conv_operation(256, 128, 1, 1)
        module_list.extend([c2,c3])
    
    last = get_conv_operation(128, 256, 3, 1, 1)
    module_list.append(last)
    
    #final =  get_conv_operation(256, 255, 1, 1)   
    #module_list.append(final)
    
    final = get_conv_255(256, 255, 1, 1)
    module_list.append(final)
    
    return module_list
 


#%%

class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super().__init__()
        self.img_size = 416
        self.num_outputs = 5 + 80 # for COCO data set
        
        self.num_anchors = len(anchors)
        anchors = torch.Tensor(anchors)
        self.register_buffer('anchors', anchors)
        self.register_buffer('anchor_grid', anchors.clone().view(1, -1, 1, 1, 2))
        self.grid = torch.zeros(1)
        
    def forward(self, x):
        
        stride = self.img_size // x.size(2)
        self.stride = stride
        bs, _, ny, nx = x.shape  
        x = x.view(bs, self.num_anchors, 
                   self.num_outputs, ny, nx).permute(
                       0, 1, 3, 4, 2).contiguous()

        if not self.training:  # inference
            if self.grid.shape[2:4] != x.shape[2:4]:
                self.grid = self._make_grid(nx, ny).to(x.device)

            y = x.sigmoid()
            y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid.to(x.device)) * stride  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid  # wh
            y = y.view(bs, -1, self.num_outputs)

        return x if self.training else y

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

        
        
#%%   
def chn_sixtyFour_to_128_V1():
    """
    Input: 64 x 208 x 208
    
    return: 128 x 104 x 104
    """
    module_list = nn.ModuleList()
    
    # Iteratative twice as we hv two reidual block
    # when convoving from 64 channel to 128 channel
    res_block_pos = []
    #[1,0,3,4,0,6]
    #(1,3),(4,6)
    for x in range(1,3):       
        c1 = get_conv_operation(64 * x, 128 * x, 3, 2, 1) #R: 128 x 104 x 104 ==> a
        c2 = get_conv_operation(128 * x, 64 * x, 1, 1)    #R: 64 x 104 x 104 
        c3 = get_conv_operation(64 * x, 128 * x, 3, 1, 1) #R: 128 x 104 x 104 ==> b
        #c4 = get_conv_operation(128, 64, 1, 1)    #R: 64 x 104 x 104
        module_list.extend([c1,c2,c3])
        
        module_list_len = len(module_list)
        for_res_block = (module_list_len-3, module_list_len-1)
        res_block_pos.append(for_res_block)
        
# =============================================================================
#     c1 = get_conv_operation(64, 128, 3, 2, 1) #R: 128 x 104 x 104 ==> a
#     c2 = get_conv_operation(128, 64, 1, 1)    #R: 64 x 104 x 104 
#     c3 = get_conv_operation(64, 128, 3, 1, 1) #R: 128 x 104 x 104 ==> b
#     #c4 = get_conv_operation(128, 64, 1, 1)    #R: 64 x 104 x 104 
#     
#     # add a and b in forward call
#     module_list.extend([c1,c2,c3])
#     
# =============================================================================
    return module_list, res_block_pos
    
  
#%%

def load_dark_net_weights(yl_model, wt_path):
    model_components = ["three_to_64", "sixtyFour_to_128","conv_128_to_256","conv_256_512","conv_512_1024"]
    
    with open(wt_path, "rb") as f:
        header = np.fromfile(f, dtype=np.int32, count=5)
        weights = np.fromfile(f, dtype=np.float32)

    ptr = 0
    for i in range(len(model_components)):
        component = getattr(yl_model, model_components[i])
        total_layers = len(component)
       
        for j in range(total_layers):
            sub_component = component[j]
            
            conv_l = sub_component[0]
            batch_l = sub_component[1]
            
            #Batch layer weights
            num_b = batch_l.bias.numel()  # Number of biases
            # Bias
            bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(batch_l.bias)
            batch_l.bias.data.copy_(bn_b)
            ptr += num_b
            
            # Weight
            bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(batch_l.weight)
            batch_l.weight.data.copy_(bn_w)
            ptr += num_b
            
            # Running Mean
            bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(batch_l.running_mean)
            batch_l.running_mean.data.copy_(bn_rm)
            ptr += num_b
            
            # Running Var
            bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(batch_l.running_var)
            batch_l.running_var.data.copy_(bn_rv)
            ptr += num_b
                 
            # Conv layer weights
            num_w = conv_l.weight.numel()
            conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_l.weight)
            conv_l.weight.data.copy_(conv_w)
            ptr += num_w
        
        #print(f"i:{i}, M:{model_components[i]}, ptr: {ptr}")
        
    return yl_model

#%%

def get_shapes_from_outputs(output_list):
    
    output_shapes = [x.shape for x in output_list]
    return output_shapes

#%%

def prediction(x,inp_dim,anchors,num_classes,CUDA=False):
    # x --> 4D feature map
    batch_size = x.size(0)
    grid_size = x.size(2)
    stride =  inp_dim // x.size(2)   # factor by which current feature map reduced from input
    #grid_size = inp_dim // stride
    
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
# 
    prediction = x.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, 
                                 grid_size*grid_size*num_anchors, 
                                 bbox_attrs)
    
    # the dimension of anchors is wrt original image.
    #We will make it corresponding to feature map
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
    #Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1) #(1,gridsize*gridsize,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    

    prediction[:,:,:2] += x_y_offset

    #log space transform height and the width
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors #width and height
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))    
    prediction[:,:,:4] *= stride    
    return prediction


