# -*- coding: utf-8 -*-
"""
Created on Mon May 10 10:40:17 2021

"""

#%%

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import numpy as np
import pdb

#%%

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


#%%

def get_targetsNanchors(targets, anchors, strides, grid_size):
    
    #a1 = torch.Tensor(anchors) 
    #a1 = a1/strides
    a1 = anchors/strides
    
    t = targets * torch.tensor([1,1,grid_size,grid_size,grid_size,grid_size]).to(targets.device)
    t = t.unsqueeze(0).repeat((3,1,1))
    
    total_bbox = targets.shape[0]
    a_ids = torch.Tensor([0,1,2]).unsqueeze(1).repeat(1, total_bbox).to(targets.device)
    t = torch.cat((t, a_ids[...,None]),2)
    
    at_wh_ration = t[...,4:6].clone() #TODO: Check with IOU
    r1 = at_wh_ration / a1[:,None]
    r2 = a1[:,None] / at_wh_ration
    
    r =  torch.cat((r1,r2),2)
    max = r.max(2)[0]
    r_max = torch.cat((r, max[...,None]),2)
    j = r_max[...,4] < 4
    
    f_targets = t[j]
    f_anchors = a1[f_targets[...,6].long()]

    return f_targets, f_anchors

#%%
def map_bbox_to_anchors(targets, model, p):
    """
    Parameters
    ----------
    targets : TYPE
        DESCRIPTION.
    model : Detector model, need it to fetch anchors
        DESCRIPTION.

    Returns
    -------
    None.

    """   
    a1 = torch.Tensor(model.a_13_x_13.copy()).to(targets.device)
    a2 = torch.Tensor(model.a_26_x_26.copy()).to(targets.device)
    a3 = torch.Tensor(model.a_52_x_52.copy()).to(targets.device)
    
    s1 = model.detect_13_x_13.stride 
    s2 = model.detect_26_x_26.stride
    s3 = model.detect_52_x_52.stride 
    
    gs1 = p[0].shape[2]
    gs2 = p[1].shape[2]
    gs3 = p[2].shape[2]
    
    ts1, anchs1 = get_targetsNanchors(targets, a1, s1, gs1)   
    ts2, anchs2 = get_targetsNanchors(targets, a2, s2, gs2)
    ts3, anchs3 = get_targetsNanchors(targets, a3, s3, gs3) 
    
    layers_targets = [ts1, ts2, ts3]
    layers_anchors = [anchs1, anchs2, anchs3]
    
    return layers_targets, layers_anchors


#%%

def yl_loss(predictions, yl_targets, yl_targets_anchors, device):
    
    # Three losses #TODO Check Loss Methods
    class_criteria = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device) )
    objectness_criteria = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=device) )
    
    #bbox_loss, class_loss, objectness_loss = 0., 0., 0.
    bbox_loss, class_loss, objectness_loss = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    
    # Compute loss per layer
    balance = [4.0, 1.0, 0.4, 0.1]  # P3-P6
    #pdb.set_trace()
    for layer_id, layer_predictions in enumerate(predictions):
        
        # bbox loss, class loss, objectness loss
        t_objectness = torch.zeros_like(layer_predictions[...,4], device=device)
        total_bbox = yl_targets[layer_id].shape[0]
        
        if total_bbox:
            #pdb.set_trace()
            gx, gy = yl_targets[layer_id][...,2:4].long().T #TODO fill <0 with 0
            gx.clamp_(0, layer_predictions.shape[2] - 1)
            gy.clamp_(0, layer_predictions.shape[3] - 1)
            
            b,cl,anchs = yl_targets[layer_id][...,[0,1,6]].long().T
                    
            # Get predictions
            ps = layer_predictions[b, anchs, gy, gx]
            pxy = ps[:,:2] .sigmoid() * 2. - 0.5
            pwh = (ps[:,2:4] .sigmoid() * 2) ** 2 * yl_targets_anchors[layer_id]
            pbox = torch.cat((pxy,pwh), 1)
            tbox = yl_targets[layer_id][...,2:6]
            
            #bbox loss
            iou = bbox_iou(pbox.T, tbox, x1y1x2y2=False, CIoU=True)
            iou_loss = (1.0 - iou).mean()
            bbox_loss += iou_loss
            
            model_gr = 1 #TODO: Look later
            
            #Objectness loss
            t_objectness[b, anchs, gy, gx] = (1.0 - model_gr) + model_gr * iou.detach().clamp(0).type(t_objectness.dtype)
            #t_objectness_loss = objectness_criteria(layer_predictions[...,4],
            #                                        t_objectness) * balance[layer_id] #TODO: check without balance
            
            # class loss Or classification loss
            pclass = ps[:,5:]
            t_cl = torch.zeros_like(pclass)
            t_cl[range(total_bbox), cl] = 1
            class_loss += class_criteria(pclass, t_cl)
            
        #objectness_loss += t_objectness_loss
        objectness_loss += objectness_criteria(layer_predictions[...,4], t_objectness) * balance[layer_id]
        
    bbox_loss *= 0.05 * (3. / 2)
    class_loss *= 0.31
    objectness_loss *= (3. / 2)
        
    final_loss = bbox_loss + class_loss + objectness_loss
    batch_size = t_objectness.shape[0]
        
    components = torch.cat((bbox_loss,objectness_loss, class_loss,  final_loss))
    return final_loss * batch_size, components.detach().cpu()

#%%

def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def yl_nms(total_preds):
    #pdb.set_trace()
    total_img = total_preds.shape[0]
    obj_scores = total_preds[...,4] > 0.5
    
    # Iterate over each image
    results = [torch.zeros((0, 6))] * total_preds.shape[0]
    for idx, imgPred in enumerate(total_preds):
        
        img_obj_score =  obj_scores[idx]
        imgF= imgPred[img_obj_score] 
        
        if not imgF.shape[0]:
            continue
        
        # multiple objnessScore with class scores
        imgF[:,5:] *=  imgF[:,4:5]
        img_box = xywh2xyxy(imgF[:,:4])
               
        rowId,colId = (imgF[:,5:] > 0.5).nonzero().T
        x = torch.cat((img_box[rowId],imgF[rowId, colId+5, None], colId[:,None].float()),1 )
                      
        
        nBox = x.shape[0]
        if not nBox:
            continue
        elif nBox > 30000:
            # select top 30k
            x = x[ x[:,4].argsort(descendig=True)[:30000]]
        c = x[:, 5:6] * 4096    #TODO check loss without c and remove c
        boxes, scores = x[:, :4] + c, x[:, 4] # x[:,:4], x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_threshold=0.5)
        
        if i.shape[0] > 300:
            i = i[:300]       
        results[idx] = x[i]
    return results

#%%

def bbox_iou_inTP(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    #TODO : reduce two bbox_iou methods to one 
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


#%%

def isPredictionTP(yl_nms1, targets, numOfSamples, iou_threshold=0.5):
    # adding last column as an ID to each box
    targets = torch.cat((targets, 
                         torch.arange(targets.shape[0])[:,None].to(targets.device)), 1)
    res = torch.zeros((0,4)).to(targets.device)
    
    for i in range(len(yl_nms1)):
        
        scoreNlabel = yl_nms1[i][:, 4:6].to(targets.device)
        tp = torch.zeros(scoreNlabel.shape[0]).to(targets.device)
        samplePreds = torch.tensor(i+numOfSamples, dtype=float).repeat(scoreNlabel.shape[0]).to(targets.device) 
        
        subset = torch.cat( (tp[:,None], scoreNlabel, samplePreds[:,None]), 1)
        subsetBox = yl_nms1[i][:, :4]
        
        # check subset label with target label
        sampleTargets = targets[ targets[:,0] == i,:]
        detected_boxes = []
        for sid, s in enumerate(subset):
            s_label = s[2]
            #break
            if s_label in sampleTargets[:,1]:
                #subTargets = sampleTargets[ sampleTargets[:,1] == s_label]
                #subTargetsB = subTargets[:,2:]
                sTargetsB = sampleTargets[ sampleTargets[:,1] == s_label][:,2:]
                sPredsB = subsetBox[sid]
                # check iou
                iou, box_index = bbox_iou_inTP(sPredsB.unsqueeze(0), sTargetsB[:,:4]).max(0)
                box_id = sTargetsB[box_index][-1]
                if iou >= iou_threshold and box_id not in detected_boxes: 
                    subset[sid, 0] = 1
                    detected_boxes.append(box_id)
                    
        res = torch.cat( (res, subset ),0)
    
    return res

def get_ap_n_f1(tp_matrix, labels):
    i = tp_matrix[:,1].sort(descending=True)[1]
    res2 = tp_matrix[i].numpy()
    
    unqClasses = np.unique(labels)
    
    ap, p, r = [], [], []
    
    for idx, c in enumerate(unqClasses):
        #print (c,idx)
        if c == 57.:
            print("class 57")
        c_preds = res2[res2[:,2] == c]
        n_actual = (labels == c).sum()
        
        if not c_preds.shape[0]:
            ap+=[0]; p+=[0]; r+=[0]
            continue
        
        fpc = (1-c_preds[:,0]).cumsum()
        tpc = (c_preds[:,0]).cumsum()
        
        recall_curve = tpc / (n_actual+ 1e-16)
        precison_curve = tpc / (fpc + tpc)
        
        r.append(tpc[-1]/n_actual)
        p.append(tpc[-1]/ ( tpc[-1] + fpc[-1]))
       
        #TODO Recall curve doesnot look accurate 
        #ap1 = compute_ap(recall_curve, precison_curve)
        #recall_curve.append(0) 
        #precison_curve.append(0)
        
        ar = np.concatenate(([0], recall_curve))
        bp = np.concatenate(([0], precison_curve))
        
        ap_val = np.sum((ar[1:] - ar[:-1]) * bp[1:])
        ap.append(ap_val)
       
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)
    return p, r, ap, f1, unqClasses.astype("int32")


#%%



