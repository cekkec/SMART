import torch
import torch.nn as nn
import torch.nn.functional as F
from os import path
from collections import defaultdict
from util.configuration import Configuration
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import os 
import numpy as np
import random

def dice_loss(input_mask, cls_gt):
    num_objects = input_mask.shape[1]
    losses = []

    for i in range(num_objects):
        mask = input_mask[:,i].flatten(start_dim=1)
        # background not in mask, so we add one to cls_gt
        gt = (cls_gt==(i+1)).float().flatten(start_dim=1)
        numerator = 2 * (mask * gt).sum(-1)
        denominator = mask.sum(-1) + gt.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        losses.append(loss)
    return torch.cat(losses).mean()

# https://stackoverflow.com/questions/63735255/how-do-i-compute-bootstrapped-cross-entropy-loss-in-pytorch
class BootstrappedCE(nn.Module):
    def __init__(self, start_warm, end_warm, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p

    def forward(self, input, target, it):
        if it < self.start_warm:
            return F.cross_entropy(input, target), 1.0

        raw_loss = F.cross_entropy(input, target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p


class DiceCE(nn.Module):
    def __init__(self, start_warm, end_warm, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p
    
    def dice_score(self, mask1, mask2):
        intersection = torch.logical_and(mask1, mask2).sum().item()
        union = mask1.sum().item() + mask2.sum().item()
    
        dice = (2.0 * intersection) / union if union > 0 else 1.0
        
        return dice
    
    def forward(self, input, target, pre_input, it):
        pre_target = F.softmax(pre_input, dim=1)
        _, pre_target = pre_target.max(dim=1, keepdim=True)
        pre_target = pre_target.squeeze(dim=0)

        ds = self.dice_score(target, pre_target)
        
        if it < self.start_warm:
            loss = ds*F.cross_entropy(input, target) + (1-ds)*F.cross_entropy(input, pre_target)
            return loss, 1.0

        raw_loss = ds*F.cross_entropy(input, target, reduction='none').view(-1) + (1-ds)*F.cross_entropy(input, pre_target, reduction='none').view(-1)        
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p

class IntersectionCE(nn.Module):
    def __init__(self, start_warm, end_warm, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p
    
    def forward(self, input, target, pre_input, it, bi, ti):
        pre_target = F.softmax(pre_input, dim=1)
        _, pre_target = pre_target.max(dim=1, keepdim=True)
        pre_target = pre_target.squeeze(dim=0)
        
        inter_target = torch.zeros_like(pre_target)
        inter_target[pre_target != 0] = 1

        inter_target = inter_target * target
        
        if it < self.start_warm:
            return F.cross_entropy(input, inter_target), 1.0

        raw_loss = F.cross_entropy(input, inter_target, reduction='none').view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p

class ReveseCE(torch.nn.Module):
    def __init__(self, alpha=0.1, beta=1.0):
        super(ReveseCE, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.cross_entropy = torch.nn.CrossEntropyLoss()
    
    def forward(self, pred, targets):

        num_classes = pred.shape[1]
    
        # softmax 적용
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        
        targets = torch.nn.functional.one_hot(targets, num_classes=2).float().to(self.device)
        targets = targets.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]
        targets = torch.clamp(targets, min=1e-4, max=1.0)
        
        new_targets = torch.zeros_like(pred)
        new_targets[:, 0, :, :] = targets[:, 0, :, :]
        if num_classes > 1:
            for i in range(1, num_classes):
                new_targets[:, i, :, :] = targets[:, 1, :, :]
        
        rce_loss = -torch.sum(pred * torch.log(new_targets), dim=1)
        
        return rce_loss

class UnionCE(nn.Module):
    def __init__(self, start_warm, end_warm, top_p=0.15):
        super().__init__()

        self.start_warm = start_warm
        self.end_warm = end_warm
        self.top_p = top_p
        self.rceloss = ReveseCE()
    
    def forward(self, input, target, pre_input, it, bi, ti):
        pre_target = F.softmax(pre_input, dim=1)
        _, pre_target = pre_target.max(dim=1, keepdim=True)
        pre_target = pre_target.squeeze(dim=0)
        
        pre_target_mask = torch.where(pre_target != 0, torch.ones_like(pre_target), torch.zeros_like(pre_target))
        target_mask = torch.where(target != 0, torch.ones_like(target), torch.zeros_like(target))
        uni_target = torch.max(pre_target_mask, target_mask)
        
        if it < self.start_warm:
            return self.rceloss(input, uni_target).mean(), 1.0

        raw_loss = self.rceloss(input, uni_target).view(-1)
        num_pixels = raw_loss.numel()

        if it > self.end_warm:
            this_p = self.top_p
        else:
            this_p = self.top_p + (1-self.top_p)*((self.end_warm-it)/(self.end_warm-self.start_warm))
        loss, _ = torch.topk(raw_loss, int(num_pixels * this_p), sorted=False)
        return loss.mean(), this_p

class LossComputer:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bce = BootstrappedCE(config['start_warm'], config['end_warm'])
        self.dice = DiceCE(config['start_warm'], config['end_warm'])
        self.inter = IntersectionCE(config['start_warm'], config['end_warm'])
        self.uni = UnionCE(config['start_warm'], config['end_warm'])

    def compute(self, data, num_objects, it):
        config = Configuration()
        config.parse()
        losses = defaultdict(int)

        b, t = data['rgb'].shape[:2] 

        set_loss = path.expanduser(config['set_loss'])        
        dice_rate = float(path.expanduser(str(config['dice_rate'])))
        inter_rate = float(path.expanduser(str(config['inter_rate'])))
        uni_rate = float(path.expanduser(str(config['uni_rate'])))
        
        losses['total_loss'] = 0
        
        for ti in range(1, t):
            for bi in range(b):
                if 'XMem' in set_loss:
                    loss, p = self.bce(data[f'logits_{ti}'][bi:bi+1, :num_objects[bi]+1], data['cls_gt'][bi:bi+1,ti,0], it)
                    losses['p'] += p / b / (t-1)
                    losses[f'ce_loss_{ti}'] += loss / b
                    losses['total_loss'] += losses['ce_loss_%d'%ti]
                    
                    losses[f'dice_loss_{ti}'] = dice_loss(data[f'masks_{ti}'], data['cls_gt'][:,ti,0])
                    losses['total_loss'] += losses[f'dice_loss_{ti}']
                
                if 'SMART' in set_loss:
                    # Cross-entropy Loss
                    loss, p = self.bce(data[f'logits_{ti}'][bi:bi+1, :num_objects[bi]+1], data['cls_gt'][bi:bi+1,ti,0], it)
                    losses['p'] += p / b / (t-1)
                    losses[f'ce_loss_{ti}'] += loss / b
                    losses['total_loss'] += losses['ce_loss_%d'%ti]
                    
                    # Reliable Loss
                    d_loss, p = self.dice(data[f'logits_{ti}'][bi:bi+1, :num_objects[bi]+1], data['cls_gt'][bi:bi+1,ti,0],
                                          data[f'pre_logits_{ti}'][bi:bi+1, :num_objects[bi]+1], it)
                    losses[f'DI_loss_{ti}'] += (dice_rate*d_loss) / b
                    losses['total_loss'] += losses[f'DI_loss_%d'%ti]

                    # Intersection Loss in Spaitial loss
                    i_loss, p = self.inter(data[f'logits_{ti}'][bi:bi+1, :num_objects[bi]+1], data['cls_gt'][bi:bi+1,ti,0],
                                          data[f'pre_logits_{ti}'][bi:bi+1, :num_objects[bi]+1], it, bi, ti)
                    losses[f'IN_loss_{ti}'] += (inter_rate*i_loss) / b
                    losses['total_loss'] += losses[f'IN_loss_%d'%ti]

                    # Union Loss in Spaitial loss
                    u_loss, p = self.uni(data[f'logits_{ti}'][bi:bi+1, :num_objects[bi]+1], data['cls_gt'][bi:bi+1,ti,0],
                                          data[f'pre_logits_{ti}'][bi:bi+1, :num_objects[bi]+1], it, bi, ti)
                    losses[f'UNI_loss_{ti}'] += (uni_rate*u_loss) / b
                    losses['total_loss'] += losses[f'UNI_loss_%d'%ti]
                
        return losses