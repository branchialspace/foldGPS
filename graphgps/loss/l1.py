import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss


@register_loss('l1_losses')
def l1_losses(pred, true):
    if cfg.model.loss_fun == 'l1': # L1 with false negative/positive penalty
        # L1 loss
        l1_loss = nn.L1Loss()
        basic_loss = l1_loss(pred, true)
        # Penalty terms
        false_negatives = (true * (1 - pred)).sum()
        false_positives = ((1 - true) * pred).sum()
        # Weighted penalty
        false_positive_penalty = 1.0
        false_negative_penalty = 1.0 
        penalty = (false_negative_penalty * false_negatives) + (false_positive_penalty * false_positives)
        # Total loss
        loss = basic_loss + penalty
        return loss, pred
        
    elif cfg.model.loss_fun == 'ogl1':
        l1_loss = nn.L1Loss()
        loss = l1_loss(pred, true)
        return loss, pred
        
    elif cfg.model.loss_fun == 'smoothl1':
        l1_loss = nn.SmoothL1Loss()
        loss = l1_loss(pred, true)
        return loss, pred
