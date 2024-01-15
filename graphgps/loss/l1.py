import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss


@register_loss('l1_losses')
def l1_losses(pred, true):
    if cfg.model.loss_fun == 'l1': # L1 Loss with non-zero count penalty
        # L1 Loss Component
        l1_loss = nn.L1Loss()
        loss_l1 = l1_loss(pred, true)
        # Non-zero Count Penalty
        non_zero_penalty_weight=1.0
        true_non_zero_count = torch.count_nonzero(true)
        pred_non_zero_count = torch.count_nonzero(pred)
        non_zero_count_penalty = abs(true_non_zero_count - pred_non_zero_count).float()
        # Combined Loss
        loss = loss_l1 + non_zero_penalty_weight * non_zero_count_penalty

        return loss, pred
        
    elif cfg.model.loss_fun == 'og_l1':
        l1_loss = nn.L1Loss()
        loss = l1_loss(pred, true)
        return loss, pred
        
    elif cfg.model.loss_fun == 'smoothl1':
        l1_loss = nn.SmoothL1Loss()
        loss = l1_loss(pred, true)
        return loss, pred
