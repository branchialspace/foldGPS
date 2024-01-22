import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss

def weighted_MSE(predictions, targets, epsilon=1e-6, weight_factor=1000):
    # Calculate the weights based on targets
    weights = (targets + epsilon) * weight_factor
    # Weighted MSE
    weighted_mse = ((predictions - targets)**2) * weights
    loss = torch.mean(weighted_mse)

    return loss


@register_loss('l1_losses')

def l1_losses(pred, true):
    if cfg.model.loss_fun == 'l1':
        pred = pred.view(true.size())
        loss = weighted_MSE(pred, true)

        return loss, pred
        
    elif cfg.model.loss_fun == 'l1_fnp': # L1 with false negative/positive penalty
        pred = pred.view(true.size())
        # L1 loss
        l1_loss = nn.L1Loss()
        basic_loss = l1_loss(pred, true)
    
        # Identifying non-zero (positive) and zero (negative) indices
        positive_preds_indices = pred.nonzero(as_tuple=True)
        positive_true_indices = true.nonzero(as_tuple=True)
        negative_true_indices = (true == 0).nonzero(as_tuple=True)
    
        # False Negatives: true is positive but pred is not positive
        false_negatives = pred[positive_true_indices] == 0
        false_negatives_count = false_negatives.sum()
    
        # False Positives: pred is positive but true is not positive
        false_positives = true[positive_preds_indices] == 0
        false_positives_count = false_positives.sum()
    
        # Weighted penalty
        false_positive_penalty = 0.1
        false_negative_penalty = 0.1
        penalty = (false_negative_penalty * false_negatives_count) + (false_positive_penalty * false_positives_count)
    
        # Total loss
        loss = basic_loss + penalty
        return loss, pred
        
    elif cfg.model.loss_fun == 'l1_og':
        l1_loss = nn.L1Loss()
        loss = l1_loss(pred, true)
        return loss, pred
        
    elif cfg.model.loss_fun == 'smoothl1':
        l1_loss = nn.SmoothL1Loss()
        loss = l1_loss(pred, true)
        return loss, pred
