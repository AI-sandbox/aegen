import torch
import numpy as np
import torch.nn.functional as F


def aeloss(x, o, mu, logvar=None, beta=1, backward=False):
    x, o, mu = x.float(), o.float(), mu.float()
    loss = F.binary_cross_entropy(o, x, reduction='sum')
    if logvar is not None:
        KL_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        if backward:
            return (loss +  beta * KL_divergence)
        return (loss +  beta * KL_divergence).item(), loss.item(), KL_divergence.item()
    else:
        if backward:
            return loss
        return loss.item(), loss.item(), 0

def L1loss(x, o, partial=True, proportion=True):
    x, o = x.float(), o.float()
    loss = F.l1_loss(o, x, reduction='sum')
    
    if partial:
        x, o = x.cpu(), o.cpu()
        one_hot_neg = (x - o.detach()).flatten()
        loss_zeros = len(np.where(one_hot_neg == -1)[0])
        loss_ones = len(np.where(one_hot_neg == 1)[0])
        ones = len(np.where(x == 1)[0])
        compression_ratio = loss / ones

        if proportion: 
            zeros = len(np.where(x == 0)[0])
            
            acc_total = loss / (x.shape[0] * x.shape[1]) * 100
            acc_zeros = loss_zeros / zeros * 100
            acc_ones = loss_ones / ones * 100
            
            return acc_total.item(), acc_zeros, acc_ones, compression_ratio
        
        return loss.item(), loss_zeros, loss_ones, compression_ratio
    
    return loss.item() if not proportion else loss.item() / (x.shape[0] * x.shape[1]) * 100
