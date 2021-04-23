import torch
import numpy as np
import torch.nn.functional as F


def VAEloss(x, o, mu, logvar, beta=1):

    loss = F.binary_cross_entropy(o, x, reduction='sum')
    KL_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
 
    return loss +  beta * KL_divergence, loss, KL_divergence

def L1loss(x, o, partial=False, proportion=False):
    loss = F.l1_loss((o > 0.5).float(), x, reduction='sum')
    
    if partial:
        x, o = x.cpu(), o.cpu()
        one_hot_neg = (x - (o.detach() > 0.5).float()).flatten()
        loss_zeros = len(np.where(one_hot_neg == -1)[0])
        loss_ones = len(np.where(one_hot_neg == 1)[0])
        ones = len(np.where(x == 1)[0])
        compression_ratio = loss / ones

        if proportion: 
            zeros = len(np.where(x == 0)[0])
            
            acc_total = loss / (x.shape[0] * x.shape[1]) * 100
            acc_zeros = loss_zeros / zeros * 100
            acc_ones = loss_ones / ones * 100
            
            return acc_total, acc_zeros, acc_ones, compression_ratio
        
        return loss, loss_zeros, loss_ones, compression_ratio
    
    return loss if not proportion else loss / (x.shape[0] * x.shape[1]) * 100
