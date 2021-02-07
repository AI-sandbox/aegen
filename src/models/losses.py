import torch
import numpy as np
import torch.nn.functional as F


def VAEloss(x, o, mu, logvar, beta=1):

    loss = F.binary_cross_entropy(o, x, reduction='sum')
    KL_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
 
    return loss +  beta * KL_divergence, loss, KL_divergence

def L1loss(x, o, partial=False):
    loss = F.l1_loss((o > 0.5).float(), x, reduction='sum')

    if partial:

        one_hot_neg = (x - (o.detach() > 0.5).float()).flatten()
        loss_zeros = len(np.where(one_hot_neg == -1)[0])
        loss_ones = len(np.where(one_hot_neg == 1)[0])
        
        return loss, loss_zeros, loss_ones
    
    return loss