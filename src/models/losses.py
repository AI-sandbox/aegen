import torch
import numpy as np
import torch.nn.functional as F


def aeloss(x, o, args, distribution, beta=1, backward=False, reduction='mean'):
    
    x, o = x.float(), o.float()
    loss = F.binary_cross_entropy(o, x, reduction=reduction)
    
    if distribution == 'Gaussian':
        mu, logvar = args
        mu, logvar = mu.float(), logvar.float()
        if backward: return (loss +  beta * KL_divergence)
        else:        return (loss +  beta * KL_divergence).item(), loss.item(), KL_divergence.item()
    
    elif (distribution == 'Multi-Bernoulli') or (distribution == 'Unknown'):
        if backward: return loss
        return loss.item(), loss.item(), 0
    
    elif distribution == 'Uniform':
        _, vq_e_loss, beta_vq_commit_loss, _, K = args
        if backward: return (loss + vq_e_loss + beta_vq_commit_loss)
        else:        return (loss + vq_e_loss + beta_vq_commit_loss).item(), loss.item(), np.log(K)

    else: raise Exception('[ERROR] Unknown distribution.')

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

## OPTIONAL: extra loss for training (optimizing). 
## Only available for window-based autoencoders.
## Enforces the embedding to be "similar" (not so variable)
## accross windows. Heuristic to compress better.
def varloss(z, backward=False, reduction='sum'):
    z = z.float()
    var_loss = z.var(axis=1).sum() 
    if reduction == 'mean': var_loss /= z.shape[0] 
    return var_loss if backward else [var_loss.item()]

## FOR VQ-Autoencoder ONLY.
def entropy(args):
    _, _, _, entropy, _ = args
    return [entropy]