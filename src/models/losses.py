import torch
import torch.nn.functional as F

def VAEloss(x, o, mu, logvar, beta=1):

    loss = F.binary_cross_entropy(o, x, reduction='sum')
    KL_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
 
    return loss +  beta * KL_divergence, loss