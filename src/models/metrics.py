import torch
import numpy as np
import blosc2 as blosc

def create_metrics_dict(metrics, prefix='train'):
    metrics_dict = {}
    if (prefix == 'train') or (prefix == 'tr'):
        prefix = 'tr' 
    elif (prefix == 'valid') or (prefix == 'vd'):
        prefix = 'vd'
    elif prefix == 'aux':
        prefix = 'aux'
    else: raise Exception('Prefix not valid.')
    for kmetric, meta in metrics.items():
        if callable(kmetric):
            for name in meta['outputs']:
                metrics_dict[f'{prefix}_{name}'] = []
        else:
            for p in meta['params']:
                metrics_dict[f'{prefix}_{p}_{kmetric}'] = []
    return metrics_dict

def metacompressor_metric(x, mu, xhat, distribution=None, algorithm=None):
    if distribution is None: raise Exception('Latent space distribution not defined.')
    if algorithm is None: raise Exception('Compression algorithm not defined.')
    
    if isinstance(x, np.ndarray): x = x.astype(bool)
    elif isinstance(x, torch.Tensor): x = x.cpu().detach().numpy().astype(bool)
    
    if isinstance(mu, np.ndarray): 
        if distribution == 'Gaussian':
            mu = mu.astype(float)
        else:
            mu = (mu + 1) / 2
            mu = mu.astype(bool)
    elif isinstance(mu, torch.Tensor): 
        if distribution == 'Gaussian':
            mu = mu.cpu().detach().numpy().astype(float)
        else:
            mu = mu.cpu().detach().numpy()
            mu = (mu + 1) / 2
            mu = mu.astype(bool)

    if isinstance(xhat, np.ndarray): xhat = xhat.astype(bool)
    elif isinstance(xhat, torch.Tensor): xhat = xhat.cpu().detach().numpy().astype(bool)    
    
    xbin  = x.tostring()
    mubin = mu.tostring()
    xhatbin  = xhat.tostring()
    
    mucom = blosc.compress(
        mubin, 
        typesize=4 if distribution == 'Gaussian' else 1, 
        cname=algorithm,
        shuffle=blosc.BITSHUFFLE
    )
    
    xhatcom = blosc.compress(
        xhatbin, 
        typesize=1, 
        cname=algorithm,
        shuffle=blosc.BITSHUFFLE
    )
    
    return len(xbin)/(len(mucom) + len(xhatcom))

def metacompressor_metric_compressed(x, mu, xhat, distribution=None, algorithm=None):
    if distribution is None: raise Exception('Latent space distribution not defined.')
    if algorithm is None: raise Exception('Compression algorithm not defined.')
    
    if isinstance(x, np.ndarray): x = x.astype(bool)
    elif isinstance(x, torch.Tensor): x = x.cpu().detach().numpy().astype(bool)
    
    if isinstance(mu, np.ndarray): 
        if distribution == 'Gaussian':
            mu = mu.astype(float)
        else:
            mu = (mu + 1) / 2
            mu = mu.astype(bool)
    elif isinstance(mu, torch.Tensor): 
        if distribution == 'Gaussian':
            mu = mu.cpu().detach().numpy().astype(float)
        else:
            mu = mu.cpu().detach().numpy()
            mu = (mu + 1) / 2
            mu = mu.astype(bool)
       
    if isinstance(xhat, np.ndarray): xhat = xhat.astype(bool)
    elif isinstance(xhat, torch.Tensor): xhat = xhat.cpu().detach().numpy().astype(bool)    
    
    xbin  = x.tostring()
    mubin = mu.tostring()
    xhatbin  = xhat.tostring()
    
    xcom = blosc.compress(
        xbin, 
        typesize=1, 
        cname=algorithm,
        shuffle=blosc.BITSHUFFLE
    )
    
    mucom = blosc.compress(
        mubin, 
        typesize=4 if distribution == 'Gaussian' else 1, 
        cname=algorithm,
        shuffle=blosc.BITSHUFFLE
    )
    
    xhatcom = blosc.compress(
        xhatbin, 
        typesize=1, 
        cname=algorithm,
        shuffle=blosc.BITSHUFFLE
    )
    
    return len(xcom)/(len(mucom) + len(xhatcom))
    
    
    
    
    
    
    
    
    