import torch
import numpy as np
import blosc2 as blosc
import ctypes

## Function to create a metrics dictionary.
## Specifies a prefix for the metrics according
## the split set on which they have been obtained.
def create_metrics_dict(metrics, prefix='train'):
    metrics_dict = {}
    if (prefix == 'train') or (prefix == 'tr'):
        prefix = 'tr' 
    elif (prefix == 'valid') or (prefix == 'vd'):
        prefix = 'vd'
    elif prefix == 'aux':
        prefix = 'aux'
    else: raise Exception('[ERROR] Prefix not valid.')
    for kmetric, meta in metrics.items():
        if callable(kmetric):
            for name in meta['outputs']:
                metrics_dict[f'{prefix}_{name}'] = []
        else:
            for p in meta['params']:
                metrics_dict[f'{prefix}_{p}_{kmetric}'] = []
    return metrics_dict

## Formats correctly the input tensors.
def to_numpy(x, z, r, distribution=None):
        
    def _to_numpy(x):
        if isinstance(x, np.ndarray): return x
        elif isinstance(x, torch.Tensor): return x.cpu().detach().numpy()
        
    def _make01(z):
        z = (z + 1) / 2
        return z
    
    x = _to_numpy(x).astype(bool)
    if distribution == 'Gaussian': z = _to_numpy(z).astype(float)
    if distribution == 'Uniform': z = _to_numpy(z).astype(np.dtype('B'))
    else: z = _make01(_to_numpy(z)).astype(bool)
    r =  _to_numpy(r).astype(bool) 
    return x, z, r

## Transforms x, z, r to binary representation, for compression.
## Takes into account the datatype of the object.
## Returns x.bin, z.bin and r.bin.   
def to_binary(x, z, r):
    return x.tostring(), z.tostring(), r.tostring()

## Return elementary datatype size in bytes.
def elem_tsize(x):
    if x.dtype == 'bool': tsize = ctypes.sizeof(ctypes.c_bool)
    elif x.dtype == 'float': tsize = ctypes.sizeof(ctypes.c_float)
    elif x.dtype == 'uint8': tsize = ctypes.sizeof(ctypes.c_ubyte)
    elif x.dtype == 'int': tsize = ctypes.sizeof(ctypes.c_int)
    else: raise Exception('[ERROR] Unknown datatype.')
    return tsize

## Compressed length in bytes.
def clen(x, typesize, algorithm, shuffle=blosc.BITSHUFFLE):
    return len(blosc.compress(x, typesize=typesize, cname=algorithm, shuffle=shuffle))

def metacompressor_metric(x, z, r, distribution=None, algorithm=None, shuffle=blosc.BITSHUFFLE, partial=None, kind='cratio'):
    if distribution is None: raise Exception('[ERROR] Latent space distribution not defined.')
    if algorithm is None: raise Exception('[ERROR] Compression algorithm not defined.')
    if (shuffle is not blosc.NOSHUFFLE) and (shuffle is not blosc.BITSHUFFLE): raise Exception('[ERROR] Unknown shuffle.')
    if (kind is not 'cratio') and (kind is not 'ccratio'): raise Exception('[ERROR] Unknown metacompressor metric kind.') 
    
    ## Unpack z:
    if distribution == 'Gaussian': z = z[0]
    elif distribution == 'Multi-Bernoulli': pass
    elif distribution == 'Uniform': z = z[0]
    else: raise Exception('[ERROR] Unpack operation failed.')
                
    x, z, r = to_numpy(x, z, r, distribution=distribution)
    xbin, zbin, recbin = to_binary(x, z, r)
    
    if kind == 'cratio':
        clen_zbin, clen_recbin = clen(zbin, elem_tsize(z), algorithm, shuffle=shuffle), clen(recbin, elem_tsize(r), algorithm, shuffle=shuffle)
        if partial is not None: raise Exception('[ERROR] Not implemented.')
        else: return len(xbin) / (clen_zbin + clen_recbin)
    elif kind == 'ccratio':
        clen_xbin, clen_zbin, clen_recbin = clen(xbin, elem_tsize(x), algorithm, shuffle=shuffle), clen(zbin, elem_tsize(z), algorithm, shuffle=shuffle), clen(recbin, elem_tsize(r), algorithm, shuffle=shuffle)
        if partial is None:        return clen_xbin / (clen_zbin + clen_recbin)
        if partial == 'embedding': return clen_xbin / clen_zbin
        if partial == 'residual':  return clen_xbin / clen_recbin
        if partial is not None:    raise Exception('[ERROR] Not implemented.')
    else: raise Exception('Unknown kind.')

## Ratio between (original x size) vs (compressed latent representation + compressed residual).
## Computes the compression ratio of the system.
## Above 1 is good.
def cratio_no_shuffle(x, z, r, distribution=None, algorithm=None):
    return metacompressor_metric(
        x, z, r, 
        distribution=distribution, 
        algorithm=algorithm, 
        shuffle=blosc.NOSHUFFLE, 
        partial=None, 
        kind='cratio'
    )

def cratio_bitshuffle(x, z, r, distribution=None, algorithm=None):
    return metacompressor_metric(
        x, z, r, 
        distribution=distribution, 
        algorithm=algorithm, 
        shuffle=blosc.BITSHUFFLE, 
        partial=None, 
        kind='cratio'
    )

## Ratio between (compressed x size) vs (compressed latent representation + compressed residual).
## Computes the improvement of the compression ratio over the compression algorithm.
## Above 1 is good.
def ccratio_no_shuffle(x, z, r, distribution=None, algorithm=None):
    return metacompressor_metric(
        x, z, r, 
        distribution=distribution, 
        algorithm=algorithm, 
        shuffle=blosc.NOSHUFFLE, 
        partial=None, 
        kind='ccratio'
    )

def ccratio_bitshuffle(x, z, r, distribution=None, algorithm=None):
    return metacompressor_metric(
        x, z, r, 
        distribution=distribution, 
        algorithm=algorithm, 
        shuffle=blosc.BITSHUFFLE, 
        partial=None, 
        kind='ccratio'
    )

def partial_embedding_ccratio_no_shuffle(x, z, r, distribution=None, algorithm=None):
    return metacompressor_metric(
        x, z, r, 
        distribution=distribution, 
        algorithm=algorithm, 
        shuffle=blosc.NOSHUFFLE, 
        partial='embedding', 
        kind='ccratio'
    )

def partial_residual_ccratio_no_shuffle(x, z, r, distribution=None, algorithm=None):
    return metacompressor_metric(
        x, z, r, 
        distribution=distribution, 
        algorithm=algorithm, 
        shuffle=blosc.NOSHUFFLE, 
        partial='residual', 
        kind='ccratio'
    )

def partial_embedding_ccratio_bitshuffle(x, z, r, distribution=None, algorithm=None):
    return metacompressor_metric(
        x, z, r, 
        distribution=distribution, 
        algorithm=algorithm, 
        shuffle=blosc.BITSHUFFLE, 
        partial='embedding', 
        kind='ccratio'
    )

def partial_residual_ccratio_bitshuffle(x, z, r, distribution=None, algorithm=None):
    return metacompressor_metric(
        x, z, r, 
        distribution=distribution, 
        algorithm=algorithm, 
        shuffle=blosc.BITSHUFFLE, 
        partial='residual', 
        kind='ccratio'
    )

## Computes L1 loss between input and rexonstruction
## to quantify the sparsity of the residual vector.
def residual_sparsity(x, xhat, batch_size=None):
    if batch_size is None: raise Exception('[ERROR] Batch size is mising.')
    x, xhat = x.float().cpu().detach(), xhat.float().cpu().detach()
    return [abs(x - xhat).sum().item() // batch_size]
    