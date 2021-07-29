import torch
import numpy as np
import torch.nn as nn
from models.modules.functional import *

class Quantizer(nn.Module):
    def __init__(self, latent_distribution, params, quantization, shape='global', window_size=None, n_windows=None):
        super().__init__()
        self.latent_distribution = latent_distribution
        
        if self.latent_distribution == 'Uniform':
            if quantization['codebook_size'] is None: raise Exception('[ERROR] Undefined number of embeddings.')
            self.num_embeddings = quantization['codebook_size']
            self.embedding_dim = params['layer0']['size']
            self.codebook = nn.Embedding(self.num_embeddings, self.embedding_dim)
            
        self.shape = shape
        ## Define the depth of the network.
        depth = len(params.keys()) - 1
        ## If shape is not global, define window size and number of windows.
        if window_size is not None:
            self.window_size = window_size
            if self.shape != 'global':
                self.n_windows = int(np.floor(params[f'layer{depth}']['size'] / self.window_size))
                
        elif (n_windows is not None) and (self.shape == 'hybrid'):
            self.n_windows = n_windows
            if not ((self.n_windows & (self.n_windows - 1) == 0) and (self.n_windows != 0)):
                raise Exception('[ERROR] The number of windows is not a power of 2!')
            if np.log2(self.n_windows) != depth:
                raise Exception(f'[ERROR] The number of defined layers ({depth}) does not match ' +
                                'the number of required layers ({np.log2(self.n_windows)}).')  
            self.window_size = int(np.floor(params['layer0']['size'] / self.n_windows))
        else: raise Exception('Missing window_size and n_windows')
            
        ## For slitting into windows the bottleneck
        self.split_size = params['layer0']['size']
        
    def _return_code(self, ze):
        if ze.shape[-1] != self.codebook.weight.shape[-1]:
            raise RuntimeError(
                f'[Error] Invalid argument: ze.shape[-1] ({ze.shape[-1]}) must \
                be equal to self.codebook.weight.shape[-1] ({self.codebook.weight.shape[-1]})'
            )
        sq_norm = (torch.sum(ze**2, dim = -1, keepdim = True) 
                + torch.sum(self.codebook.weight**2, dim = 1)
                - 2 * torch.matmul(ze, self.codebook.weight.t()))
        _, argmin = sq_norm.min(-1)
        zq = self.codebook.weight.index_select(0, argmin.view(-1)).view(ze.shape)
        return argmin, zq
         
    def forward(self, ze):
        ## In Multi-Bernoulli LS, the quantizer
        ## uses a threshold to binarize the 
        ## latent vectors.
        if self.latent_distribution == 'Multi-Bernoulli':
            zq = torch.ones(ze.shape).cuda()
            zq[ze < 0] = -1
        ## In Uniform LS, the quantizer
        ## computes the distances to the
        ## codebook vectors and takes the argmin.
        elif self.latent_distribution == 'Uniform':
            if self.shape == 'window-based':
                indices, zq = [], []
                for w in range(self.n_windows):
                    ze_windowed = ze[..., w * self.split_size: (w + 1) * self.split_size]
                    idx_windowed, zq_windowed = self._return_code(ze_windowed)
                    indices.append(idx_windowed.unsqueeze(1))
                    zq.append(zq_windowed)
                indices = torch.cat(indices, axis=1)
                zq = torch.cat(zq, axis=-1)
            else: indices, zq = self._return_code(ze)
            
            # The VQ objective uses the l2 error to move the embedding vectors 
            # ei towards the encoder outputs ze(x)
            vq_e_loss = torch.mean((zq - ze.detach()) ** 2)
            # Commitment loss
            vq_commit_loss = torch.mean((zq.detach() - ze)**2) 

            probs = torch.zeros(self.num_embeddings)
            unique, counts = np.unique(indices.cpu().detach().numpy(), return_counts=True)
            for i, c in zip(unique, counts): probs[i] = c.astype(float)/10
            entropy = torch.sum(probs * torch.log(probs + 1e-10))
            
            return indices, zq, vq_e_loss, vq_commit_loss, entropy
            
        else: return zq
    
    def backward(self, grad_zq):
        ## Clone decoder gradients to encoder.
        grad_ze = grad_zq.clone()
        return grad_ze, None