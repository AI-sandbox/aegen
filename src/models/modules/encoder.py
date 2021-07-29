import torch
import numpy as np
import torch.nn as nn
from functional import *

class Encoder(nn.Module):
    def __init__(self, latent_distribution, params, num_classes=None, shape='global', window_size=None, n_windows=None, window_cloning=None, heads=1):
        super().__init__()
        ## Define latent distribution.
        self.latent_distribution = latent_distribution
        
        ## Define the shape of the Encoder.
        self.shape = shape
        
        ## Define the depth of the network.
        depth = len(params.keys()) - 1
        self.depth = depth
        
        ## If shape is not global, define window size and number of windows.
        if window_size is not None:
            self.window_size = window_size
            if shape != 'global':
                self.n_windows = int(np.floor(params['layer0']['size'] / window_size))
        elif (n_windows is not None) and (self.shape == 'hybrid'):
            self.n_windows = n_windows
            if not ((self.n_windows & (self.n_windows - 1) == 0) and (self.n_windows != 0)):
                raise Exception('[ERROR] The number of windows is not a power of 2!')
            if np.log2(self.n_windows) != depth:
                raise Exception(f'[ERROR] The number of defined layers ({depth}) does not match ' +
                                'the number of required layers ({np.log2(self.n_windows)}).')  
            self.window_size = int(np.floor(params['layer0']['size'] / self.n_windows))
        else: raise Exception('Missing window_size and n_windows')
        self.window_cloning = window_cloning
        if (self.shape == 'window-based') and (self.window_cloning is None):
            raise Exception('[ERROR] Window cloning is not defined using window-based shape.')
            
        ## Define multi-head if desires.
        self.heads = heads
        if (self.latent_distribution != 'Uniform') and self.heads > 1:
            print('[WARNING] Multi-head for non VQ. Ignoring multi-head requirement.')
            self.heads = 1

        ## Check if activations are correct.                
        if (self.latent_distribution == 'Multi-Bernoulli') and (params[f'layer{depth - 1}']['activation'] != 'Tanh'):
            print(params[f'layer{depth - 2}']['activation'])
            raise Exception('[ERROR] Missing tanh activation!')
        if (self.latent_distribution == 'Gaussian') and (params[f'layer{depth - 1}']['activation'] != None):
            print('[WARNING] Activation for Mu and Logvar features is not the identity.')
            
        ## Shape global modules definitions.
        if self.shape == 'global':
            modules = []

            for i in range(depth - 1 if self.latent_distribution == 'Gaussian' else depth):
                modules.append(
                    FullyConnected(
                        ## Only can condition input (layer0).
                        input = (
                            params[f'layer{i}']['size'] if num_classes is None else (params[f'layer{i}']['size'] + num_classes)
                        ) if i == 0 else params[f'layer{i}']['size'],
                        output     = params[f'layer{i + 1}']['size'],
                        dropout    = params[f'layer{i}']['dropout'],
                        normalize  = params[f'layer{i}']['normalize'],
                        activation = params[f'layer{i}']['activation'],
                    )
                )
            self.FCs = nn.Sequential(*modules)
            if self.latent_distribution == 'Gaussian':
                self.FCmu = FullyConnected(
                    input      = params[f'layer{depth - 1}']['size'],
                    output     = params[f'layer{depth}']['size'],
                    dropout    = params[f'layer{depth - 1}']['dropout'],
                    normalize  = params[f'layer{depth - 1}']['normalize'],
                    activation = params[f'layer{depth - 1}']['activation'],
                )
                self.FClogvar = FullyConnected(
                    input      = params[f'layer{depth - 1}']['size'],
                    output     = params[f'layer{depth}']['size'],
                    dropout    = params[f'layer{depth - 1}']['dropout'],
                    normalize  = params[f'layer{depth - 1}']['normalize'],
                    activation = params[f'layer{depth - 1}']['activation'],
                )
        ## Shape window-based (independent) modules definitions
        elif self.shape == 'window-based': ## TODO: Implement conditioned window-based
            modules = {}
            ## For each layer and for each window we define a FC matrix.
            for i in range(depth):
                if not self.window_cloning:
                    for w in range(self.n_windows):
                        ## If first layer0, we split the input size.
                        ## Otherwise, we use the size defined in params.
                        if i == 0:
                            wsize = self.window_size
                            if num_classes is not None:
                                wsize += num_classes 
                            if w == self.n_windows - 1:
                                wsize += (params['layer0']['size'] % self.window_size)
                        else: wsize = params[f'layer{i}']['size']
                        ## Append modules;
                        if self.latent_distribution == 'Gaussian':
                            modules[f'layer{i}_win{w}_mu'] = FullyConnected(
                                input      = wsize,
                                output     = params[f'layer{i + 1}']['size'],
                                dropout    = params[f'layer{i}']['dropout'],
                                normalize  = params[f'layer{i}']['normalize'],
                                activation = params[f'layer{i}']['activation'],
                            )
                            modules[f'layer{i}_win{w}_logvar'] = FullyConnected(
                                input      = wsize,
                                output     = params[f'layer{i + 1}']['size'],
                                dropout    = params[f'layer{i}']['dropout'],
                                normalize  = params[f'layer{i}']['normalize'],
                                activation = params[f'layer{i}']['activation'],
                            )
                        else:
                            ## If multi-head, define heads in the last layer.
                            if (self.heads > 1) and (i == depth - 1):
                                for h in range(self.heads):
                                    print(f'Adding layer{i}_win{w}_head{h}...')
                                    modules[f'layer{i}_win{w}_head{h}'] = FullyConnected(
                                        input      = wsize,
                                        output     = params[f'layer{i + 1}']['size'],
                                        dropout    = params[f'layer{i}']['dropout'],
                                        normalize  = params[f'layer{i}']['normalize'],
                                        activation = params[f'layer{i}']['activation'],
                                    )
                            else:
                                modules[f'layer{i}_win{w}'] = FullyConnected(
                                    input      = wsize,
                                    output     = params[f'layer{i + 1}']['size'],
                                    dropout    = params[f'layer{i}']['dropout'],
                                    normalize  = params[f'layer{i}']['normalize'],
                                    activation = params[f'layer{i}']['activation'],
                                )
                else: ## Use cloning.
                    if self.latent_distribution == 'Gaussian': raise Exception('Not implemented.')
                    else:
                        wsize = self.window_size
                        if num_classes is not None:
                            wsize += num_classes 
                        modules[f'layer{i}'] = FullyConnected(
                                input      = wsize,
                                output     = params[f'layer{i + 1}']['size'],
                                dropout    = params[f'layer{i}']['dropout'],
                                normalize  = params[f'layer{i}']['normalize'],
                                activation = params[f'layer{i}']['activation'],
                            )
                    
            ## We create funnels for each window.
            if not self.window_cloning:
                if self.latent_distribution == 'Gaussian':
                    self.funnel_mu, self.funnel_logvar = nn.ModuleList(), nn.ModuleList()
                    for w in range(self.n_windows):
                        self.funnel_mu.append(nn.Sequential(*[modules[f'layer{i}_win{w}_mu'] for i in range(depth)]))
                        self.funnel_logvar.append(nn.Sequential(*[modules[f'layer{i}_win{w}_logvar'] for i in range(depth)]))
                else:
                    self.funnel = nn.ModuleList()
                    for w in range(self.n_windows):
                        funnel_depth = depth - 1 if self.heads > 1 else depth
                        self.funnel.append(nn.Sequential(*[modules[f'layer{i}_win{w}'] for i in range(funnel_depth)]))
                    if self.heads > 1: 
                        self.endheads = nn.ModuleList()
                        for w in range(self.n_windows):
                            for h in range(self.heads):
                                self.endheads.append(modules[f'layer{depth - 1}_win{w}_head{h}'])
            else: ## Use cloning.
                if self.latent_distribution == 'Gaussian': raise Exception('Not implemented.')
                else:
                    self.funnel = nn.Sequential(*[modules[f'layer{i}'] for i in range(depth)])

        ## Shape window-based (dependent) modules definitions
        elif self.shape == 'hybrid':
            self.params = params
            self.dmodules = nn.ModuleDict()
            ## Define a binary tree of layers using a dict
            nw = self.n_windows
            for i in range(depth):
                aux = nn.ModuleList()
                ## If first layer0, we split the input size.
                ## Otherwise, we use the size defined in params.
                for w in range(nw):
                    if i == 0:
                        wsize = self.window_size
                        if num_classes is not None:
                            wsize += num_classes
                        if w == self.n_windows - 1:
                            wsize += (params['layer0']['size'] % self.window_size)
                    else: wsize = params[f'layer{i}']['size']
                    
                    if (self.latent_distribution == 'Gaussian') and (i == depth - 1) and (nw == 2): 
                        layer_mu = FullyConnected( ## mu
                                input      = wsize,
                                output     = params[f'layer{i + 1}']['size'],
                                dropout    = params[f'layer{i}']['dropout'],
                                normalize  = params[f'layer{i}']['normalize'],
                                activation = params[f'layer{i}']['activation'],
                            )
                        layer_logvar = FullyConnected( ## logvar
                                input      = wsize,
                                output     = params[f'layer{i + 1}']['size'],
                                dropout    = params[f'layer{i}']['dropout'],
                                normalize  = params[f'layer{i}']['normalize'],
                                activation = params[f'layer{i}']['activation'],
                            )
                    else: 
                        aux.append( ## hidden
                            FullyConnected(
                                input      = wsize,
                                output     = params[f'layer{i + 1}']['size'],
                                dropout    = params[f'layer{i}']['dropout'],
                                normalize  = params[f'layer{i}']['normalize'],
                                activation = params[f'layer{i}']['activation'],
                            )
                        )
                if (self.latent_distribution == 'Gaussian') and (i == depth - 1) and (nw == 2): 
                    self.dmodules.update({
                        f'group{i}': nn.ModuleDict({
                            'mu': nn.ModuleList([layer_mu] * 2), 
                            'logvar': nn.ModuleList([layer_logvar] * 2)
                        })
                    })  
                else:
                    self.dmodules.update({f'group{i}': aux})      
                nw //= 2
        else: raise Exception('Unknown shape.')
    
    def _binary_tree_path(window):
        ws, curr_w = [], window
        for group in self.modules.keys():
            curr_w //= 2
            ws.append(self.modules[group][curr_w])
        return ws

    def forward(self, x, c=None):
        if self.shape == 'global':
            ## Append the conditioning if desired.
            ## If the latent space is discrete, we are done.
            ## If not, we must parametrize.
            o = self.FCs(x if c is None else torch.cat([x, c], -1))
            if self.latent_distribution == 'Gaussian':
                o_mu = self.FCmu(o)
                o_var = self.FClogvar(o)
                return o_mu, o_var
            else: return o
        elif self.shape == 'window-based':
            if self.latent_distribution == 'Gaussian':
                os_mu, os_logvar = [], [] ## Stores the outputs of each window.
                for w in range(self.n_windows):
                    if w == self.n_windows - 1:
                        x_windowed = x[..., w * self.window_size:]
                    else:
                        x_windowed = x[..., w * self.window_size:(w + 1) * self.window_size]
                    ## Append class if needed
                    if c is not None:
                        x_windowed = torch.cat([x_windowed, c], -1)
                    ## Process each window with the corresponding funnel.
                    os_mu.append(self.funnel_mu[w](x_windowed))
                    os_logvar.append(self.funnel_logvar[w](x_windowed))
                ## Return means and logvars per window
                return os_mu, os_logvar
            else:
                os = [] ## Stores the outputs of each window.
                for w in range(self.n_windows):
                    if w == self.n_windows - 1:
                        x_windowed = x[..., w * self.window_size:]
                    else:
                        x_windowed = x[..., w * self.window_size:(w + 1) * self.window_size]
                    ## Append class if needed
                    if c is not None:
                        x_windowed = torch.cat([x_windowed, c], -1)
                    ## Process each window with the corresponding funnel.
                    if self.window_cloning: os.append(self.funnel(x_windowed))
                    else: 
                        ## If multi-head:
                        if self.heads > 1:
                            pre_head_windowed = self.funnel[w](x_windowed)
                            pos_head_windowed = []
                            for h in range(self.heads):
                                pos_head_windowed.append(self.endheads[w + h](pre_head_windowed).unsqueeze(1))
                            pos_head_windowed = torch.cat(pos_head_windowed, axis=1)
                            os.append(pos_head_windowed)
                        else: os.append(self.funnel[w](x_windowed))
                ## Concat all outputs into a single o vector.
                o = torch.cat(os, axis=-1)
            return o
        elif self.shape == 'hybrid':
            aux = [] ## Stores the outputs of each window.
            ws = self.n_windows
            for i, group in enumerate(self.dmodules.keys()):
                for w_id in range(0,ws,2):
                    ## If input layer, resize last window if needed
                    if group == 'group0':
                        if w_id == ws - 2:
                            w1 = x[..., w_id * self.window_size:(w_id + 1) * self.window_size]
                            w2 = x[..., (w_id + 1) * self.window_size:]
                        else: 
                            w1 = x[..., w_id * self.window_size:(w_id + 1) * self.window_size]
                            w2 = x[..., (w_id + 1) * self.window_size:(w_id + 2) * self.window_size]
                        ## Append class if needed
                        if c is not None:
                            w1 = torch.cat([w1, c], -1)
                            w2 = torch.cat([w2, c], -1)
                    else: ## Recompute window sizes (defined in params)
                        wsize = self.params[f'layer{i}']['size']
                        w1 = x[..., w_id * wsize:(w_id + 1) * wsize]
                        w2 = x[..., (w_id + 1) * wsize:(w_id + 2) * wsize]
                    ## Add nodes in binary tree
                    if (self.latent_distribution == 'Gaussian') and (group == f'group{self.depth - 1}'):
                        node_mu = self.dmodules[group]['mu'][0](w1) + self.dmodules[group]['mu'][1](w2)
                        node_logvar = self.dmodules[group]['logvar'][0](w1) + self.dmodules[group]['logvar'][1](w2)
                        return node_mu, node_logvar
                    else:
                        node = self.dmodules[group][w_id](w1) + self.dmodules[group][w_id + 1](w2)
                    aux.append(node)
                ## Reduce number of windows on next layer
                ws //= 2
                x = torch.cat(aux, axis=-1)
                aux = []
            o = x
            return o
        else: raise Exception('Unknown shape.')