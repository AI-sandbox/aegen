import torch
import numpy as np
import torch.nn as nn
from models.modules.functional import *
    
class Decoder(nn.Module):
    def __init__(self, params, num_classes=None, shape='global', window_size=None, n_windows=None, window_cloning=None, heads=1):
        super().__init__()
        ## Define the shape of the Decoder.
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
            self.window_size = int(np.floor(params[f'layer{depth}']['size'] / self.n_windows))
        else: raise Exception('Missing window_size and n_windows')
        self.window_cloning = window_cloning
        if (self.shape == 'window-based') and (self.window_cloning is None):
            raise Exception('[ERROR] Window cloning is not defined using window-based shape.')
            
        ## Define multi-head if desires.
        self.heads = heads
        
        ## Shape global modules definitions.
        if self.shape == 'global':
            modules = []
            for i in range(depth):
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
        ## Shape window-based (independent) modules definitions
        elif self.shape == 'window-based':
            modules = {}
            ## For each layer and for each window we define a FC matrix.
            for i in range(depth):
                if not self.window_cloning:
                    for w in range(self.n_windows):
                        ## If last layerD, we split the output size.
                        ## Otherwise, we use the size defined in params.
                        zsize = params[f'layer{i}']['size']
                        if (i == 0) and (num_classes is not None):
                            zsize += num_classes
                        if i == depth - 1:
                            wsize = self.window_size
                            if w == self.n_windows - 1:
                                wsize += (params[f'layer{depth}']['size'] % self.window_size)
                        else: wsize = params[f'layer{i + 1}']['size']
                        ## Append modules;
                        modules[f'layer{i}_win{w}'] = FullyConnected(
                            input      = zsize,
                            output     = wsize,
                            dropout    = params[f'layer{i}']['dropout'],
                            normalize  = params[f'layer{i}']['normalize'],
                            activation = params[f'layer{i}']['activation'],
                        )
                else:
                    zsize = params[f'layer{i}']['size']
                    if (i == 0) and (num_classes is not None):
                        zsize += num_classes
                    if i == depth - 1:
                        wsize = self.window_size
                    else: wsize = params[f'layer{i + 1}']['size']
                    modules[f'layer{i}'] = FullyConnected(
                        input      = zsize,
                        output     = wsize,
                        dropout    = params[f'layer{i}']['dropout'],
                        normalize  = params[f'layer{i}']['normalize'],
                        activation = params[f'layer{i}']['activation'],
                    )
            ## We create funnels for each window.
            if not self.window_cloning:
                self.funnel = nn.ModuleList()
                for w in range(self.n_windows):
                    self.funnel.append(nn.Sequential(*[modules[f'layer{i}_win{w}'] for i in range(depth)]))
            else:
                self.funnel = nn.Sequential(*[modules[f'layer{i}'] for i in range(depth)])
            self.split_size = params['layer0']['size']
            
        elif self.shape == 'hybrid':
            self.params = params
            self.dmodules = nn.ModuleDict()
            ## Define a binary tree of layers using a dict
            nw = 2
            for i in range(depth):
                aux = nn.ModuleList()
                ## If last layerD, we split the input size.
                ## Otherwise, we use the size defined in params.
                for w in range(nw):
                    zsize = params[f'layer{i}']['size']
                    if (i == 0) and (num_classes is not None):
                        zsize += num_classes
                    if i == depth - 1:
                        if w != self.n_windows - 1:
                            wsize = self.window_size
                        else: wsize = self.window_size + (params[f'layer{depth}']['size'] % self.window_size)
                    else: wsize = params[f'layer{i + 1}']['size']
                    aux.append(
                        FullyConnected(
                            input      = zsize,
                            output     = wsize,
                            dropout    = params[f'layer{i}']['dropout'],
                            normalize  = params[f'layer{i}']['normalize'],
                            activation = params[f'layer{i}']['activation'],
                        )
                    )
                self.dmodules.update({f'group{i}' : aux})
                nw *= 2
        else: raise Exception('Unknown shape.')

    def forward(self, z, c=None):
        if self.shape == 'global':
            o = self.FCs(z if c is None else torch.cat([z, c], -1))
            return o
        elif self.shape == 'window-based':
            os = [] ## Stores the outputs of each window.
            for w in range(self.n_windows):
                z_windowed = z[..., w * self.split_size: (w + 1) * self.split_size]
                if c is not None:
                    z_windowed = torch.cat([z_windowed, c], -1)
                if self.window_cloning: os.append(self.funnel(z_windowed))
                else: 
                    if self.heads > 1: z_windowed = z_windowed.sum(axis=1)
                    os.append(self.funnel[w](z_windowed))
            ## Concat all outputs into a single o vector.
            o = torch.cat(os, axis=-1)
            return o
        elif self.shape == 'hybrid':
            aux = [] ## Stores the outputs of each window.
            ws = 2
            for i, group in enumerate(self.dmodules.keys()):
                # print(f'In group we have nodes in group {i}: {len(self.modules[group])}')
                # print(f'Current num of window to split into: {ws}')
                w_id_split = 0
                for w_id in range(0,ws,2):
                    # print(f'Current nodes: {w_id} and {w_id + 1}')
                    ## Recompute sizes on each layer given params
                    if i == 0: ## Forward the bottleneck completely
                        if c is not None:
                             z = torch.cat([z, c], -1)
                        w1 = w2 = z
                    else:
                        wsize = self.params[f'layer{i}']['size']
                        w1 = z[..., w_id_split * wsize:(w_id_split + 1) * wsize]
                        w2 = z[..., w_id_split * wsize:(w_id_split + 1) * wsize]
                        # print(f'Shape of w1: {w1.shape}, sliced: [{w_id_split * wsize},{(w_id_split + 1) * wsize}]')
                        # print(f'Shape of w2: {w2.shape}, sliced: [{w_id_split * wsize},{(w_id_split + 1) * wsize}]')
                    ## Store nodes in aux
                    node1 = self.dmodules[group][w_id](w1)
                    # print(f'Shape of node1: {node1.shape}')
                    node2 = self.dmodules[group][w_id + 1](w2)
                    # print(f'Shape of node2: {node2.shape}')
                    aux.append(node1)
                    aux.append(node2)
                    w_id_split += 1
                ## Increment number of windows on next layer
                ws *= 2
                z = torch.cat(aux, axis=-1)
                # print(f'Shape of out: {z.shape}')
                aux = []
            o = z
            return o
        else: raise Exception('Unknown shape.')