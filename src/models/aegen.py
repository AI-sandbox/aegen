import torch
import numpy as np
import torch.nn as nn

class FullyConnected(nn.Module):
    def __init__(self, input, output, dropout=0, normalize=True, activation=None):
        super().__init__()  
        modules = []
        ## Append dropout if desired.
        if dropout is not None and dropout > 0:
            modules.append(nn.AlphaDropout(p=dropout))
        modules.append(nn.Linear(input, output))
        ## Normalize layer if desired.
        if normalize: ## TODO: BatchNorm
            modules.append(nn.BatchNorm1d(output))
            
        ## Add activation function if desired.
        if activation is not None:
            if activation == 'ReLU':
                modules.append(nn.ReLU())
            elif activation == 'Sigmoid':
                modules.append(nn.Sigmoid())
            elif activation == 'Softplus':
                modules.append(nn.Softplus())
            elif activation == 'GELU':
                modules.append(nn.GELU())
            elif activation == 'Tanh':
                modules.append(nn.Tanh())
        self.FC = nn.Sequential(*modules)

    def forward(self, x):
        o = self.FC(x)
        return o

class Encoder(nn.Module):
    def __init__(self, latent_distribution, params, num_classes=None, shape='global', window_size=None, n_windows=None, window_cloning=None):
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
                modules.append(FullyConnected(
                    ## Only can condition input (layer0).
                    input = (
                        params[f'layer{i}']['size'] if num_classes is None else (params[f'layer{i}']['size'] + num_classes)
                    ) if i == 0 else params[f'layer{i}']['size'],
                    output     = params[f'layer{i + 1}']['size'],
                    dropout    = params[f'layer{i}']['dropout'],
                    normalize  = params[f'layer{i}']['normalize'],
                    activation = params[f'layer{i}']['activation'],
                ))
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
                        self.funnel.append(nn.Sequential(*[modules[f'layer{i}_win{w}'] for i in range(depth)]))
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
        elif self.shape == 'window-based': ## TODO: Implement conditioned window-based
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
                    if not self.window_cloning: os.append(self.funnel[w](x_windowed))
                    else: os.append(self.funnel(x_windowed))
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
        
class Quantizer(nn.Module):
    def __init__(self, latent_distribution, params, shape='global', codebook = None, window_size=None, n_windows=None):
        super().__init__()
        self.latent_distribution = latent_distribution
        self.codebook = codebook
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
        if ze.size(-1) != self.codebook.size(-1):
            raise RuntimeError(
                f'[Error] Invalid argument: ze.size(-1) ({ze.size(-1)}) must \
                be equal to self.codebook.size(-1) ({self.codebook.size(-1)})'
            )
        sq_norm = (torch.sum(ze**2, dim = -1, keepdim = True) 
                + torch.sum(self.codebook**2, dim = 1)
                - 2 * torch.matmul(ze, self.codebook.t()))
        _, argmin = sq_norm.min(-1)
        zq = self.codebook.index_select(0, argmin.view(-1)).view(ze.shape)
        return zq
         
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
                zq = []
                for w in range(self.n_windows):
                    ze_windowed = ze[..., w * self.split_size: (w + 1) * self.split_size]
                    zq.append(self._return_code(ze_windowed))
                zq = torch.cat(zq, axis=-1)
            else: zq = self._return_code(ze)
        return zq
    
    def backward(self, grad_zq):
        ## Clone decoder gradients to encoder.
        grad_ze = grad_zq.clone()
        return grad_ze, None
        
class Decoder(nn.Module):
    def __init__(self, params, num_classes=None, shape='global', window_size=None, n_windows=None, window_cloning=None):
        super().__init__()
        ## Define the shape of the Encoder.
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
                if not self.window_cloning: os.append(self.funnel[w](z_windowed))
                else: os.append(self.funnel(z_windowed))
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

class aegen(nn.Module):
    def __init__(self, params, conditional=False, sample_mode=False, imputation=False):
        super().__init__()
        ## AE shape can be:
        ## - global: regular MLP.
        ## - window-based: independent MLPs by windows.
        ## - hybrid: independent MLP combined into one.
        self.shape = params['shape']
        self.window_size = params['window_size'] if params['window_size'] is not None else None
        self.n_windows = params['n_windows'] if params['n_windows'] is not None else None
        if (self.window_size is not None) and (self.n_windows is not None):
            raise Exception('[ERROR] Too many arguments.')
        self.window_cloning = params['window_cloning']
        if (self.shape == 'window-based') and (self.window_cloning is None):
            raise Exception('[ERROR] Window cloning is not defined using window-based shape.')
        ## Latent space distrubution can be:
        ## - None: regular AE.
        ## - Gaussian: regular VAE.
        ## - Multi-Bernoulli: LBAE. http://proceedings.mlr.press/v119/fajtl20a/fajtl20a.pdf
        ## - Uniform: VQ-VAE.
        self.latent_distribution = params['distribution']
        ## Encoder is defined by:
        ## - Latent distribution.
        ## - Parameters: layers' definitions.
        ## - Num_classes: if conditional Encoder.
        self.encoder = Encoder(
            latent_distribution = self.latent_distribution,
            params=params['encoder'], 
            num_classes=params['conditioning']['num_classes'] if conditional else None,
            shape=self.shape,
            window_size=self.window_size,
            n_windows=self.n_windows,
            window_cloning=self.window_cloning,
        )
        ## Decoder is defined by:
        ## - Parameters: layers' definitions.
        ## - Num_classes: if conditional Decoder.
        self.decoder = Decoder(
            params=params['decoder'], 
            num_classes=params['conditioning']['num_classes'] if conditional else None,
            shape=self.shape,
            window_size=self.window_size,
            n_windows=self.n_windows,
            window_cloning=self.window_cloning,
        ) 
        ## Optionally: a quantizer.
        ## - Latent distribution.
        ## Makes the latent space discrete.
        self.codebook = nn.Parameter(
            torch.bernoulli(
                torch.empty(
                    params['codebook_size'], params['decoder']['layer0']['size']
                ).uniform_(0, 1)
            ), requires_grad = True
        ) if self.latent_distribution == 'Uniform' else None
        self.quantizer = Quantizer(
            latent_distribution = self.latent_distribution,
            codebook = self.codebook, 
            shape = self.shape, 
            params = params['decoder'],
            window_size=self.window_size,
            n_windows=self.n_windows
        )
    
    ## Variational Auto-encoder: Gaussian latent space
    def reparametrize(self, mu, logvar, sample_mode=False):
        if self.shape == 'window-based':
            if self.training or sample_mode:
                z = []
                for i in range(len(mu)):
                    std = torch.exp(0.5 * logvar[i])
                    eps = torch.randn_like(std)
                    z.append(mu[i] + std * eps)
                return torch.cat(z, axis=-1)
            else: return torch.cat(mu, axis=-1)
        else:
            if self.training or sample_mode:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + std * eps
            else: return mu
        
    # def sample(self, )

    def forward(self, x, c=None):
        ## Regular Autoencoder.
        if self.latent_distribution == None:
            z = self.encoder(x, c)
            o = self.decoder(z, c) 
        ## VAE encoder outputs 2 feature maps:
        ## - Mu feature.
        ## - Logvar feature.
        ## These 2 feature maps are reparametrized
        ## to yield z - the latent factors' vector.
        elif self.latent_distribution == 'Gaussian':
            z_mu, z_logvar = self.encoder(x, c)
            z = self.reparametrize(z_mu, z_logvar)
            o = self.decoder(z, c)
        ## LBAE or VQ-VAE (Discrete Latent Spaces)
        elif self.latent_distribution == 'Multi-Bernoulli':
            ze = self.encoder(x, c)
            zq = self.quantizer(ze)
            o = self.decoder(zq, c)
        elif self.latent_distribution == 'Uniform':
            ze = self.encoder(x, c)
            zq = self.quantizer(ze)
            o = self.decoder(zq, c)
        ## Unknown distribution
        else: raise Exception('Unknown distribution.')
        return o