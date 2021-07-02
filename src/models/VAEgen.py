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
            modules.append(nn.BatchNorm1d(1))
            
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
    def __init__(self, latent_distribution, params, num_classes=None):
        super().__init__()
        self.latent_distribution = latent_distribution
        modules = []
        depth = len(params.keys()) - 1 if self.latent_distribution != 'Gaussian' else len(params.keys()) - 2
        if (self.latent_distribution == 'Multi-Bernoulli') and (params[f'layer{depth - 1}']['activation'] != 'Tanh'):
            raise Exception('[ERROR] Missing tanh activation!')
        for i in range(depth):
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
            if params[f'layer{depth}']['activation'] != None:
                print('[WARNING] Activation for Mu and Logvar features is not the identity.')
            self.FCmu = FullyConnected(
                input      = params[f'layer{depth}']['size'],
                output     = params[f'layer{depth + 1}']['size'],
                dropout    = params[f'layer{depth}']['dropout'],
                normalize  = params[f'layer{depth}']['normalize'],
                activation = params[f'layer{depth}']['activation'],
            )
            self.FClogvar = FullyConnected(
                input      = params[f'layer{depth}']['size'],
                output     = params[f'layer{depth + 1}']['size'],
                dropout    = params[f'layer{depth}']['dropout'],
                normalize  = params[f'layer{depth}']['normalize'],
                activation = params[f'layer{depth}']['activation'],
            )

    def forward(self, x, c=None):
        o = self.FCs(x if c is None else torch.cat([x, c], -1))
        if self.latent_distribution == 'Gaussian':
            o_mu = self.FCmu(o)
            o_var = self.FClogvar(o)
            return o_mu, o_var
        else:
            return o
        
class Quantizer(nn.Module):
    def __init__(self, latent_distribution, codebook = None):
        super().__init__()
        self.latent_distribution = latent_distribution
        self.codebook = codebook
        
    def forward(self, ze):
        ## In Multi-Bernoulli LS, the quantizer
        ## uses a threshold to binarize the 
        ## latent vectors.
        if self.latent_distribution == 'Multi-Bernoulli':
            zq = torch.ones(ze.shape)
            zq[ze < 0] = -1
        ## In Uniform LS, the quantizer
        ## computes the distances to the
        ## codebook vectors and takes the argmin.
        elif self.latent_distribution == 'Uniform':
            if ze.size(-1) != self.codebook.size(-1):
                raise RuntimeError(
                    f'[Error] Invalid argument: ze.size(-1) ({ze.size(-1)}) must be equal to self.codebook.size(-1) ({self.codebook.size(-1)})'
                )
            # print(f'Codebook shape: {self.codebook.shape}')
            # print(ze)
            sq_norm = (torch.sum(ze**2, dim = -1, keepdim = True) 
                    + torch.sum(self.codebook**2, dim = 1)
                    - 2 * torch.matmul(ze, self.codebook.t()))
            # print(f'Sq Norm: {sq_norm}')
            _, argmin = sq_norm.min(-1)
            # print(f'Codebook argmin: {argmin}')
            # print(f'Codebook indices: {argmin.view(-1)}')
            zq = self.codebook.index_select(0, argmin.view(-1)).view(ze.shape)

        return zq
    
    def backward(self, grad_zq):
        ## Clone decoder gradients to encoder.
        grad_ze = grad_zq.clone()
        return grad_ze, None
        

class Decoder(nn.Module):
    def __init__(self, params, num_classes=None):
        super().__init__()
        
        modules = []
        depth = len(params.keys()) - 1
        for i in range(depth):
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

    def forward(self, z, c=None):
        o = self.FCs(z if c is None else torch.cat([z, c], -1))
        return o

class AEgen(nn.Module):
    def __init__(self, params, conditional=False, sample_mode=False):
        super().__init__()
        ## AE shape can be:
        ## - global: regular MLP.
        ## - window-based: independent MLPs by windows.
        ## - hybrid: independent MLP combined into one.
        self.shape = params['shape']
        ## Latent space distrubution can be:
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
            num_classes=params['num_classes'] if conditional else None
        )
        ## Decoder is defined by:
        ## - Parameters: layers' definitions.
        ## - Num_classes: if conditional Decoder.
        self.decoder = Decoder(
            params=params['decoder'], 
            num_classes=params['num_classes'] if conditional else None
        ) 
        ## Optionally: a quantizer.
        ## - Latent distribution.
        ## Makes the latent space discrete.
        self.codebook = nn.Parameter(
            torch.bernoulli(
                torch.empty(
                    params['codebook_size'], params['decoder']['layer0']['size']
                ).uniform_(0, 1)
            )
        ) if self.latent_distribution == 'Uniform' else None
        self.quantizer = Quantizer(
            latent_distribution = self.latent_distribution,
            codebook = self.codebook
        )
    
    ## Variational Auto-encoder: Gaussian latent space
    def reparametrize(self, mu, logvar):
        if self.training or sample_mode:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        else: return mu
        
    # def sample(self, )

    def forward(self, x, c=None):
        ## VAE encoder outputs 2 feature maps:
        ## - Mu feature.
        ## - Logvar feature.
        ## These 2 feature maps are reparametrized
        ## to yield z - the latent factors' vector.
        if self.latent_distribution == 'Gaussian':
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