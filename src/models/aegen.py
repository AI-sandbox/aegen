import torch
import numpy as np
import torch.nn as nn
from modules.encoder import Encoder
from modules.quantizer import Quantizer
from modules.decoder import Decoder

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
        ## - Multi-Bernoulli: LBAE.
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
            heads=params['quantizer']['features'] if params['quantizer']['using'] else 1
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
            heads=params['quantizer']['features'] if params['quantizer']['using'] else 1
        ) 
        ## Optionally: a quantizer.
        ## - Latent distribution.
        ## Makes the latent space discrete.
        self.quantizer = Quantizer(
            latent_distribution=self.latent_distribution, 
            shape=self.shape, 
            params=params['decoder'],
            quantization=params['quantization'],
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
        if self.latent_distribution == 'Unknown':
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