import torch
import numpy as np
import torch.nn as nn

class FullyConnected(nn.Module):
    def __init__(self, input, output, dropout=0, normalize=True, activation=None):
        super().__init__()  

        modules = []
        if dropout > 0:
            modules.append(nn.AlphaDropout(p=dropout))
        modules.append(nn.Linear(input, output))
        if normalize:
            modules.append(nn.BatchNorm1d(output))
        if activation is not None:
            if activation == 'ReLU':
                modules.append(nn.ReLU())
            elif activation == 'Sigmoid':
                modules.append(nn.Sigmoid())
            elif activation == 'Softplus':
                modules.append(nn.Softplus())
            elif activation == 'GELU': # GAUSSIAN ERROR LINEAR UNITS
                modules.append(nn.GELU())
        self.FC = nn.Sequential(*modules)

    def forward(self, x):
        o = self.FC(x)
        return o


class Encoder(nn.Module):
    def __init__(self, params):
        super().__init__()

        modules = []
        depth = len(params.keys()) - 2
        for i in range(depth):
            modules.append(FullyConnected(
                input=params['input']['size'] if i == 0 else params[f'hidden{i}']['size'],
                output=params[f'hidden{i + 1}']['size'],
                dropout=params['input']['dropout'] if i == 0 else params[f'hidden{i}']['dropout'],
                normalize=params['input']['normalize'] if i == 0 else params[f'hidden{i}']['normalize'],
                activation=params['input']['activation'] if i == 0 else params[f'hidden{i}']['activation'],
            ))
        self.FCs = nn.Sequential(*modules)
        self.FCmu = FullyConnected(
            input=params[f'hidden{depth}']['size'],
            output=params[f'hidden{depth + 1}']['size'],
            dropout=params[f'hidden{depth}']['dropout'],
            normalize=params[f'hidden{depth}']['normalize'],
            activation=params[f'hidden{depth}']['activation'],
        )
        self.FClogvar = FullyConnected(
            input=params[f'hidden{depth}']['size'],
            output=params[f'hidden{depth + 1}']['size'],
            dropout=params[f'hidden{depth}']['dropout'],
            normalize=params[f'hidden{depth}']['normalize'],
            activation=params[f'hidden{depth}']['activation'],
        )

    def forward(self, x):
        o = self.FCs(x)
        o_mu = self.FCmu(o)
        o_var = self.FClogvar(o)
        return o_mu, o_var

class Decoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        modules = []
        depth = len(params.keys())
        for i in range(1, depth):
            modules.append(FullyConnected(
                input=params[f'hidden{depth - i}']['size'],
                output=params['output']['size'] if i == depth - 1 else params[f'hidden{depth - i - 1}']['size'],
                dropout=params[f'hidden{depth - i}']['dropout'],
                normalize=params[f'hidden{depth - i}']['normalize'],
                activation=params[f'hidden{depth - i}']['activation'],
            ))
        self.FCs = nn.Sequential(*modules)

    def forward(self, z):
        o = self.FCs(z)
        return o

class VAEgen(nn.Module):
    def __init__(self, params):
        super().__init__()
        
        self.encoder = Encoder(params=params['encoder'])
        self.decoder = Decoder(params=params['decoder'])
    
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        z_mu, z_logvar = self.encoder(x)
        z = self.reparametrize(z_mu, z_logvar)
        o = self.decoder(z)
        return o, z_mu, z_logvar