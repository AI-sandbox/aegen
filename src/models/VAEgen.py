import torch
import numpy as np
import torch.nn as nn

class FullyConnected(nn.Module):
    def __init__(self, input, output, dropout=0, normalize=True, activation=None):
        super().__init__()  

        modules = []
        if dropout is not None and dropout > 0:
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
            elif activation == 'GELU':
                modules.append(nn.GELU())
        self.FC = nn.Sequential(*modules)

    def forward(self, x):
        o = self.FC(x)
        return o


class Encoder(nn.Module):
    def __init__(self, params, num_classes=None):
        super().__init__()
        modules = []
        depth = len(params.keys()) - 2
        for i in range(depth):
            modules.append(FullyConnected(
                input=(
                    params['input']['size'] if num_classes is None else (params['input']['size'] + num_classes)
                ) if i == 0 else params[f'hidden{i}']['size'],
                output=params[f'hidden{i + 1}']['size'],
                dropout=params['input']['dropout'] if i == 0 else params[f'hidden{i}']['dropout'],
                normalize=params['input']['normalize'] if i == 0 else params[f'hidden{i}']['normalize'],
                activation=params['input']['activation'] if i == 0 else params[f'hidden{i}']['activation'],
            ))
        self.FCs = nn.Sequential(*modules)
        self.FCmu = FullyConnected(
            input=params['input']['size'] if depth == 0 else params[f'hidden{depth}']['size'],
            output=params[f'hidden{depth + 1}']['size'],
            dropout=params['input']['dropout'] if depth == 0 else params[f'hidden{depth}']['dropout'],
            normalize=params['input']['normalize'] if depth == 0 else params[f'hidden{depth}']['normalize'],
            activation=params['input']['activation'] if depth == 0 else params[f'hidden{depth}']['activation'],
        )
        self.FClogvar = FullyConnected(
            input=params['input']['size'] if depth == 0 else params[f'hidden{depth}']['size'],
            output=params[f'hidden{depth + 1}']['size'],
            dropout=params['input']['dropout'] if depth == 0 else params[f'hidden{depth}']['dropout'],
            normalize=params['input']['normalize'] if depth == 0 else params[f'hidden{depth}']['normalize'],
            activation=params['input']['activation'] if depth == 0 else params[f'hidden{depth}']['activation'],
        )

    def forward(self, x, c=None):
        o = self.FCs(x if c is None else torch.cat([x, c], 1))
        o_mu = self.FCmu(o)
        o_var = self.FClogvar(o)
        return o_mu, o_var

class Decoder(nn.Module):
    def __init__(self, params, num_classes=None):
        super().__init__()
        
        modules = []
        depth = len(params.keys())
        for i in range(1, depth):
            input_size = params[f'hidden{depth - i}']['size'] 
            if num_classes is not None and i == 1: input_size += num_classes
            modules.append(FullyConnected(
                input=input_size,
                output=params['output']['size'] if i == depth - 1 else params[f'hidden{depth - i - 1}']['size'],
                dropout=params[f'hidden{depth - i}']['dropout'],
                normalize=params[f'hidden{depth - i}']['normalize'],
                activation=params[f'hidden{depth - i}']['activation'],
            ))
        self.FCs = nn.Sequential(*modules)

    def forward(self, z, c=None):
        o = self.FCs(z if c is None else torch.cat([z, c], 1))
        return o

class VAEgen(nn.Module):
    def __init__(self, params, conditional=False, imputation=False):
        super().__init__()
        
        self.imputation, self.missing = imputation, params['missing']
        self.encoder = Encoder(
            params=params['encoder'], 
            num_classes=params['num_classes'] if conditional else None
        ) 
        self.decoder = Decoder(
            params=params['decoder'], 
            num_classes=params['num_classes'] if conditional else None
        ) 
    
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x, c=None):
        if self.imputation: 
            x, sz = 2 * x - 1, x.shape
            if self.missing > 0:
                mask = torch.from_numpy((np.random.uniform(size=np.prod(sz)) < self.missing)).type(torch.bool).reshape(sz).cuda()
                x[mask] = 0
        z_mu, z_logvar = self.encoder(x, c)
        z = self.reparametrize(z_mu, z_logvar)
        o = self.decoder(z, c)
        if self.imputation and self.missing > 0: return o, z_mu, z_logvar, mask
        else: return o, z_mu, z_logvar