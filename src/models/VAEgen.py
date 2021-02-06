import torch
import numpy as np
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input, hidden, latent):
        super().__init__()

        self.fc_1 = nn.Linear(input, hidden)
        self.fc_mu = nn.Linear(hidden, latent)
        self.fc_logvar = nn.Linear(hidden, latent)
        self.ReLU = nn.ReLU()

    def forward(self, x):
        o = self.ReLU(self.fc_1(x))
        o_mu = self.ReLU(self.fc_mu(o))
        o_var = self.ReLU(self.fc_logvar(o))
        return o_mu, o_var

class Decoder(nn.Module):
    def __init__(self, latent, hidden, output):
        super().__init__()

        self.fc_1 = nn.Linear(latent, hidden)
        self.fc_2 = nn.Linear(hidden, output)
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()

    def forward(self, z):
        o = self.ReLU(self.fc_1(z))
        o = self.Sigmoid(self.fc_2(o))
        return o

class VAEgen(nn.Module):
    def __init__(self, input, hidden, latent):
        super().__init__()
        
        self.encoder = Encoder(input=input, hidden=hidden, latent=latent)
        self.decoder = Decoder(latent=latent, hidden=hidden, output=input)
    
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        z_mu, z_logvar = self.encoder(x)
        z = self.reparametrize(z_mu, z_logvar)
        o = self.decoder(z)
        return o, z_mu, z_logvar

