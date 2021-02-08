import torch
import numpy as np
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input, hidden1, hidden2, latent):
        super().__init__()

        self.bn_0 = nn.BatchNorm1d(input)
        self.fc_1 = nn.Linear(input, hidden1)
        self.bn_1 = nn.BatchNorm1d(hidden1)
        self.ac_1 = nn.ReLU()
        self.dp_1 = nn.Dropout(p=0.5)
        self.fc_2 = nn.Linear(hidden1, hidden2)
        self.bn_2 = nn.BatchNorm1d(hidden2)
        self.ac_2 = nn.ReLU()
        self.dp_2 = nn.Dropout(p=0.5)
        self.fc_mu = nn.Linear(hidden2, latent)
        self.bn_mu = nn.BatchNorm1d(latent)
        self.ac_mu = nn.ReLU()
        self.fc_logvar = nn.Linear(hidden2, latent)
        self.bn_logvar = nn.BatchNorm1d(latent)
        self.ac_logvar = nn.ReLU()

    def forward(self, x):
        o = self.bn_0(x)
        o = self.ac_1(self.bn_1(self.fc_1(x)))
        o = self.dp_1(o)
        o = self.ac_2(self.bn_2(self.fc_2(o)))
        o = self.dp_2(o)
        o_mu = self.bn_mu(self.fc_mu(o))
        o_var = self.bn_logvar(self.fc_logvar(o))
        return o_mu, o_var

class Decoder(nn.Module):
    def __init__(self, latent, hidden2, hidden1, output):
        super().__init__()

        self.fc_1 = nn.Linear(latent, hidden2)
        self.bn_1 = nn.BatchNorm1d(hidden2)
        self.ac_1 = nn.ReLU()
        self.dp_1 = nn.Dropout(p=0.5)
        self.fc_2 = nn.Linear(hidden2, hidden1)
        self.bn_2 = nn.BatchNorm1d(hidden1)
        self.ac_2 = nn.ReLU()
        self.dp_2 = nn.Dropout(p=0.5)
        self.fc_3 = nn.Linear(hidden1, output)
        self.ac_3 = nn.Sigmoid()

    def forward(self, z):
        o = self.ac_1(self.bn_1(self.fc_1(z)))
        o = self.dp_1(o)
        o = self.ac_2(self.bn_2(self.fc_2(o)))
        o = self.dp_2(o)
        o = self.ac_3(self.fc_3(o))
        return o

class VAEgen(nn.Module):
    def __init__(self, input, hidden1, hidden2, latent):
        super().__init__()
        
        self.encoder = Encoder(input=input, hidden1=hidden1, hidden2=hidden2, latent=latent)
        self.decoder = Decoder(latent=latent, hidden2=hidden2, hidden1=hidden1, output=input)
    
    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        z_mu, z_logvar = self.encoder(x)
        z = self.reparametrize(z_mu, z_logvar)
        o = self.decoder(z)
        return o, z_mu, z_logvar