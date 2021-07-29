import torch
import numpy as np
import torch.nn as nn

## Function to compute the padding size to output a tensor of the same
## size as the input tensor.
def optimal_padding(in_shape, kernel_size, stride=1):
    return int(((stride - 1)* in_shape - stride + kernel_size) // 2)

## Module for custom MLP
class FullyConnected(nn.Module):
    def __init__(self, input, output, dropout=0, normalize=True, activation=None):
        super().__init__()  
        modules = []
        ## Append dropout if desired.
        if dropout is not None and dropout > 0: modules.append(nn.Dropout(p=dropout))
        modules.append(nn.Linear(input, output))
        ## Normalize layer if desired.
        if normalize: modules.append(nn.BatchNorm1d(output))
            
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