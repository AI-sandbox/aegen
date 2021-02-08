import os
import time
import torch
import random
import h5py
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.decorators import timer 

class SNPs(Dataset):
    def __init__(self, ipath, split_set='train'):
        """
        Inputs:
            split_set: to load.
        """

        h5f = h5py.File(os.path.join(os.environ.get('USER_PATH'), f'{ipath}/{split_set}.h5'), 'r')
        self.snps = h5f[split_set][:].astype(float)
        h5f.close()

        print(self.snps.shape)

    def __len__(self):
        return self.snps.shape[0]

    def __getitem__(self, index):
        snps_array = self.snps[index,:]
        # snps_array[np.where(snps_array == 0)] = -1 for denoising/imputation VAE
        snps_array = torch.from_numpy(snps_array).float()
        # return (snps_array, pop_id)
        return snps_array

@timer
def loader(ipath, batch_size, split_set='train'):
    dataset = SNPs(ipath=ipath, split_set=split_set)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return dataloader

if __name__ == '__main__':
    data = loader(
        ipath=os.path.join(os.environ.get('USER_PATH'), 'data/prepared/single_ancestry'),
        batch_size=64, 
        split_set='train'
    )
    print(data)