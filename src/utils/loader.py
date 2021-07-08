import os
import time
import torch
import random
import h5py
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.decorators import timer 
from utils.assemblers import one_hot_encoder

class SNPs(Dataset):
    def __init__(self, ipath, split_set='train', ksize=5, only=None, conditional=False):
        """
        Inputs:
            split_set: to load.
            ksize: size of SNP arrays in terms of thousands.
            only: load one population.
            conditional: return one-hot labels of classes with num classes.
        """

        h5f = h5py.File(os.path.join(ipath, f'{split_set}/{split_set}{ksize}K.h5'), 'r')
        self.snps = h5f['snps'][:].astype(float)
        self.populations = h5f['populations'][:].astype(int)
        h5f.close()

        if only is not None:
            pop_idx = np.where(np.asarray(self.populations) == only)[0]
            self.snps = self.snps[pop_idx,:]
        
        if conditional:
            self.populations = one_hot_encoder(self.populations, len(np.unique(self.populations)))

    def __len__(self):
        return self.snps.shape[0]

    def __getitem__(self, index):
        snps_array = self.snps[index,:]
        label = self.populations[index]
        # snps_array[np.where(snps_array == 0)] = -1 for denoising/imputation VAE
        snps_array = torch.from_numpy(snps_array).float()
        return (snps_array, label)

@timer
def loader(ipath, batch_size, split_set='train', ksize=5, only=None, conditional=False):
    if only is not None and conditional: raise Exception('Conditional VAE with a unique population.')
    dataset = SNPs(ipath=ipath, split_set=split_set, ksize=ksize, only=only, conditional=conditional)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return dataloader

if __name__ == '__main__':
    data = loader(
        ipath=os.path.join(os.environ.get('IN_PATH'), 'data/human/chr22/prepared'),
        batch_size=64, 
        split_set='train',
        ksize=5
    )
    