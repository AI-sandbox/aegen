import os
import time
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from .decorators import timer 

class SNPs(Dataset):
    def __init__(self, data, max_limit=25000, max_variance=True, split_set='train', seed=123):
        """
        Inputs:
            data: SNPs data in .npz files
            max_limit: maximum number of SNPs to use
            max_variance: select SNPs with maximum variance
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        data = np.load(data, allow_pickle=True)
        self.snps = data["snps"].astype(float)
        self.pops = data["populations"]
        # self.spops = data["subpopulations"][:max_limit]
        self._max_limit = max_limit

        print(self.snps.shape)
        print(self.pops.shape)

        pops = ['AFR', 'EAS', 'EUR', 'AMR', 'OCE', 'SAS', 'WAS']
        self.mapper = dict(zip(pops, np.arange(0, len(pops))))

        if max_variance:
            variances = self.snps.var(axis=0)
            max_var_snps_idx = variances.argpartition(variances.size-max_limit, axis=None)[-max_limit:]
            self.snps = self.snps[:, max_var_snps_idx]
        else:
            self.snps = self.snps[:,:max_limit]

    def __len__(self):
        return self.snps.shape[0]

    def __getitem__(self, index):
        snps_array = self.snps[index,:]
        # snps_array[np.where(snps_array == 0)] = -1 for denoising/imputation VAE
        snps_array = torch.tensor(snps_array).float()
        # pop_id = torch.tensor(self.mapper[self.pops[index]]).int()
        # spop_id = torch.tensor(self.spops[index]).int()
        # return (snps_array, pop_id)
        return snps_array

@timer
def loader(data_path, batch_size, max_limit, split_set='train', seed='123'):
    path = f'{data_path}/All_chm_World/all_chm_combined_snps_world_2M_with_labels.npz'
    dataset = SNPs(data=path, max_limit=max_limit, max_variance=True, split_set=split_set, seed=seed)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return dataloader

if __name__ == '__main__':
    data = loader(
        data_path=os.path.join(os.environ.get('USER_PATH'), 'data/ancestry_datasets'),
        batch_size=64, 
        max_limit=100000,
        split_set='train'
    )
    print(data)