import os
import gc
import glob
import h5py
import torch
import logging
import numpy as np
from utils.decorators import timer
from torch.autograd import Variable

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def encode(catvar):
    """
    Encodes a categorical variable with strings into a numeric vector.
    Needed for hdf5 storage.

    Inputs:
        catvar: the categorical variable to encode.
    """
    cat = np.unique(catvar)
    ordinal = dict(zip(cat, np.arange(0, len(cat))))
    mapper = np.vectorize(lambda x: ordinal[x])
    return mapper(catvar), ordinal

def one_hot_encoder(labels, num_classes):
    """
    Encodes in one-hot encoding a list of labels
    given the total number of classes.

    Inputs:
        labels: a 1-D array with labels.
        num_classes: the total number of classes.
    """
    targets = torch.zeros(labels.shape[0], num_classes)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return Variable(targets)

@timer
def npy2hdf5(ipath, opath, max_limit=25000, max_variance=True):
    """
    Converts a SNPs .npy file into a .hdf5 dataset.

    Inputs:
        ipath: location of SNPs data in .npz file.
        opath: location to store the .hdf5 file.
        max_limit: maximum number of SNPs to use.
        max_variance: select SNPs with maximum variance.
    """
    print('Loading npy...')
    data = np.load(ipath, mmap_mode='r+', allow_pickle=True)
    print('Loaded.')
    snps = data['snps']

    assert(max_limit <= snps.shape[1])

    if max_variance:
        print('Computing max variance...')
        variances = snps.var(axis=0)
        max_var_snps_idx = variances.argpartition(variances.size-max_limit, axis=None)[-max_limit:]
        snps = snps[:, max_var_snps_idx]
        print('Storing...')
    else:
        snps = snps[:,:max_limit]
    
    print('Encoding labels...')
    populations, _ = encode(data['populations'])
    subpopulations, _ = encode(data['subpopulations'])
    print('Encoded.')

    print('Storing hdf5...')
    h5f = h5py.File(os.path.join(os.environ.get('USER_PATH'), opath), 'w')
    h5f.create_dataset('snps', data=snps)
    h5f.create_dataset('populations', data=populations)
    h5f.create_dataset('subpopulations', data=subpopulations)
    print('Done.\n')

    h5f.close()

def splitter(arr, ratios, seed=123):
    """
    Splits an array by ratios.
    Note: ratios must be a 'tuple'.

    Returns a list with the array splits by ratio.

    Inputs:
        arr: the array to split.
        ratios: a tuple with the % to split.
        seed: random seed to permute the array before splitting.
    """

    np.random.seed(int(seed))
    arr = np.random.permutation(arr)
    ind = np.add.accumulate(np.array(ratios) * len(arr)).astype(int)
    return [x.tolist() for x in np.split(arr, ind)][:len(ratios)]

@timer
def holdout_by_pop(snps, populations, *ratios, seed=123, verbose=True):
    """
    Creates a holdout partition by equal proportions for each population.
    The ratios for partitions are passed by 'ratios' parameter.

    Returns the sets with proportional subsets of each population.

    Inputs:
        snps: the SNPs matrix.
        populations: indices of the corresponding population of each SNP array.
        ratios: % to split.
        seed: random seed to permute the array before splitting and shuffling.
    """
    
    _r = len(ratios)
    _sets = [None] * _r
    _pops = [None] * _r
    
    for id, pop in enumerate(np.unique(populations)):
        pop_idx = np.where(np.asarray(populations) == pop)[0]
        _subsets = splitter(pop_idx, ratios, seed=seed)
        
        if verbose: print(f'Population #{pop} has {len(pop_idx)} samples.\n'+'-' * 50)
        for i in range(_r):
            if verbose: print(f'Subsetting {ratios[i] * 100}%, {len(_subsets[i])} samples.')
            if id == 0:
                _sets[i] = snps[_subsets[i],:]
                _pops[i] = populations[_subsets[i]]
                if verbose: print(' ' * 5 + f'Stored {_sets[i].shape} matrix in set #{i}.')
            else:
                _subset_i = snps[_subsets[i],:]
                _subpop_i = populations[_subsets[i]]
                _sets[i] = np.vstack((_sets[i],_subset_i))
                _pops[i] = np.append(_pops[i], _subpop_i, axis=0)
                if verbose: print(' ' * 5 + f'Stored {_subset_i.shape} matrix in set #{i}.')
                if verbose: print(' ' * 5 + f'In total {_sets[i].shape} matrix in set #{i}.')
        if verbose: print('\n')
    
    if verbose: print('Shuffling each set...\n')
    for i in range(_r):
        np.random.seed(int(seed))
        np.random.shuffle(_sets[i])
        np.random.seed(int(seed))
        np.random.shuffle(_pops[i])
        
    if verbose:
        print('-' * 50 + f'\nTotal counts:\n'+'-' * 50)
        for i in range(_r):
            print(f'{_sets[i].shape[0]} samples in set #{i}.')
    
    return _sets, _pops

def get_snps_by_pop(pop, split, max_size=5000):
    log.info(f'Fetching SNPs for population {pop}')
    for i, snps_arr in enumerate(glob.glob(os.path.join(os.environ.get('IN_PATH'), f'data/chr22/prepared/{split}/{pop}/generations/{pop}_gen_*.npy'))):
        aux = np.load(snps_arr, mmap_mode='r')[:,:max_size]
        log.info(f'Generation {i+1} has {aux.shape[0]} individuals')
        if i == 0:
            arr = np.empty((0, aux.shape[1]), int)
        arr = np.vstack((arr, aux))
        del aux
        gc.collect()
    log.info('Done.')
    return arr

def create_dataset(max_size=5000, seed=123):
    pops = ['EUR', 'EAS', 'AMR', 'SAS', 'AFR', 'OCE', 'WAS']
    for split in ['train', 'valid', 'test']:
        for i in range(1, len(pops)):
            if i == 1:
                pop0, pop1 = get_snps_by_pop(pops[0], split=split, max_size=max_size), get_snps_by_pop(pops[1], split=split, max_size=max_size)
                X, Y = np.vstack((pop0, pop1)), np.concatenate((np.array([0]*len(pop0)), np.array([1]*len(pop1))), axis=0)
            else:
                popI = get_snps_by_pop(pops[i], split=split, max_size=max_size)
                X, Y = np.vstack((X, popI)), np.concatenate((Y, np.array([i]*len(popI))), axis=0)
            assert len(X) == len(Y)
        np.random.seed(seed)
        idxs = np.arange(len(X))
        np.random.shuffle(idxs)
        X, Y = X[idxs], Y[idxs]
        log.info(f'Storing {split} hdf5 of shape ({X.shape})...')
        h5f = h5py.File(os.path.join(os.environ.get('OUT_PATH'),f'data/chr22/prepared/{split}/{split}{int(max_size/1000)}K.h5'), 'w')
        h5f.create_dataset('snps', data=X)
        h5f.create_dataset('populations', data=Y)
        h5f.close()
        log.info('Done.\n')

if __name__ == '__main__':
    npy2hdf5(
        ipath='data/ancestry_datasets/All_chm_World/all_chm_combined_snps_world_2M_with_labels.npz',
        opath='data/prepared/single_ancestry/data.h5',
        max_limit=2000,
        max_variance=False,
    )

    h5f = h5py.File(os.path.join(os.environ.get('USER_PATH'), 'data/prepared/single_ancestry/data.h5'), 'r')
    snps = h5f['snps'][:]
    populations = h5f['populations'][:]
    h5f.close()

    train, valid = holdout_by_pop(snps, populations, 0.8, 0.2, seed=123)

    print('Storing train.hdf5...')
    h5f = h5py.File(os.path.join(os.environ.get('OUT_PATH'), 'data/prepared/train.h5'), 'w')
    h5f.create_dataset('train', data=train)
    h5f.close()
    print('Done.\n')

    print('Storing valid.hdf5...')
    h5f = h5py.File(os.path.join(os.environ.get('OUT_PATH'), 'data/prepared/valid.h5'), 'w')
    h5f.create_dataset('valid', data=valid)
    h5f.close()
    print('Done.\n')
