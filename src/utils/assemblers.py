import os
import h5py
import numpy as np
from utils.decorators import timer

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
def holdout_by_pop(snps, populations, *ratios, seed=123):
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
    
    for id, pop in enumerate(np.unique(populations)):
        pop_idx = np.where(np.asarray(populations) == pop)[0]
        _subsets = splitter(pop_idx, ratios, seed=seed)
        
        print(f'Population #{pop} has {len(pop_idx)} samples.\n'+'-' * 50)
        for i in range(_r):
            print(f'Subsetting {ratios[i] * 100}%, {len(_subsets[i])} samples.')
            if id == 0:
                _sets[i] = snps[_subsets[i],:]
                print(' ' * 5 + f'Stored {_sets[i].shape} matrix in set #{i}.')
            else:
                _subset_i = snps[_subsets[i],:]
                _sets[i] = np.vstack((_sets[i],_subset_i))
                print(' ' * 5 + f'Stored {_subset_i.shape} matrix in set #{i}.')
                print(' ' * 5 + f'In total {_sets[i].shape} matrix in set #{i}.')
        print('\n')
    
    print('Shuffling each set...\n')
    for i in range(_r):
        np.random.shuffle(_sets[i])
        
    print('-' * 50 + f'\nTotal counts:\n'+'-' * 50)
    for i in range(_r):
        print(f'{_sets[i].shape[0]} samples in set #{i}.')
    
    return _sets

if __name__ == '__main__':
    npy2hdf5(
        ipath='data/ancestry_datasets/All_chm_World/all_chm_combined_snps_world_2M_with_labels.npz',
        opath='data/prepared/single_ancestry/data.h5',
        max_limit=1000,
        max_variance=True,
    )

    h5f = h5py.File(os.path.join(os.environ.get('USER_PATH'), 'data/prepared/single_ancestry/data.h5'), 'r')
    snps = h5f['snps'][:]
    populations = h5f['populations'][:]
    h5f.close()

    train, valid = holdout_by_pop(snps, populations, 0.8, 0.2, seed=123)

    print('Storing train.hdf5...')
    h5f = h5py.File(os.path.join(os.environ.get('USER_PATH'), 'data/prepared/single_ancestry/train.h5'), 'w')
    h5f.create_dataset('train', data=train)
    h5f.close()
    print('Done.\n')

    print('Storing valid.hdf5...')
    h5f = h5py.File(os.path.join(os.environ.get('USER_PATH'), 'data/prepared/single_ancestry/valid.h5'), 'w')
    h5f.create_dataset('valid', data=valid)
    h5f.close()
    print('Done.\n')
