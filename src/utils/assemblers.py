import os
import h5py
import numpy as np
from decorators import timer

def encode(catvar):
    cat = np.unique(catvar)
    ordinal = dict(zip(cat, np.arange(0, len(cat))))
    mapper = np.vectorize(lambda x: ordinal[x])
    return mapper(catvar), ordinal

@timer
def npy2hdf5(ipath, opath, max_limit=25000, max_variance=True):
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
    print('Done.')
    print(h5f.keys())

    h5f.close()

if __name__ == '__main__':
    npy2hdf5(
        ipath='data/ancestry_datasets/All_chm_World/all_chm_combined_snps_world_2M_with_labels.npz',
        opath='data/prepared/single_ancestry/data.h5',
        max_limit=100000,
        max_variance=True,
    )







    

