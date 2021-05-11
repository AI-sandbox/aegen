import os
import h5py
import yaml
import torch
import numpy as np
import torch.nn as nn
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib

# Set environment variables
os.chdir(os.path.join(os.environ.get('HOME'),'VAEgen/scripts'))
os.environ["USER_PATH"]="/home/users/geleta/VAEgen"
os.environ["IN_PATH"]="/scratch/groups/cdbustam/rita"
os.environ["OUT_PATH"]="/scratch/groups/cdbustam/rita"

# Define colormap
populations = ['EUR', 'EAS', 'AMR', 'SAS', 'AFR', 'OCE', 'WAS']
cmap = matplotlib.cm.get_cmap('rainbow')
colors = [cmap(col) for col in np.linspace(0,1,len(populations))]

# Load data function
def load_data(n_snps, split_type, of_pop=None, n_samples_2_ret=-1):
    ipath = os.path.join(os.environ.get('IN_PATH'), f'data/chr22/prepared/{split_type}')
    h5f = h5py.File(os.path.join(ipath, f'{split_type}{int(n_snps/1000)}K.h5'), 'r')
    snps = h5f['snps'][:].astype(float)
    popis = h5f['populations'][:].astype(int)
    # Filter if specific population
    if of_pop is not None:
        pop_idx = np.where(np.asarray(popis) == of_pop)[0]
        snps = snps[pop_idx,:]
    # Return specific number of samples if needed
    snps = snps[:n_samples_2_ret,:]
    popis = popis[:n_samples_2_ret]
    h5f.close()
    print(f"Data loaded")
    return snps, popis

# Constructor PCA for genomic data
class genPCA:
    def __init__(self, n_components):
        self._constructor = PCA(n_components=n_components)
    
    def fit(self, data):
        # Fits n PCA components to data
        self._constructor.fit(data)
    
    def project(self, data):
        # Projects data to PCA components >> Encoder to n
        return self._constructor.transform(data)

    def expand(self, latent):
        # Expand latent back to original >> Decoder from latent_shape
        # N.B. latent_shape does not have necessarily coincide with n
        reverse = np.dot(latent, gpca._constructor.components_[:latent.shape[1],:]) 
        + gpca._constructor.mean_
        return (reverse > 0.5).astype(int)

# RLE function
def RLE(array):
    val = array[0]
    rle_array, counter = np.array([val]), 0
    for i in array:
        if i == val: 
            counter += 1
            continue
        else:
            rle_array = np.append(rle_array, counter)
            val, counter = i, 1
    return rle_array

# Test the compression performance of PCA with all data
def exp1():
    # Load training data
    tr_snps, tr_popis = load_data(n_snps = 10000, split_type='train')
    # Number of test samples
    totals = 600
    # Number of components to test (powers of 2)
    components_2_test_up_2 = 512
    # Fit PCs to data
    gpca = genPCA(n_components=components_2_test_up_2)
    gpca.fit(tr_snps)
    # Dimension to start from
    curr_dim = 2
    rles_pca = np.empty([0,7])
    while curr_dim <= components_2_test_up_2:
        print(f'TESTING PCA WITH {curr_dim} COMPONENTS')
        rles = []
        for i, pop in enumerate(populations):
            print(f'PROCESSING POPULATION {pop}')
            # Load test data
            snps, popis = load_data(n_snps = 10000, split_type='test', of_pop=i, n_samples_2_ret=totals)
            torch_snps = torch.from_numpy(snps)
            reconstructed_snps = np.empty((0,torch_snps.shape[1]), int)
            # Project snps to latent space
            latent_snps = gpca.project(torch_snps)[:,:curr_dim]
            print(f'Latent SNPs size: {latent_snps.shape}')
            # Expand back to original size
            reconstructed_snps = gpca.expand(latent_snps)
            print(f'\tReconstructed {reconstructed_snps.shape} SNPs')
            # Compute RLE lengths
            rle_lengths_rec = []
            for individual in range(totals):
                if (individual % totals == 0) and (individual != 0): break
                diff = np.abs((reconstructed_snps[individual,:] - snps[individual,:]))
                length = len(RLE(diff.astype(bool)))
                rle_lengths_rec.append(length)
            rles.append(rle_lengths_rec)
            print(f'MEAN RLE LENGTH {np.mean(rles[i])}')
            # Compute size after compression 
            compressed = np.mean(rles[i]) * 16 + curr_dim * 32
            original = 10 * (10 ** 3)
            compression_ratio = compressed / original
            print(f'COMPRESSION RATIO: {compression_ratio}\n')
        
        rles_pca = np.vstack([rles_pca,np.asarray([np.mean(np.asarray(pop)) for pop in rles])])
        curr_dim = curr_dim * 2
    # Store lengths 
    np.save('pca_compression_exp_1.npy', rles_pca)

# Test the compression performance of PCA fitted to specific populations
def exp2():
    # Load training data
    tr_snps, tr_popis = load_data(n_snps = 10000, split_type='train')
    # Number of test samples
    totals = 600
    # Number of components to test (powers of 2)
    components_2_test_up_2 = 512
    # Fit PCs to data of each population
    gpcas = []
    for i, pop in enumerate(populations):
        pop_idx = np.where(np.asarray(tr_popis) == i)[0]
        snps = tr_snps[pop_idx,:]
        print(f'Fitting genPCA for population {pop}...')
        gpca = genPCA(n_components=components_2_test_up_2)
        gpca.fit(snps)
        gpcas.append(gpca)
    # Dimension to start from
    curr_dim = 2
    rles_pca = np.empty([0,7])
    while curr_dim <= components_2_test_up_2:
        rles = []
        for i, pop in enumerate(populations):
            print(f'TESTING ({pop})-PCA WITH {curr_dim} COMPONENTS')
            # Load test data
            snps, popis = load_data(n_snps = 10000, split_type='test', of_pop=i, n_samples_2_ret=totals)
            torch_snps = torch.from_numpy(snps)
            reconstructed_snps = np.empty((0,torch_snps.shape[1]), int)
            # Project snps to latent space
            latent_snps = gpcas[i].project(torch_snps)[:,:curr_dim]
            print(f'Latent SNPs size: {latent_snps.shape}')
            # Expand back to original size
            reconstructed_snps = gpcas[i].expand(latent_snps)
            print(f'\tReconstructed {reconstructed_snps.shape} SNPs')
            # Compute RLE lengths
            rle_lengths_rec = []
            for individual in range(totals):
                if (individual % totals == 0) and (individual != 0): break
                diff = np.abs((reconstructed_snps[individual,:] - snps[individual,:]))
                length = len(RLE(diff.astype(bool)))
                rle_lengths_rec.append(length)
            rles.append(rle_lengths_rec)
            print(f'MEAN RLE LENGTH {np.mean(rles[i])}')
            # Compute size after compression 
            compressed = np.mean(rles[i]) * 16 + curr_dim * 32
            original = 10 * (10 ** 3)
            compression_ratio = compressed / original
            print(f'COMPRESSION RATIO: {compression_ratio}\n')
        
        rles_pca = np.vstack([rles_pca,np.asarray([np.mean(np.asarray(pop)) for pop in rles])])
        curr_dim = curr_dim * 2
    # Store lengths 
    np.save('pca_compression_exp_2.npy', rles_pca)

if __name__ == '__main__':
    exp2()





