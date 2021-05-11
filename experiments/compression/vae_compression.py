import os
import sys
import h5py
import yaml
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
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

# One hot encoding function for C-VAE
def one_hot(labels, num_classes):
    targets = torch.zeros(labels.shape[0], num_classes)
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return Variable(targets)

# Test the compression performance of VAE with data from all populations
def exp1():
    # Number of test samples
    totals = 600
    # Number of components to test (powers of 2)
    components_2_test_up_2 = 512
    # Dimension to start from
    curr_dim, idx_dim = 2, 29
    rles_pca = np.empty([0,7])
    while curr_dim <= components_2_test_up_2:
        print(f'TESTING VAE WITH LATENT SPACE dim:{curr_dim}')
        rles = []
        for i, pop in enumerate(populations):
            print(f'PROCESSING POPULATION {pop}')
            # Load test data
            snps, popis = load_data(n_snps = 10000, split_type='test', of_pop=i, n_samples_2_ret=totals)
            torch_snps = torch.from_numpy(snps)
            reconstructed_snps = np.empty((0,torch_snps.shape[1]), int)
            # Prepare VAE
            sys.path.insert(0, '/home/users/geleta/VAEgen/src')
            model = torch.load(f'/scratch/groups/cdbustam/rita/experiments/exp{idx_dim}/VAEgen_weights_{idx_dim}.pt')
            vae = model['body']
            vae.load_state_dict(model['weights'])
            if model['parallel']: vae = nn.DataParallel(vae)
            device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            vae = vae.to(device)
            vae.eval()
            print(f'VAE{idx_dim} at {device}')
            # Forward through VAE
            o, _, _ = vae(torch_snps.float().cuda())
            reconstructed_snps = (o.detach().cpu().squeeze(0) > 0.5)
            print(f'\tReconstructed {reconstructed_snps.shape} SNPs')
            # Compute RLE lengths
            rle_lengths_rec = []
            for individual in range(totals):
                if (individual % totals == 0) and (individual != 0): break
                diff = np.abs((reconstructed_snps[individual,:] - snps[individual,:]).numpy())
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
        curr_dim, idx_dim = curr_dim * 2, idx_dim + 1
    # Store lengths 
    np.save('vae_compression_exp_1.npy', rles_pca)

# Test the compression performance of VAE with data from 
# all populations conditioned on ancestry
def exp1():
    # Number of test samples
    totals = 600
    # Number of components to test (powers of 2)
    components_2_test_up_2 = 512
    # Dimension to start from
    curr_dim, idx_dim = 2, 38
    rles_pca = np.empty([0,7])
    while curr_dim <= components_2_test_up_2:
        print(f'TESTING VAE WITH LATENT SPACE dim:{curr_dim}')
        rles = []
        for i, pop in enumerate(populations):
            print(f'PROCESSING POPULATION {pop}')
            # Load test data
            snps, popis = load_data(n_snps = 10000, split_type='test', of_pop=i, n_samples_2_ret=totals)
            popis = one_hot(popis, len(populations))
            torch_snps = torch.from_numpy(snps)
            reconstructed_snps = np.empty((0,torch_snps.shape[1]), int)
            # Prepare VAE
            sys.path.insert(0, '/home/users/geleta/VAEgen/src')
            model = torch.load(f'/scratch/groups/cdbustam/rita/experiments/exp{idx_dim}/VAEgen_weights_{idx_dim}.pt')
            vae = model['body']
            vae.load_state_dict(model['weights'])
            if model['parallel']: vae = nn.DataParallel(vae)
            device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            vae = vae.to(device)
            vae.eval()
            print(f'VAE{idx_dim} at {device}')
            # Forward through VAE
            o, _, _ = vae(torch_snps.float().cuda(), popis.float().cuda())
            reconstructed_snps = (o.detach().cpu().squeeze(0) > 0.5)
            print(f'\tReconstructed {reconstructed_snps.shape} SNPs')
            # Compute RLE lengths
            rle_lengths_rec = []
            for individual in range(totals):
                if (individual % totals == 0) and (individual != 0): break
                diff = np.abs((reconstructed_snps[individual,:] - snps[individual,:]).numpy())
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
        curr_dim, idx_dim = curr_dim * 2, idx_dim + 1
    # Store lengths 
    np.save('vae_compression_exp_2.npy', rles_pca)

if __name__ == '__main__':
    exp1()
