import os
import sys
import h5py
import yaml
import torch
import allel
import numpy as np
import pandas as pd
import torch.nn as nn
from scipy import linalg
from adjustText import adjust_text
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

# Load trained VAEs 
sys.path.insert(0, '/home/users/geleta/VAEgen/src')
vaes = []
ini_idx, end_idx = 20, 28
for pop in range(ini_idx, end_idx + 1):
    model = torch.load(f'/scratch/groups/cdbustam/rita/experiments/exp{pop}/VAEgen_weights_{pop}.pt')
    vae = model['body']
    vae.load_state_dict(model['weights'])
    if model['parallel']: vae = nn.DataParallel(vae)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = vae.to(device)
    vae.eval()
    vaes.append(vae)

# Sign correction to ensure deterministic output from SVD.
# Adjusts the columns of u and the rows of v such that the loadings in the
# columns in u that are largest in absolute value are always positive.
def svd_flip(u, v):
    max_abs_cols = np.argmax(np.abs(u), axis=0)
    signs = np.sign(u[max_abs_cols, range(u.shape[1])])
    u *= signs
    v *= signs[:, np.newaxis]
    return u, v

# Read the reference file and filter by default criteria of single_ancestry=1
def filter_reference_file(ref_file_path, verbose=True):
    ref_sample_map = pd.read_csv(ref_file_path, sep="\t")
    ref_sample_map['ref_idx'] = ref_sample_map.index
    ref_sample_map = ref_sample_map[ref_sample_map['Single_Ancestry']==1].reset_index(drop=True)
    if verbose:
        print(f"Total {len(ref_sample_map)} number of samples selected")
    return ref_sample_map

# Reads the sample map and returns an tsv file with 
# sample ID: unique sample ID
# ref_idx: index from reference file 
# superpop: superpop out of the 7 continents, range 0-6
# granularpop: granular ancestries, range 0-135
def get_sample_map(sample_map):
    granular_pop_arr = sample_map['Population'].unique()
    granular_pop_dict = {k:v for k,v in zip(granular_pop_arr, range(len(granular_pop_arr)))}

    pop = ['EUR', 'EAS', 'AMR', 'SAS', 'AFR', 'OCE', 'WAS']
    superpop_dict = {k:v for k,v in zip(pop, range(len(pop)))}

    pop_sample_map = sample_map.loc[:,['Sample','ref_idx']]
    pop_sample_map['granular_pop'] = list(map(lambda x:granular_pop_dict[x], sample_map['Population'].values))
    pop_sample_map['superpop'] = list(map(lambda x:superpop_dict[x], sample_map['Superpopulation code'].values))

    return pop_sample_map, granular_pop_dict, superpop_dict

# Function to reverse a dictionary
def reverse_dict(d):
    new = {}
    for k,v in d.items(): new[v]=k
    return new

# Read VCF file of founders
vcf_master = allel.read_vcf(os.path.join(os.environ.get('IN_PATH'),'data/chr22/ref_final_beagle_phased_1kg_hgdp_sgdp_chr22_hg19.vcf.gz'))
# Create mapfile
mapfile = filter_reference_file(os.path.join(os.environ.get('IN_PATH'),'data/reference_panel_metadata.tsv'))
# Get label mappings and inverse mappings
pop_sample_map, gpop_2_idx, superpop_2_idx = get_sample_map(mapfile)
idx_2_gpop, idx_2_superpop = reverse_dict(gpop_2_idx), reverse_dict(superpop_2_idx)
# Store SNPs and labels to plot
snps, labels = [], []
for i, individual in enumerate(vcf_master['samples']):
    data = pop_sample_map[pop_sample_map.Sample == individual]
    if data.size > 0:
        labels.append([data.superpop.values[0], data.granular_pop.values[0]])
        labels.append([data.superpop.values[0], data.granular_pop.values[0]])
        # Maternal as well as paternal
        maternal = vcf_master["calldata/GT"][:,i,0]
        paternal = vcf_master["calldata/GT"][:,i,1]
        snps.append(maternal)
        snps.append(paternal)
snps, labels = np.asarray(snps), np.asarray(labels)

# Plot the latent space of tha VAE
# with id=vae_trained_index
# min_individuals is the minimum number of individuals
# that a population has to have in order to be plotted
# in the chart
# vae_trained_index in 0...6 is reserved to 7VAE models
# normal VAEs must have vae_trained_index > 6
def plot_latent_space(vae_trained_index, min_individuals=40):
    fig, ax = plt.subplots(1, 1, figsize=(10,10))
    coords=[]
    populations = list(map(lambda x: idx_2_superpop[x], np.unique(labels[:,0])))
    print(populations)
    selected = range(len(labels[:,0])) # np.where(labels[:,0] == vae_trained_index)[0] if vae_trained_index<7 else range(len(labels[:,0]))
    print(selected)
    # Select populations satisfying the min_individuals constraint
    aux1 = np.bincount(labels[selected][:,1])
    aux2 = np.nonzero(aux1)[0]
    subpops_with_minimum = aux2[aux1[aux2]>min_individuals]
    selected = np.where(np.isin(labels[:,1], subpops_with_minimum))[0]
    
    # Number of SNPs is set to default 10K
    torch_snps = torch.from_numpy(snps[selected,:10000])
    torch_labels = labels[selected]
    subpopulations_pop = list(map(lambda x: idx_2_gpop[x], np.unique(torch_labels[:,-1])))
    print(f'Total shape: {torch_snps.shape}')
    
    # Map SNPs to latent space of VAE
    #for i in range(torch_snps.shape[0]):
    #    if i % 10000 == 0: 
            #if vae_trained_index <7:
            #    print(f'Processed {i} samples from pop {populations[vae_trained_index]}.')
            #else:
            #print(f'Processed {i} samples from all populations.')
        #input = torch_snps[i,:].reshape(1,-1)
    mu, logvar = vaes[vae_trained_index].encoder(torch_snps.float().cuda())
    #mu, logvar = vae.encoder(input.float().cuda())
    # z = reparametrize(mu, logvar)
    # coords.append(z[0].cpu().detach().numpy())
    # coords.append(mu[0].cpu().detach().numpy())
    print(f'Mu size: {mu.shape}')
    
    
    # Latent coordinates
    #coords = np.asarray(coords)
    coords = mu.cpu().detach().numpy()
    print(coords.shape)
    # No centering
    U, S, V = linalg.svd(coords, full_matrices=False)
    U, V = svd_flip(U, V)
    projected_latent = U[:, :2]

    # Define colormap
    cmap = matplotlib.cm.get_cmap('rainbow')
    # colors = [cmap(col) for col in np.linspace(0,1,len(subpopulations_pop))]
    colors = [cmap(col) for col in np.linspace(0,1,len(populations))]

    patches = []
    texts = []
    # Start plotting each subpopulation
    for i, pop in enumerate(subpopulations_pop):
        pop_idx = np.where(torch_labels[:,1] == gpop_2_idx[pop])[0]
        centroid = np.mean(projected_latent[pop_idx,:], axis=0)
        ax.scatter(
            projected_latent[pop_idx,0],
            projected_latent[pop_idx,1], 
            color=colors[torch_labels[pop_idx,0][0]], # colors[i], 
            label=pop,
            alpha=0.75, # 0.75 if vae_trained_index == i else 0.3, 
            marker='.',
            zorder=2 # 10 if vae_trained_index == i else 2
        )
        #ax.annotate(pop, centroid, zorder=12)
        texts.append(plt.text(centroid[0], centroid[1], pop, ha='center', va='center', zorder=12))
        for j in pop_idx:
            ax.plot(
                [projected_latent[j,0],centroid[0]], 
                [projected_latent[j,1],centroid[1]], 
                c=colors[torch_labels[pop_idx,0][0]], # colors[i],
                alpha=0.3,
                zorder=1
            )
    # Adjust the texts of the subpopulation labels
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'), ax=ax)
    # Set patches for the legend
    for i, pop in enumerate(populations):
        patches.append(mpatches.Patch(color=colors[i], label=pop))
    # Set plot title
    ax.set_title(
        # f'SVD of z from VAE trained on {populations[vae_trained_index]}' if vae_trained_index < 7 else
        f'SVD of z from VAE trained on all populations'
    )
    # Final settings
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.axhline(y=0, color="black", linestyle="--")
    ax.axvline(x=0, color="black", linestyle="--")
    ax.set_xlabel('U1')
    ax.set_ylabel('U2')
    # if vae_trained_index != 7: 
    ax.legend(handles=patches, bbox_to_anchor=(0. ,0.80 ,1.9,0.1),loc=10,ncol=1,)
    plt.show();
    return fig

if __name__ == '__main__':
    # 2 dimensions
    _ = plot_latent_space(0, min_individuals=40)

