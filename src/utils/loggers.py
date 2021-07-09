import os
import psutil
import wandb
import torch
import numpy as np
import seaborn as sns
import logging
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def progress(current, total, train=True, bar=True, time=None, **kwargs):

    if train:
        indicator = f'Progress: [{current} / {total}]'
    else:
        indicator = f'Validation:'
        
    bar_progress = ''
    metrics = ''

    if bar:
        bar_progress += '=' * (current) + '-' * (total - current) 
    
    if time is not None:
        bar_progress += f'({np.round(time, 2)}s)'

    for k, v in kwargs.items():
        metrics += f' {k}: {np.round(np.mean(v), 2)}'
    
    print(f'{indicator} {bar_progress} {metrics}')

def latentPCA(original, latent, labels, only=None):

    populations = ['EUR', 'EAS', 'AMR', 'SAS', 'AFR', 'OCE', 'WAS']
    colors = sns.color_palette("rainbow", len(populations))
    pops = dict(zip(populations, colors))
    pop_mapper = dict(zip(np.arange(0, len(pops.keys())), pops.keys()))

    pca = PCA(n_components=2)
    projected_original = pca.fit_transform(original)
    projected_latent = pca.fit_transform(latent)

    fig, ax = plt.subplots(1,2, figsize=(20, 8))
    ax[0].scatter(projected_original[:,0], projected_original[:,1], c=[colors[x] if only is None else pops[pop_mapper[only]] for x in labels] , marker='.')
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    patches = []
    for i, pop in enumerate(pops.keys()):
        patches.append(mpatches.Patch(color=colors[i], label=pop))
    ax[0].legend(handles=patches, bbox_to_anchor=(0. ,0.80 ,1.,0.3),loc=10,ncol=7,)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].set_title(f'(Original) PCA with {original.shape[1]}K SNPs')

    ax[1].scatter(projected_latent[:,0], projected_latent[:,1], c=[colors[x] if only is None else pops[pop_mapper[only]] for x in labels], marker='.')
    ax[1].set_xlabel('PC1')
    ax[1].set_ylabel('PC2')
    patches = []
    for i, pop in enumerate(pops.keys()):
        patches.append(mpatches.Patch(color=colors[i], label=pop))
    ax[1].legend(handles=patches, bbox_to_anchor=(0. ,0.80 ,1.,0.3),loc=10,ncol=7,)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].set_title(f'Latent space of VAE({latent.shape[1]}) projected with PCA')
    
    return wandb.Image(fig)

def saver(obj, num, state):

    if obj == 'model':
        log.info('Storing best weights.')
        torch.save(state, os.path.join(os.environ.get('OUT_PATH'), f'experiments/exp{num}/VAEgen_weights_{num}.pt'))
    elif obj == 'optimizer':
        log.info('Storing optimizer state.')
        torch.save(state, os.path.join(os.environ.get('OUT_PATH'), f'experiments/exp{num}/OPT_state_{num}.pt'))
    elif obj == 'stats':
        log.info('Storing stats.')
        torch.save(state, os.path.join(os.environ.get('OUT_PATH'), f'experiments/exp{num}/VAEgen_stats_{num}.pt'))
    else: 
        log.error('Unknown object to store. Exiting...')
        exit(1)
        
def system_info():
    log.info('\n\n'+'='*50)
    log.info(f'Number of physical cores: {psutil.cpu_count(logical=False)}')
    log.info(f'Current system-wide CPU utilization: {psutil.cpu_percent()}%')
    mem = psutil.virtual_memory()
    log.info(f'Total physical memory (exclusive swap): {psutil.virtual_memory().total // (2**30)} GB')
    log.info(f'Total available memory: {psutil.virtual_memory().available // (2**30)} GB')
    log.info(f'Percentage of used RAM: {psutil.virtual_memory().percent}%')
    log.info(f'Percentage of available memory: {psutil.virtual_memory().available * 100 / psutil.virtual_memory().total}%')
    log.info('='*50+'\n\n')
    