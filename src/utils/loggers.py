import wandb
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

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

def latentPCA(original, latent, labels):

    pops = ['EUR', 'EAS', 'AMR', 'SAS', 'AFR', 'OCE', 'WAS']
    colors = ['#d23be7','#4355db','#34bbe6','#49da9a','#49da9a','#f7d038','#f7d038']
    colors = sns.color_palette("rainbow", len(pops))

    pca = PCA(n_components=2)
    projected_original = pca.fit_transform(original)
    projected_latent = pca.fit_transform(latent)

    fig, ax = plt.subplots(1,2, figsize=(20, 8))
    ax[0].scatter(projected_original[:,0], projected_original[:,1], c=[colors[x] for x in labels], marker='.')
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    patches = []
    for i, pop in enumerate(pops):
        patches.append(mpatches.Patch(color=colors[i], label=pop))
    ax[0].legend(handles=patches, bbox_to_anchor=(0. ,0.80 ,1.,0.3),loc=10,ncol=7,)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)
    ax[0].set_title(f'(Original) PCA with {projected_original.shape[1]}K SNPs')

    ax[1].scatter(projected_latent[:,0], projected_latent[:,1], c=[colors[x] for x in labels], marker='.')
    ax[1].set_xlabel('PC1')
    ax[1].set_ylabel('PC2')
    patches = []
    for i, pop in enumerate(pops):
        patches.append(mpatches.Patch(color=colors[i], label=pop))
    ax[1].legend(handles=patches, bbox_to_anchor=(0. ,0.80 ,1.,0.3),loc=10,ncol=7,)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['top'].set_visible(False)
    ax[1].set_title(f'Latent space of VAE({projected_latent.shape[1]}) projected with PCA')
    
    return wandb.Image(fig)

    
