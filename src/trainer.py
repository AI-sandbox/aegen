import os
import time
import json
import yaml
import wandb
import torch
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
from models.VAEgen import VAEgen
from models.losses import VAEloss, L1loss
from models.initializers import init_xavier
from utils.loader import loader 
from utils.decorators import timer 
from utils.loggers import progress, latentPCA

parser = argparse.ArgumentParser()
parser.add_argument('--params', 
                        type=str, 
                        default=os.path.join(os.environ.get('USER_PATH'), 'params.yaml'), 
                        metavar='YAML_FILE',
                        help='YAML file with parameters and hyperparameters'
                    )
parser.add_argument('--experiment', 
                        type=str, 
                        default=None, 
                        metavar='SUMMARY',
                        help='Summary of the experiment'
                    )
parser.add_argument('--num', 
                        type=int, 
                        default=0, 
                        metavar='NUMBER',
                        help='NUMBER of the experiment'
                    )                 
parser.add_argument('--verbose', 
                        type=bool, 
                        default=False, 
                        metavar='BOOL',
                        help='Verbose output indicator'
                    )

parser.add_argument('--evolution', 
                        type=bool, 
                        default=False, 
                        metavar='BOOL',
                        help='Evolution plots at wandb'
                    )

def save_checkpoint(state, is_best, filename=os.path.join(os.environ.get('USER_PATH'), 'checkpoints/checkpoint.pt')):
     """Save checkpoint if a new best is achieved"""
     if is_best:
         print ("=> Saving a new best model")
         torch.save(state, filename)  # Save checkpoint
     else:
         print ("=> Loss did not improve")

@timer
def train(model, tr_loader, vd_loader, hyperparams, summary=None, num=0, verbose=False, slide=50, ts_loader=None):

    model.apply(init_xavier)
    wandb.init(project='VAEgen')
    wandb.run.name = summary
    wandb.run.save()

    wandb.watch(model)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=hyperparams['lr'], weight_decay=hyperparams['weight_decay'])

    # This is the number of parameters used in the model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of model parameters: {num_params}')

    # Set to training mode
    model.train()

    best_loss = np.inf
    datalen = len(tr_loader)

    total_vae_loss, total_rec_loss, total_KL_div  = [], [], []
    total_L1_loss, total_zeros_loss, total_ones_loss = [], [], []
    total_compression_ratio = []

    print('Training loop starting now ...')
    for epoch in range(hyperparams['epochs']):
        
        ini = time.time()

        epoch_vae_loss, epoch_rec_loss, epoch_KL_div  = [], [], []
        epoch_L1_loss, epoch_zeros_loss, epoch_ones_loss = [], [], []
        epoch_compression_ratio = []
        
        for i, batch in enumerate(tr_loader):

            snps_array = batch[0].to(device)

            snps_reconstruction, latent_mu, latent_logvar = model(snps_array)
            loss, rec_loss, KL_div = VAEloss(snps_array, snps_reconstruction, latent_mu, latent_logvar)
            L1_loss, zeros_loss, ones_loss, compression_ratio = L1loss(snps_array, snps_reconstruction, partial=True, proportion=True)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # One step of the optimizer (using the gradients from backpropagation)
            optimizer.step()

            epoch_vae_loss.append(loss.item())
            epoch_rec_loss.append(rec_loss.item())
            epoch_KL_div.append(KL_div.item())

            epoch_L1_loss.append(L1_loss.item())
            epoch_zeros_loss.append(zeros_loss)
            epoch_ones_loss.append(ones_loss)

            epoch_compression_ratio.append(compression_ratio)

            if not verbose:
                progress(
                    current=i+1, total=datalen, time=time.time()-ini,
                    vae_loss=epoch_vae_loss, rec_loss=epoch_rec_loss, KL_div=epoch_KL_div,
                    L1_loss=epoch_L1_loss, zeros_loss=epoch_zeros_loss, ones_loss=epoch_ones_loss,
                    compression_ratio=epoch_compression_ratio
                )

        total_vae_loss.append(np.mean(epoch_vae_loss[-slide:]))
        total_rec_loss.append(np.mean(epoch_rec_loss[-slide:]))
        total_KL_div.append(np.mean(epoch_KL_div[-slide:]))

        total_L1_loss.append(np.mean(epoch_L1_loss[-slide:]))
        total_zeros_loss.append(np.mean(epoch_zeros_loss[-slide:]))
        total_ones_loss.append(np.mean(epoch_ones_loss[-slide:]))

        total_compression_ratio.append(np.mean(epoch_compression_ratio[-slide:]))

        wandb.log({
            'tr_VAE_loss': total_vae_loss[-1],
            'tr_rec_loss': total_rec_loss[-1],
            'tr_KL_div': total_KL_div[-1],
        })
		
        wandb.log({
            'tr_L1_loss': total_L1_loss[-1],
            'tr_zeros_loss': total_zeros_loss[-1],
            'tr_ones_loss': total_ones_loss[-1],
        })

        wandb.log({
            'compression_ratio': total_compression_ratio[-1],
        })

        print(f"Epoch [{epoch + 1} / {hyperparams['epochs']}] ({time.time()-ini}s) VAE error: {total_vae_loss[-1]}")

        validate(model, vd_loader, epoch, verbose)

        if (ts_loader is not None) and (epoch % 50 == 0): #and (epoch != 0):
            print('Testing...')
            test(model, ts_loader, epoch)
        
        is_best = bool(total_vae_loss[-1] < best_loss)
        if is_best:
            best_loss = total_vae_loss[-1]
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'beta': hyperparams['beta'],
                'lr': hyperparams['lr'],
                'vae_losses': total_vae_loss,
                'rec_losses': total_rec_loss,
                'kl_div': total_KL_div,
                'l1_losses': total_L1_loss,
                'zeros_losses': total_zeros_loss,
                'ones_losses': total_ones_loss,
                'compression_ratio': total_compression_ratio
            }, is_best = is_best, filename=os.path.join(os.environ.get('USER_PATH'), f'checkpoints/checkpoint_VAE_{num}.pt'))
    
    print(f'Training finished in {time.time() - ini}s.')

@timer
def validate(model, vd_loader, epoch, verbose=False):

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    model.to(device)

    # Set to evaluation mode
    model.eval()
    loss = 0

    total_vae_loss, total_rec_loss, total_KL_div  = [], [], []
    total_L1_loss, total_zeros_loss, total_ones_loss = [], [], []
    total_compression_ratio = []
    
    ini = time.time()
    with torch.no_grad():
        print('Validating current model...')
        for i, batch in enumerate(vd_loader):

            snps_array = batch[0].to(device)

            snps_reconstruction, latent_mu, latent_logvar = model(snps_array)

            loss, rec_loss, KL_div = VAEloss(snps_array, snps_reconstruction, latent_mu, latent_logvar)
            L1_loss, zeros_loss, ones_loss, compression_ratio = L1loss(snps_array, snps_reconstruction, partial=True, proportion=True)

            total_vae_loss.append(loss.item())
            total_rec_loss.append(rec_loss.item())
            total_KL_div.append(KL_div.item())

            total_L1_loss.append(L1_loss.item())
            total_zeros_loss.append(zeros_loss)
            total_ones_loss.append(ones_loss)

            total_compression_ratio.append(compression_ratio)

            wandb.log({
                'vd_VAE_loss': total_vae_loss[-1],
                'vd_rec_loss': total_rec_loss[-1],
                'vd_KL_div': total_KL_div[-1],
            })
            
            wandb.log({
                'vd_L1_loss': total_L1_loss[-1],
                'vd_zeros_loss': total_zeros_loss[-1],
                'vd_ones_loss': total_ones_loss[-1],
            })

            wandb.log({
                'vd_compression_ratio': total_compression_ratio[-1],
            })

        if not verbose:
            progress(
                current=0, total=0, train=False, bar=False, time=time.time()-ini,
                vae_loss=total_vae_loss, rec_loss=total_rec_loss, KL_div=total_KL_div,
                L1_loss=total_L1_loss, zeros_loss=total_zeros_loss, ones_loss=total_ones_loss,
                compression_ratio=total_compression_ratio
            )

@timer
def test(model, ts_loader, epoch):

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    model.to(device)

    # Set to evaluation mode
    model.eval()

    original, latent, labels = np.empty((0, model.encoder.FCs[0].FC[0].in_features), int), np.empty((0, model.decoder.FCs[0].FC[0].in_features), int), torch.Tensor([])
    for i, batch in enumerate(ts_loader):
        original = np.vstack((original, batch[0]))
        mu, _ = vae.encoder(batch[0].to(device))
        latent = np.vstack((latent, mu.detach().cpu().squeeze(0)))
        labels = torch.cat((labels, batch[1]))

    fig = latentPCA(original, latent, labels.int())
    wandb.log({f"Latent space at epoch {epoch}": fig})
    plt.close('all')

if __name__ == '__main__':

    args = parser.parse_args()
    with open(args.params, 'r') as f:
        params = yaml.safe_load(f)
    model_params = params['model']
    hyperparams = params['hyperparams']
    ksize=int(model_params['encoder']['input']['size'] / 1000)
    
    tr_loader = loader(
        ipath=os.path.join(os.environ.get('IN_PATH'), 'data/chr22/prepared'),
        batch_size=hyperparams['batch_size'], 
        split_set='train',
        ksize=ksize
    )

    vd_loader = loader(
        ipath=os.path.join(os.environ.get('IN_PATH'), 'data/chr22/prepared'),
        batch_size=hyperparams['batch_size'], 
        split_set='valid',
        ksize=ksize
    )
    
    ts_loader = loader(
        ipath=os.path.join(os.environ.get('IN_PATH'), 'data/chr22/prepared'),
        batch_size=hyperparams['batch_size'], 
        split_set='test',
        ksize=ksize
    ) if bool(args.evolution) else None
    
    vae = VAEgen(params=model_params)

    train(
        model=vae, 
        tr_loader=tr_loader, 
        vd_loader=vd_loader,
        hyperparams=hyperparams,
        summary=args.experiment,
        num=args.num,
        verbose=bool(args.verbose),
        slide=15,
        ts_loader=ts_loader
    )