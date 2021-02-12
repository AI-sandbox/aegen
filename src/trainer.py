import os
import time
import json
import wandb
import torch
import datetime
import argparse
import numpy as np
from models.VAEgen import VAEgen
from models.losses import VAEloss, L1loss
from models.initializers import init_xavier
from utils.loader import loader 
from utils.decorators import timer 
from utils.loggers import progress

parser = argparse.ArgumentParser()
parser.add_argument('--params', 
                        type=str, 
                        default=os.path.join(os.environ.get('OUT_PATH'), 'params.json'), 
                        metavar='JSON_FILE',
                        help='JSON file with parameters and hyperparameters'
                    )
parser.add_argument('--experiment', 
                        type=str, 
                        default=None, 
                        metavar='SUMMARY',
                        help='Summary of the experiment'
                    )
parser.add_argument('--verbose', 
                        type=bool, 
                        default=False, 
                        metavar='BOOL',
                        help='Verbose output indicator'
                    )

def save_checkpoint(state, is_best, filename=os.path.join(os.environ.get('USER_PATH'), 'checkpoints/checkpoint.pt')):
     """Save checkpoint if a new best is achieved"""
     if is_best:
         print ("=> Saving a new best model")
         torch.save(state, filename)  # Save checkpoint
     else:
         print ("=> Loss did not improve")

@timer
def train(model, tr_loader, vd_loader, hyperparams, summary=None, verbose=False):

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

    print('Training loop starting now ...')
    for epoch in range(hyperparams['epochs']):
        
        ini = time.time()

        epoch_vae_loss, epoch_rec_loss, epoch_KL_div  = [], [], []
        epoch_L1_loss, epoch_zeros_loss, epoch_ones_loss = [], [], []
        
        for i, snps_array in enumerate(tr_loader):

            snps_array = snps_array.to(device)

            snps_reconstruction, latent_mu, latent_logvar = model(snps_array)
            loss, rec_loss, KL_div = VAEloss(snps_array, snps_reconstruction, latent_mu, latent_logvar)
            L1_loss, zeros_loss, ones_loss = L1loss(snps_array, snps_reconstruction, partial=True, proportion=True)

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

            if not verbose:
                progress(
                    current=i+1, total=datalen, time=time.time()-ini,
                    vae_loss=epoch_vae_loss, rec_loss=epoch_rec_loss, KL_div=epoch_KL_div,
                    L1_loss=epoch_L1_loss, zeros_loss=epoch_zeros_loss, ones_loss=epoch_ones_loss
                )

        total_vae_loss.append(np.mean(epoch_vae_loss))
        total_rec_loss.append(np.mean(epoch_rec_loss))
        total_KL_div.append(np.mean(epoch_KL_div))

        total_L1_loss.append(np.mean(epoch_L1_loss))
        total_zeros_loss.append(np.mean(epoch_zeros_loss))
        total_ones_loss.append(np.mean(epoch_ones_loss))

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

        print(f"Epoch [{epoch + 1} / {hyperparams['epochs']}] ({time.time()-ini}s) VAE error: {total_vae_loss[-1]}")

        validate(model, vd_loader, epoch, verbose)
        
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
            }, is_best = is_best, filename=os.path.join(os.environ.get('USER_PATH'), f'checkpoints/checkpoint_VAE.pt'))
    
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
    
    ini = time.time()
    with torch.no_grad():
        print('Validating current model...')
        for i, snps_array in enumerate(vd_loader):

            snps_array = snps_array.to(device)

            snps_reconstruction, latent_mu, latent_logvar = model(snps_array)

            loss, rec_loss, KL_div = VAEloss(snps_array, snps_reconstruction, latent_mu, latent_logvar)
            L1_loss, zeros_loss, ones_loss = L1loss(snps_array, snps_reconstruction, partial=True, proportion=True)

            total_vae_loss.append(loss.item())
            total_rec_loss.append(rec_loss.item())
            total_KL_div.append(KL_div.item())

            total_L1_loss.append(L1_loss.item())
            total_zeros_loss.append(zeros_loss)
            total_ones_loss.append(ones_loss)

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

        if not verbose:
            progress(
                current=0, total=0, train=False, bar=False, time=time.time()-ini,
                vae_loss=total_vae_loss, rec_loss=total_rec_loss, KL_div=total_KL_div,
                L1_loss=total_L1_loss, zeros_loss=total_zeros_loss, ones_loss=total_ones_loss
            )

if __name__ == '__main__':

    args = parser.parse_args()
    with open(args.params, 'r') as f:
        params = json.load(f)
    model_params = params['model']
    hyperparams = params['hyperparams']
    
    tr_loader = loader(
        ipath=os.path.join(os.environ.get('IN_PATH'), 'data/prepared/'),
        batch_size=hyperparams['batch_size'], 
        split_set='train'
    )

    vd_loader = loader(
        ipath=os.path.join(os.environ.get('IN_PATH'), 'data/prepared/'),
        batch_size=hyperparams['batch_size'], 
        split_set='valid'
    )
    
    vae = VAEgen(params=model_params)

    train(
        model=vae, 
        tr_loader=tr_loader, 
        vd_loader=vd_loader,
        hyperparams=hyperparams,
        summary=args.experiment,
        verbose=bool(args.verbose),
    )
    