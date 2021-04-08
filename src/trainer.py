import os
import time
import json
import yaml
import wandb
import torch
import datetime
import logging
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from parser import create_parser
from models.VAEgen import VAEgen
from models.losses import VAEloss, L1loss
from models.initializers import init_xavier
from utils.loader import loader 
from utils.decorators import timer 
from utils.loggers import progress, latentPCA, saver

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

@timer
def train(model, optimizer, data, hyperparams, stats, summary=None, num=0, only=None):

    #======================== Set data ========================#
    tr_loader, vd_loader, ts_loader = data

    #======================== Set model ========================#
    model['body'].apply(init_xavier)
    wandb.init(project='VAEgen')
    wandb.run.name = summary
    wandb.run.save()

    wandb.watch(model['body'])

    # Set to training mode
    model['body'].train()

    #======================== Prepare stats ========================#
    best_loss = np.inf
    best_epoch = 0
    datalen = len(tr_loader)

    total_vae_loss, total_rec_loss, total_KL_div  = [], [], []
    total_L1_loss, total_zeros_loss, total_ones_loss = [], [], []
    total_compression_ratio = []

    #======================== Start training loop ========================#
    log.info('Training loop starting now ...')
    for epoch in range(hyperparams['epochs']):
        
        ini = time.time()

        epoch_vae_loss, epoch_rec_loss, epoch_KL_div  = [], [], []
        epoch_L1_loss, epoch_zeros_loss, epoch_ones_loss = [], [], []
        epoch_compression_ratio = []
        
        for i, batch in enumerate(tr_loader):

            snps_array = batch[0].to(device)
            labels = batch[1].to(device) if model['architecture'] == 'C-VAE' else None

            snps_reconstruction, latent_mu, latent_logvar = model['body'](snps_array, labels)
            loss, rec_loss, KL_div = VAEloss(snps_array, snps_reconstruction, latent_mu, latent_logvar)
            L1_loss, zeros_loss, ones_loss, compression_ratio = L1loss(snps_array, snps_reconstruction, partial=True, proportion=True)

            # Backpropagation
            optimizer['body'].zero_grad()
            loss.backward()
            
            # One step of the optimizer (using the gradients from backpropagation)
            optimizer['body'].step()

            epoch_vae_loss.append(loss.item())
            epoch_rec_loss.append(rec_loss.item())
            epoch_KL_div.append(KL_div.item())

            epoch_L1_loss.append(L1_loss.item())
            epoch_zeros_loss.append(zeros_loss)
            epoch_ones_loss.append(ones_loss)

            epoch_compression_ratio.append(compression_ratio.item())

            if stats['verbose']:
                progress(
                    current=i+1, total=datalen, time=time.time()-ini,
                    vae_loss=epoch_vae_loss, rec_loss=epoch_rec_loss, KL_div=epoch_KL_div,
                    L1_loss=epoch_L1_loss, zeros_loss=epoch_zeros_loss, ones_loss=epoch_ones_loss,
                    compression_ratio=epoch_compression_ratio
                )

        total_vae_loss.append(np.mean(epoch_vae_loss[-stats['slide']:]))
        total_rec_loss.append(np.mean(epoch_rec_loss[-stats['slide']:]))
        total_KL_div.append(np.mean(epoch_KL_div[-stats['slide']:]))

        total_L1_loss.append(np.mean(epoch_L1_loss[-stats['slide']:]))
        total_zeros_loss.append(np.mean(epoch_zeros_loss[-stats['slide']:]))
        total_ones_loss.append(np.mean(epoch_ones_loss[-stats['slide']:]))

        total_compression_ratio.append(np.mean(epoch_compression_ratio[-stats['slide']:]))

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

        log.info(f"Epoch [{epoch + 1} / {hyperparams['epochs']}] ({time.time()-ini}s) VAE error: {total_vae_loss[-1]}")

        vd_vae_loss, vd_rec_loss, vd_KL_div, vd_L1_loss, vd_zeros_loss, vd_ones_loss = validate(model, vd_loader, epoch, stats['verbose'])

        if (ts_loader is not None) and (epoch % 100 == 0): #and (epoch != 0):
            log.info('Testing...')
            test(model, ts_loader, epoch, only)
        
        if bool(vd_vae_loss[-1] < best_loss):
            saver(
                obj='model', 
                num=num, 
                state={
                    'architecture': model['architecture'],
                    'body': model['body'], 
                    'parallel': model['parallel'],
                    'num_params': model['num_params'],
                    'weights': model['body'].state_dict()
                }
            )
            saver(
                obj='optimizer', 
                num=num, 
                state={
                    'body': optimizer['body'], 
                    'state': optimizer['body'].state_dict()
                }
            )
            best_epoch = epoch + 1
            best_loss = vd_vae_loss[-1]
            saver(
                obj='stats',
                num=num, 
                state={
                    'epoch': epoch + 1,
                    'verbose': stats['verbose'],
                    'slide': stats['slide'],
                    'best_epoch': best_epoch,
                    'best_loss': best_loss,
                    # Training stats:
                    'tr_vae_losses': total_vae_loss,
                    'tr_rec_losses': total_rec_loss,
                    'tr_KL_div': total_KL_div,
                    'tr_L1_losses': total_L1_loss,
                    'tr_zeros_losses': total_zeros_loss,
                    'tr_ones_losses': total_ones_loss,
                    # Validation stats:
                    'vd_vae_losses': vd_vae_loss, 
                    'vd_rec_losses': vd_rec_loss, 
                    'vd_KL_div': vd_KL_div, 
                    'vd_L1_losses': vd_L1_loss, 
                    'vd_zeros_losses': vd_zeros_loss, 
                    'vd_ones_losses': vd_ones_loss
                }
            )
        elif epoch % 10 == 0:
            saver(
                obj='stats',
                num=num, 
                state={
                    'epoch': epoch + 1,
                    'verbose': stats['verbose'],
                    'slide': stats['slide'],
                    'best_epoch': best_epoch,
                    'best_loss': best_loss,
                    # Training stats:
                    'tr_vae_losses': total_vae_loss,
                    'tr_rec_losses': total_rec_loss,
                    'tr_KL_div': total_KL_div,
                    'tr_L1_losses': total_L1_loss,
                    'tr_zeros_losses': total_zeros_loss,
                    'tr_ones_losses': total_ones_loss,
                    # Validation stats:
                    'vd_vae_losses': vd_vae_loss, 
                    'vd_rec_losses': vd_rec_loss, 
                    'vd_KL_div': vd_KL_div, 
                    'vd_L1_losses': vd_L1_loss, 
                    'vd_zeros_losses': vd_zeros_loss, 
                    'vd_ones_losses': vd_ones_loss
                }
            )
    
    print(f'Training finished in {time.time() - ini}s.')

@timer
def validate(model, vd_loader, epoch, verbose):

    # Set to evaluation mode
    model['body'].eval()
    loss = 0

    total_vae_loss, total_rec_loss, total_KL_div  = [], [], []
    total_L1_loss, total_zeros_loss, total_ones_loss = [], [], []
    total_compression_ratio = []
    
    ini = time.time()
    with torch.no_grad():
        log.info('Validating current model...')
        for i, batch in enumerate(vd_loader):

            snps_array = batch[0].to(device)
            labels = batch[1].to(device) if model['architecture'] == 'C-VAE' else None

            snps_reconstruction, latent_mu, latent_logvar = model['body'](snps_array, labels)

            loss, rec_loss, KL_div = VAEloss(snps_array, snps_reconstruction, latent_mu, latent_logvar)
            L1_loss, zeros_loss, ones_loss, compression_ratio = L1loss(snps_array, snps_reconstruction, partial=True, proportion=True)

            total_vae_loss.append(loss.item())
            total_rec_loss.append(rec_loss.item())
            total_KL_div.append(KL_div.item())

            total_L1_loss.append(L1_loss.item())
            total_zeros_loss.append(zeros_loss)
            total_ones_loss.append(ones_loss)

            total_compression_ratio.append(compression_ratio.item())

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

        if verbose:
            progress(
                current=0, total=0, train=False, bar=False, time=time.time()-ini,
                vae_loss=total_vae_loss, rec_loss=total_rec_loss, KL_div=total_KL_div,
                L1_loss=total_L1_loss, zeros_loss=total_zeros_loss, ones_loss=total_ones_loss,
                compression_ratio=total_compression_ratio
            )
        
        return total_vae_loss, total_rec_loss, total_KL_div, total_L1_loss, total_zeros_loss, total_ones_loss

@timer
def test(model, ts_loader, epoch, only=None):

    # Set to evaluation mode
    model['body'].eval()

    original = np.empty((0, model['body'].module.encoder.FCs[0].FC[0].in_features if model['parallel'] else model['body'].encoder.FCs[0].FC[0].in_features), int)
    latent = np.empty((0, model['body'].module.decoder.FCs[0].FC[0].in_features if model['parallel'] else model['body'].decoder.FCs[0].FC[0].in_features), int)
    labels = torch.Tensor([])

    for i, batch in enumerate(ts_loader):
        original = np.vstack((original, batch[0]))
        mu, _ = model['body'].module.encoder(batch[0].to(device)) if model['parallel'] else model['body'].encoder(batch[0].to(device))
        latent = np.vstack((latent, mu.detach().cpu().squeeze(0))).astype(float)
        labels = torch.cat((labels, batch[1].float()))

    fig = latentPCA(original, latent, labels.int(), only=only)
    wandb.log({f"Latent space at epoch {epoch}": fig})
    plt.close('all')

if __name__ == '__main__':

    #======================== Prepare params ========================#
    parser = create_parser()
    args = parser.parse_args()
    with open(args.params, 'r') as f:
        params = yaml.safe_load(f)
    model_params = params['model']
    hyperparams = params['hyperparams']
    ksize=int(model_params['encoder']['input']['size'] / 1000)
    
    #======================== Prepare data ========================#
    log.info('Loading data...')
    tr_loader = loader(
        ipath=os.path.join(os.environ.get('IN_PATH'), 'data/chr22/prepared'),
        batch_size=hyperparams['batch_size'], 
        split_set='train',
        ksize=ksize,
        only=args.only,
        one_hot=model_params['conditional']['num_classes'] if model_params['conditional'] is not None else None
    )

    vd_loader = loader(
        ipath=os.path.join(os.environ.get('IN_PATH'), 'data/chr22/prepared'),
        batch_size=hyperparams['batch_size'], 
        split_set='valid',
        ksize=ksize,
        only=args.only,
        one_hot=model_params['conditional']['num_classes'] if model_params['conditional'] is not None else None
    )
    
    ts_loader = loader(
        ipath=os.path.join(os.environ.get('IN_PATH'), 'data/chr22/prepared'),
        batch_size=hyperparams['batch_size'], 
        split_set='test',
        ksize=ksize,
        only=args.only
    ) if bool(args.evolution) else None
    log.info('Data loaded ++')
    log.info(f"Training set of shape <= {len(tr_loader) * hyperparams['batch_size']}")
    log.info(f"Validation set of shape <= {len(vd_loader) * hyperparams['batch_size']}")
    if bool(args.evolution): log.info(f"Test set of shape <= {len(ts_loader) * hyperparams['batch_size']}")
    
    #======================== Prepare model ========================#
    model = VAEgen(params=model_params)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f'Using device: {device}')
    model_parallel = False
    if torch.cuda.device_count() > 1:
        log.info(f"Using {torch.cuda.device_count()} GPUs")
        model, model_parallel = nn.DataParallel(model), True
    model.to(device)

    # Number of parameters used in the model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'Number of model parameters: {num_params}')

    #======================== Prepare optimizer ========================#
    optimizer = torch.optim.Adam(
        params=model.parameters(), 
        lr=hyperparams['lr'], 
        weight_decay=hyperparams['weight_decay']
    )

    #======================== Start training ========================#
    train(
        model={
            'architecture': 'VAE' if not model_params['conditional'] else 'C-VAE',
            'body': model, 
            'parallel': model_parallel,
            'num_params': num_params
        }, 
        optimizer={
            'body': optimizer,
        },
        data=(tr_loader, vd_loader, ts_loader),
        hyperparams=hyperparams,
        stats={
            'epoch': 0,
            'verbose': bool(args.verbose),
            'slide': 15,
            'best_epoch': 0,
            'best_loss': np.inf,
        },
        summary=args.experiment,
        num=args.num,
        only=args.only
    )
    #======================== End training ========================#
