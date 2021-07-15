import os
import gc
import time
import wandb
import torch
import psutil
import logging
import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models.losses import VAEloss, L1loss
from models.initializers import init_xavier
from utils.decorators import timer 
from utils.loggers import progress, latentPCA, saver, system_info
from utils.simulators import OnlineSimulator

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

@timer
def train(model, optimizer, hyperparams, stats, tr_loader, vd_loader, ts_loader, device='cpu', summary=None, num=0, only=None, monitor=None, metadata=None):

    #======================== Set model ========================#
    log.info('Setting model ...')
    model['body'].apply(init_xavier)
    if monitor == 'wandb':
        log.info('Initializing wandb ...')
        system_info()
        log.info('Setting environ variable ...')
        os.environ["WANDB_START_METHOD"] = "thread"
        wandb.init(
            project='AEgen',
            dir=os.path.join(os.environ.get('OUT_PATH'), f'experiments/'),
            # resume='allow',
        )
        wandb.run.name = summary
        wandb.run.save()
        log.info('Wandb set.')
        wandb.watch(model['body'])
    system_info()
    log.info('Setting training mode ...')
    # Set to training mode
    model['body'].train()
    log.info(model)
    log.info('Model set.')
    #======================== Prepare stats ========================#
    log.info('Preparing stats ...')
    best_loss = np.inf
    best_epoch = 0

    ## TODO: a better way to store metrics.
    total_vae_loss, total_rec_loss, total_KL_div  = [], [], []
    total_L1_loss, total_zeros_loss, total_ones_loss = [], [], []
    total_compression_ratio = []
    if model['imputation']: total_imputation_L1_loss = []
    log.info('Stats prepared.')
    #======================== Define simulation type ========================#
    ## If training simulation is offline just traverse the TR dataloader.
    if hyperparams['training']['simulation'] == 'offline':
        offline = True
        datalen = len(tr_loader)
    ## Elif training simulation is online, instantiate an OnlineSimulation
    ## constructor with the simulation parameters.
    ## OnlineSimulation will simulate a data batch when .simulate() is 
    ## called upon.
    elif hyperparams['training']['simulation'] == 'online':
        offline = False
        datalen = hyperparams['training']['n_batches']
        batch_counter = [None] * datalen
        log.info('Initializating online simulator...')
        assert(model_params['num_classes'] == vd_metadata['n_populations'])
        online_simulator = OnlineSimulator(
            batch_size = hyperparams['batch_size'],
            n_populations = model_params['num_classes'],
            mode = hyperparams['training']['mode'],
            balanced = hyperparams['training']['balanced'],
            device = hyperparams['training']['device']
        )
        log.info('Online simulation initialized.')
    else: raise Exception('Simulation can be either [online, offline].')
    #======================== Start training loop ========================#
    log.info('Training loop starting now ...')                              
    for epoch in range(hyperparams['epochs']):
        
        ini = time.time()

        ## TODO: a better way to store metrics.
        epoch_vae_loss, epoch_rec_loss, epoch_KL_div  = [], [], []
        epoch_L1_loss, epoch_zeros_loss, epoch_ones_loss = [], [], []
        epoch_compression_ratio = []
        if model['imputation']: epoch_imputation_L1_loss = []
        
        for i, batch in enumerate(tr_loader if offline else batch_counter):

            if offline:
                snps_array = batch[0].to(device)#.unsqueeze(1)
                labels = batch[1].to(device) if model['conditional'] else None
            else: 
                snps_array, labels = online_simulator.simulate()
                snps_array, labels = snps_array.to(device), labels.to(device)

            if model['imputation']:
                snps_reconstruction, latent_mu, latent_logvar, mask = model['body'](snps_array, labels)
                imputation_L1_loss = F.l1_loss((snps_reconstruction[mask] > 0.5).float(), snps_array[mask], reduction='mean') * 100
            else:
                latent_mu, latent_logvar = model['body'].encoder(snps_array, labels)
                z = model['body'].reparametrize(latent_mu, latent_logvar)
                snps_reconstruction = model['body'].decoder(z, labels)
            
            if model['shape'] == 'window-based':
                latent_mu = torch.cat(latent_mu, axis=-1)
                latent_logvar = torch.cat(latent_logvar, axis=-1)
            
            ## Compute losses and metrics.
            loss, rec_loss, KL_div = VAEloss(snps_array, snps_reconstruction, latent_mu, latent_logvar)
            L1_loss, zeros_loss, ones_loss, compression_ratio = L1loss(snps_array, snps_reconstruction, partial=True, proportion=True)
            
            ## Clear memory and caches.
            del snps_array
            del labels
            del latent_mu
            del latent_logvar
            del z
            del snps_reconstruction
            gc.collect()
            torch.cuda.empty_cache()

            # Backpropagation.
            optimizer['body'].zero_grad()
            loss.backward()
            
            # One step of the optimizer (using the gradients from backpropagation).
            optimizer['body'].step()

            epoch_vae_loss.append(loss.item())
            epoch_rec_loss.append(rec_loss.item())
            epoch_KL_div.append(KL_div.item())

            epoch_L1_loss.append(L1_loss.item())
            epoch_zeros_loss.append(zeros_loss)
            epoch_ones_loss.append(ones_loss)

            epoch_compression_ratio.append(compression_ratio.item())
            if model['imputation']: epoch_imputation_L1_loss.append(imputation_L1_loss.item())

            if stats['verbose']:
                progress(
                    current=i+1, total=datalen, time=time.time()-ini,
                    vae_loss=epoch_vae_loss, rec_loss=epoch_rec_loss, KL_div=epoch_KL_div,
                    L1_loss=epoch_L1_loss, zeros_loss=epoch_zeros_loss, ones_loss=epoch_ones_loss,
                    compression_ratio=epoch_compression_ratio, imputation_L1_loss=epoch_imputation_L1_loss if model['imputation'] else [0]
                )

        total_vae_loss.append(np.mean(epoch_vae_loss[-stats['slide']:]))
        total_rec_loss.append(np.mean(epoch_rec_loss[-stats['slide']:]))
        total_KL_div.append(np.mean(epoch_KL_div[-stats['slide']:]))

        total_L1_loss.append(np.mean(epoch_L1_loss[-stats['slide']:]))
        total_zeros_loss.append(np.mean(epoch_zeros_loss[-stats['slide']:]))
        total_ones_loss.append(np.mean(epoch_ones_loss[-stats['slide']:]))

        total_compression_ratio.append(np.mean(epoch_compression_ratio[-stats['slide']:]))
        if model['imputation']: total_imputation_L1_loss.append(np.mean(epoch_imputation_L1_loss[-stats['slide']:]))
            
        if optimizer['scheduler'] is not None: optimizer['scheduler'].step(np.mean(epoch_vae_loss[-stats['slide']:]))

        if monitor == 'wandb':
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

            if model['imputation']:
                wandb.log({
                    'tr_imputation_L1_loss': total_imputation_L1_loss[-1],
                })

        log.info(f"Epoch [{epoch + 1} / {hyperparams['epochs']}] ({time.time()-ini}s) VAE error: {total_vae_loss[-1]}")
        
        ## Validate according validation scheduler.
        if (epoch % hyperparams['validation']['scheduler'] == 0):
            
            if model['imputation']: 
                vd_vae_loss, vd_rec_loss, vd_KL_div, vd_L1_loss, vd_zeros_loss, vd_ones_loss, vd_imputation_L1_loss = validate(
                    model, 
                    vd_loader, 
                    epoch, 
                    stats['verbose'], 
                    monitor=monitor, 
                    device=device
                )
            else:
                vd_vae_loss, vd_rec_loss, vd_KL_div, vd_L1_loss, vd_zeros_loss, vd_ones_loss = validate(
                    model, 
                    vd_loader, 
                    epoch, 
                    stats['verbose'], 
                    monitor=monitor,
                    device=device
                )
            
        if bool(vd_vae_loss[-1] < best_loss):
            saver(
                obj='model', 
                num=num, 
                state={
                    'architecture': model['architecture'],
                    'imputation': model['imputation'],
                    #'body': model['body'], 
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
                    'tr_imputation_L1_losses': total_imputation_L1_loss if model['imputation'] else None,
                    # Validation stats:
                    'vd_vae_losses': vd_vae_loss, 
                    'vd_rec_losses': vd_rec_loss, 
                    'vd_KL_div': vd_KL_div, 
                    'vd_L1_losses': vd_L1_loss, 
                    'vd_zeros_losses': vd_zeros_loss, 
                    'vd_ones_losses': vd_ones_loss,
                    'vd_imputation_L1_losses': vd_imputation_L1_loss if model['imputation'] else None,
                }
            )
            
        ## Test according testing scheduler.
        if (ts_loader is not None) and (epoch % hyperparams['testing']['scheduler'] == 0): #and (epoch != 0):
            log.info('Testing...')
            conditional = (model['architecture'] == 'C-VAE')
            test(
                model=model, 
                ts_loader=ts_loader, 
                epoch=epoch, 
                metadata=metadata, 
                only=only, 
                conditional=conditional, 
                monitor=monitor,
                device=device
            )
            
        ## Save checkpoint according checkpointing scheduler.
        if epoch % hyperparams['checkpointing']['scheduler'] == 0:
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
                    'tr_imputation_L1_losses': total_imputation_L1_loss if model['imputation'] else None,
                    # Validation stats:
                    'vd_vae_losses': vd_vae_loss, 
                    'vd_rec_losses': vd_rec_loss, 
                    'vd_KL_div': vd_KL_div, 
                    'vd_L1_losses': vd_L1_loss, 
                    'vd_zeros_losses': vd_zeros_loss, 
                    'vd_ones_losses': vd_ones_loss,
                    'vd_imputation_L1_losses': vd_imputation_L1_loss if model['imputation'] else None,
                }
            )
    
    print(f'Training finished in {time.time() - ini}s.')

@timer
def validate(model, vd_loader, epoch, verbose, monitor=None, device='cpu'):

    # Set to evaluation mode
    model['body'].eval()
    loss = 0

    total_vae_loss, total_rec_loss, total_KL_div  = [], [], []
    total_L1_loss, total_zeros_loss, total_ones_loss = [], [], []
    total_compression_ratio = []
    if model['imputation']: total_imputation_L1_loss = []
    
    ini = time.time()
    with torch.no_grad():
        log.info('Validating current model...')
        for i, batch in enumerate(vd_loader):

            snps_array = batch[0].to(device)
            labels = batch[1].to(device) if model['architecture'] == 'C-VAE' else None

            if model['imputation']:
                snps_reconstruction, latent_mu, latent_logvar, mask = model['body'](snps_array, labels)
                imputation_L1_loss =  F.l1_loss((snps_reconstruction[mask] > 0.5).float(), snps_array[mask], reduction='mean') * 100
            else:
                latent_mu, latent_logvar = model['body'].encoder(snps_array, labels)
                z = model['body'].reparametrize(latent_mu, latent_logvar)
                snps_reconstruction = model['body'].decoder(z, labels)

            if model['shape'] == 'window-based':
                latent_mu = torch.cat(latent_mu, axis=-1)
                latent_logvar = torch.cat(latent_logvar, axis=-1)
            loss, rec_loss, KL_div = VAEloss(snps_array, snps_reconstruction, latent_mu, latent_logvar)
            L1_loss, zeros_loss, ones_loss, compression_ratio = L1loss(snps_array, snps_reconstruction, partial=True, proportion=True)
            
            del snps_array
            del labels
            del latent_mu
            del latent_logvar
            del z
            del snps_reconstruction
            gc.collect()
            torch.cuda.empty_cache()

            total_vae_loss.append(loss.item())
            total_rec_loss.append(rec_loss.item())
            total_KL_div.append(KL_div.item())

            total_L1_loss.append(L1_loss.item())
            total_zeros_loss.append(zeros_loss)
            total_ones_loss.append(ones_loss)

            total_compression_ratio.append(compression_ratio.item())
            if model['imputation']: total_imputation_L1_loss.append(imputation_L1_loss.item())
        
        if monitor == 'wandb':
            wandb.log({
                'vd_VAE_loss': np.mean(total_vae_loss),
                'vd_rec_loss': np.mean(total_rec_loss),
                'vd_KL_div': np.mean(total_KL_div),
            })

            wandb.log({
                'vd_L1_loss': np.mean(total_L1_loss),
                'vd_zeros_loss': np.mean(total_zeros_loss),
                'vd_ones_loss': np.mean(total_ones_loss),
            })

            wandb.log({
                'vd_compression_ratio': np.mean(total_compression_ratio),
            })

            if model['imputation']: 
                wandb.log({
                    'vd_imputation_L1_loss': np.mean(total_imputation_L1_loss),
                })

        if verbose:
            progress(
                current=0, total=0, train=False, bar=False, time=time.time()-ini,
                vae_loss=total_vae_loss, rec_loss=total_rec_loss, KL_div=total_KL_div,
                L1_loss=total_L1_loss, zeros_loss=total_zeros_loss, ones_loss=total_ones_loss,
                compression_ratio=total_compression_ratio, imputation_L1_loss=total_imputation_L1_loss if model['imputation'] else [0]
            )
        
        if model['imputation']: return total_vae_loss, total_rec_loss, total_KL_div, total_L1_loss, total_zeros_loss, total_ones_loss, total_imputation_L1_loss
        else: return total_vae_loss, total_rec_loss, total_KL_div, total_L1_loss, total_zeros_loss, total_ones_loss

@timer
def test(model, ts_loader, epoch, metadata, show_original=False, only=None, conditional=False, monitor=None, device='cpu'):

    # Set to evaluation mode
    model['body'].eval()

    original = np.empty((0, model['body'].module.encoder.FCs[0].FC[0].in_features if model['parallel'] else model['body'].encoder.FCs[0].FC[0].in_features), int)
    # latent = np.empty((0, model['body'].module.encoder.FCmu.FC[0].out_features if model['parallel'] else model['body'].encoder.FCmu.FC[0].out_features), int)
    labels = torch.Tensor([])

    for i, batch in enumerate(ts_loader):
        original = np.vstack((original, batch[0] if not conditional else (torch.cat([batch[0], batch[1].float()], 1))))
        labels = torch.cat((labels, batch[1].float() if not conditional else (np.argmax(batch[1].float(), axis=1))))
        if i == 5000: break
    
    original = torch.from_numpy(original).float()
    mu, _ = model['body'].module.encoder(original.to(device)) if model['parallel'] else model['body'].encoder(original.to(device))
    latent = mu.detach().cpu().squeeze(0).numpy().astype(float)
    log.info(f"Num of pops: {metadata['ts_metadata']['n_populations']}")
    print(f'Show original: {show_original}')
    fig = latentPCA(
        latent=latent, 
        labels=labels.int(), 
        original=original if show_original else None,
        only=only, 
        n_populations=metadata['ts_metadata']['n_populations']
    )
    if monitor == 'wandb': wandb.log({f"Latent space at epoch {epoch}": fig})
    plt.show()
    plt.close('all')
    
    del original
    del labels
    del latent
    del _
    gc.collect()
    torch.cuda.empty_cache()