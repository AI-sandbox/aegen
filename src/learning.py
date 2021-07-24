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

from models.losses import *
from models.metrics import create_metrics_dict
from models.initializers import init_xavier

from utils.decorators import timer 
from utils.loggers import progress, latentPCA, saver, system_info
from utils.simulators import OnlineSimulator
from utils.assemblers import one_hot_encoder

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

@timer
def train(model, optimizer, hyperparams, stats, tr_loader, vd_loader, ts_loader, device='cpu', summary=None, num=0, only=None, monitor=None, metadata=None, metrics=None):

    #======================== Set model ========================#
    if metrics is None: log.info('[WARNING] No metrics defined.')   
    log.info('Setting model ...')
    model['body'].apply(init_xavier)
    if (monitor == 'wandb') and (metrics is not None):
        log.info('Initializing wandb ...')
        system_info()
        log.info('Setting environ variable ...')
        os.environ["WANDB_START_METHOD"] = "thread"
        
        ## Automate tag creation on run launch:
        wandb_tags = []
        ## Filter by latent space distribution --
        wandb_tags.append(model['distribution'])
        ## Filter by conditioning --
        if model['conditional']: wandb_tags.append('conditional')
        ## Filter by window size if window-based --
        if model['shape'] == 'window-based':
            if model['window_size'] <= 1000: wandb_tags.append('small wsize')
            else: wandb_tags.append('large wsize')
            ## Filter by bottleneck size --
            if model['bsize']*model['isize']//model['window_size'] < 256: wandb_tags.append('small bsize')
            else: wandb_tags.append('large bsize')
        
        wandb.init(
            project='AEgen_v2',
            dir=os.path.join(os.environ.get('OUT_PATH'), f'experiments/'),
            tags=wandb_tags,
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
    # log.info(model)
    log.info('Model set.')
    #======================== Prepare stats ========================#
    log.info('Preparing stats ...')
    best_loss = np.inf
    best_epoch = 0

    ## Initialize metrics dict.
    tr_metrics = create_metrics_dict(metrics, prefix='train')
    vd_metrics = create_metrics_dict(metrics, prefix='valid')
    
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
        assert(model['num_classes'] == metadata['vd_metadata']['n_populations'])
        online_simulator = OnlineSimulator(
            batch_size = hyperparams['batch_size'],
            n_populations = model['num_classes'],
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
        epoch_metrics = create_metrics_dict(metrics, prefix='aux')
        
        for i, batch in enumerate(tr_loader if offline else batch_counter):

            if offline:
                snps_array = batch[0].to(device)#.unsqueeze(1)
                labels = batch[1].to(device) if model['conditional'] else None
            else: 
                snps_array, labels = online_simulator.simulate()
                ## TODO: change metadata to isize from model...
                snps_array, labels = snps_array[:, :metadata['vd_metadata']['n_snps']].to(device), one_hot_encoder(labels[:,0].int(), model['num_classes']).to(device)
                labels = labels if model['conditional'] else None
            
            ## Forward inputs through net.
            if model['distribution'] == 'Gaussian':
                mu, logvar = model['body'].encoder(snps_array, labels)
                z = model['body'].reparametrize(mu, logvar)
                if model['shape'] == 'window-based':
                    mu = torch.cat(mu, axis=-1)
                    logvar = torch.cat(logvar, axis=-1)
            else: 
                z = mu = model['body'].encoder(snps_array, labels)
                if (model['distribution'] == 'Multi-Bernoulli') or (model['distribution'] == 'Uniform'):
                    z = mu = model['body'].quantizer(z)
            snps_reconstruction = model['body'].decoder(z, labels)
            input_mapper = {
                'input' : snps_array.bool(),
                'output': snps_reconstruction.float(),
                'reconstruction' : (snps_reconstruction > 0.5).bool(),
                'residual' : abs(snps_array - (snps_reconstruction > 0.5).float()).bool(),
                'mu' : mu.float() if model['distribution'] == 'Gaussian' else mu.bool(),
                'distribution' : model['distribution'],
                'batch_size' : int(snps_array.shape[0]),
            }
            if model['distribution'] == 'Gaussian': input_mapper['logvar'] = logvar
            
            ## Compute losses and metrics.
            for kmetric, meta in metrics.items():
                if callable(kmetric):
                    inputs = []
                    for var in meta['inputs']:
                        try: inputs.append(input_mapper[var]) 
                        except KeyError: pass 
                    for name, val in zip(meta['outputs'],kmetric(*inputs)):
                        epoch_metrics[f'aux_{name}'].append(val)
                else:
                    if i == 0:
                        for p in meta['params']:
                            inputs = []
                            for var in meta['inputs']:
                                try: inputs.append(input_mapper[var]) 
                                except KeyError: pass 
                            inputs.append(p)
                            epoch_metrics[f'aux_{p}_{kmetric}'].append(meta['function'](*inputs))
            # Backpropagation.
            optimizer['body'].zero_grad()
            
            opt_loss = aeloss(
                snps_array, 
                snps_reconstruction, 
                mu, 
                logvar if model['distribution'] == 'Gaussian' else None, 
                backward=True
            )
            if hyperparams['loss']['varloss']:
                opt_loss += varloss(mu, backward=True)
                
            opt_loss.backward()
            
            # One step of the optimizer (using the gradients from backpropagation).
            optimizer['body'].step()
            
            ## Clear memory and caches.
            del snps_array
            del labels
            del mu
            if model['distribution'] == 'Gaussian': del logvar
            del z
            del snps_reconstruction
            gc.collect()
            torch.cuda.empty_cache()

            if stats['verbose']:
                progress(
                    current=i+1, total=datalen, time=time.time()-ini,
                    vae_loss=epoch_metrics['aux_ae_loss'],
                    rec_loss=epoch_metrics['aux_reconstruction_loss'],
                    KL_div=epoch_metrics['aux_KL_divergence'],
                    L1_loss=epoch_metrics['aux_L1_loss'],
                    zeros_loss=epoch_metrics['aux_zeros_reconstruction_loss'],
                    ones_loss=epoch_metrics['aux_ones_reconstruction_loss'],
                )
        
        for kmetric, meta in metrics.items():
            if callable(kmetric):
                for name in meta['outputs']:
                    val = np.mean(epoch_metrics[f'aux_{name}'][-stats['slide']:])
                    tr_metrics[f'tr_{name}'].append(val)
            else:
                for p in meta['params']:
                    val = np.mean(epoch_metrics[f'aux_{p}_{kmetric}'][-stats['slide']:])
                    tr_metrics[f'tr_{p}_{kmetric}'].append(val)
        
        if optimizer['scheduler'] is not None: optimizer['scheduler'].step(np.mean(epoch_metrics['aux_ae_loss'][-stats['slide']:]))

        if monitor == 'wandb':
            for kmetric,val in tr_metrics.items():
                wandb.log({ kmetric: val[-1] })

        log.info(f"Epoch [{epoch + 1} / {hyperparams['epochs']}] ({time.time()-ini}s) VAE error: {tr_metrics['tr_ae_loss'][-1]}")
        
        ## Validate according validation scheduler.
        if (epoch % hyperparams['validation']['scheduler'] == 0):
            
            aux_vd_metrics = validate(
                model, 
                vd_loader, 
                epoch, 
                stats['verbose'], 
                monitor=monitor, 
                device=device,
                metrics=metrics
            )
            
            for kmetric, meta in metrics.items():
                if callable(kmetric):
                    for name in meta['outputs']:
                        val = aux_vd_metrics[f'aux_{name}']
                        vd_metrics[f'vd_{name}'].append(val)
                else:
                    for p in meta['params']:
                        val = aux_vd_metrics[f'aux_{name}']
                        vd_metrics[f'vd_{p}_{kmetric}'].append(val)
            del aux_vd_metrics
            
        if bool(vd_metrics['vd_ae_loss'][-1] < best_loss):
            saver(
                obj='model', 
                num=num, 
                state={
                    'architecture': model['architecture'],
                    'imputation': model['imputation'],
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
            best_loss = vd_metrics['vd_ae_loss'][-1]
            saver(
                obj='stats',
                num=num, 
                state={
                    'epoch': epoch + 1,
                    'tr_time': time.time() - ini,
                    'verbose': stats['verbose'],
                    'slide': stats['slide'],
                    'best_epoch': best_epoch,
                    'best_loss': best_loss,
                    # Training stats:
                    'tr_metrics': tr_metrics,
                    # Validation stats:
                    'vd_metrics': vd_metrics
                }
            )
        
        ## Save checkpoint according checkpointing scheduler.
        elif epoch % hyperparams['checkpointing']['scheduler'] == 0:
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
                    'tr_metrics': tr_metrics,
                    # Validation stats:
                    'vd_metrics': vd_metrics
                }
            )
            
        ## Test according testing scheduler.
        if (ts_loader is not None) and (epoch % hyperparams['testing']['scheduler'] == 0): #and (epoch != 0):
            log.info('Testing...')
            test(
                model=model, 
                ts_loader=ts_loader, 
                epoch=epoch, 
                metadata=metadata, 
                only=only, 
                conditional=model['conditional'], 
                monitor=monitor,
                device=device
            )
    
    print(f'Training finished in {time.time() - ini}s.')

@timer
def validate(model, vd_loader, epoch, verbose, monitor=None, device='cpu', metrics=None):

    # Set to evaluation mode
    model['body'].eval()
    loss = 0

    ## Initialize metrics dict.
    aux_vd_metrics = create_metrics_dict(metrics, prefix='aux')
    
    ini = time.time()
    with torch.no_grad():
        log.info('Validating current model...')
        for i, batch in enumerate(vd_loader):

            snps_array = batch[0].to(device)
            labels = batch[1].to(device) if model['conditional'] else None

            ## Forward inputs through net.
            if model['distribution'] == 'Gaussian':
                mu, logvar = model['body'].encoder(snps_array, labels)
                z = model['body'].reparametrize(mu, logvar)
                if model['shape'] == 'window-based':
                    mu = torch.cat(mu, axis=-1)
                    logvar = torch.cat(logvar, axis=-1)
            else: 
                z = mu = model['body'].encoder(snps_array, labels)
                if (model['distribution'] == 'Multi-Bernoulli') or (model['distribution'] == 'Uniform'):
                    z = mu = model['body'].quantizer(z)
            snps_reconstruction = model['body'].decoder(z, labels)
            input_mapper = {
                'input' : snps_array.bool(),
                'output': snps_reconstruction.float(),
                'reconstruction' : (snps_reconstruction > 0.5).bool(),
                'residual' : abs(snps_array - (snps_reconstruction > 0.5).float()).bool(),
                'mu' : mu.float() if model['distribution'] == 'Gaussian' else mu.bool(),
                'distribution' : model['distribution'],
                'batch_size' : int(snps_array.shape[0]),
            }
            if model['distribution'] == 'Gaussian': input_mapper['logvar'] = logvar
            
            ## Compute losses and metrics.
            for kmetric, meta in metrics.items():
                if callable(kmetric):
                    inputs = []
                    for var in meta['inputs']:
                        try: inputs.append(input_mapper[var]) 
                        except KeyError: pass 
                    for name, val in zip(meta['outputs'],kmetric(*inputs)):
                        aux_vd_metrics[f'aux_{name}'].append(val)
                else:
                    if i == 0:
                        for p in meta['params']:
                            inputs = []
                            for var in meta['inputs']:
                                try: inputs.append(input_mapper[var]) 
                                except KeyError: pass 
                            inputs.append(p)
                            aux_vd_metrics[f'aux_{p}_{kmetric}'].append(meta['function'](*inputs))
            
            del snps_array
            del labels
            del mu
            if model['distribution'] == 'Gaussian': del logvar
            del z
            del snps_reconstruction
            gc.collect()
            torch.cuda.empty_cache()
        
        for kmetric,val in aux_vd_metrics.items():
            aux_vd_metrics[kmetric] = np.mean(val)  
            if monitor == 'wandb':
                wandb.log({ 'vd'+kmetric[3:] : aux_vd_metrics[kmetric] })
                
        if verbose:
            progress(
                current=0, total=0, train=False, bar=False, time=time.time()-ini,
                vae_loss=aux_vd_metrics['aux_ae_loss'],
                rec_loss=aux_vd_metrics['aux_reconstruction_loss'],
                KL_div=aux_vd_metrics['aux_KL_divergence'],
                L1_loss=aux_vd_metrics['aux_L1_loss'],
                zeros_loss=aux_vd_metrics['aux_zeros_reconstruction_loss'],
                ones_loss=aux_vd_metrics['aux_ones_reconstruction_loss'],
            )
            
    return aux_vd_metrics

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