import os
import gc
import time
import json
import yaml
import wandb
import torch
import psutil
import datetime
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from parser import create_parser
from models.VAEgen import AEgen
from models.losses import VAEloss, L1loss
from models.initializers import init_xavier
from utils.loader import loader 
from utils.decorators import timer 
from utils.loggers import progress, latentPCA, saver, system_info

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

@timer
def train(model, optimizer, hyperparams, stats, tr_loader, vd_loader, ts_loader, summary=None, num=0, only=None, monitor=None, metadata=None):

    #======================== Set data ========================#
    #log.info('Setting data ...')
    #tr_loader, vd_loader, ts_loader = data
    #log.info('Data set.')
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
    log.info('Model set.')
    #======================== Prepare stats ========================#
    log.info('Preparing stats ...')
    best_loss = np.inf
    best_epoch = 0
    datalen = len(tr_loader)

    total_vae_loss, total_rec_loss, total_KL_div  = [], [], []
    total_L1_loss, total_zeros_loss, total_ones_loss = [], [], []
    total_compression_ratio = []
    if model['imputation']: total_imputation_L1_loss = []
    log.info('Stats prepared.')
    #======================== Start training loop ========================#
    log.info('Training loop starting now ...')
    for epoch in range(hyperparams['epochs']):
        
        ini = time.time()

        epoch_vae_loss, epoch_rec_loss, epoch_KL_div  = [], [], []
        epoch_L1_loss, epoch_zeros_loss, epoch_ones_loss = [], [], []
        epoch_compression_ratio = []
        if model['imputation']: epoch_imputation_L1_loss = []
        
        for i, batch in enumerate(tr_loader):

            snps_array = batch[0].to(device)#.unsqueeze(1)
            labels = batch[1].to(device) if model['architecture'] == 'C-VAE' else None

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

        if model['imputation']: 
            vd_vae_loss, vd_rec_loss, vd_KL_div, vd_L1_loss, vd_zeros_loss, vd_ones_loss, vd_imputation_L1_loss = validate(model, vd_loader, epoch, stats['verbose'], monitor=monitor)
        else:
            vd_vae_loss, vd_rec_loss, vd_KL_div, vd_L1_loss, vd_zeros_loss, vd_ones_loss = validate(model, vd_loader, epoch, stats['verbose'], monitor=monitor)

        if (ts_loader is not None) and (epoch % 150 == 0): #and (epoch != 0):
            log.info('Testing...')
            conditional = (model['architecture'] == 'C-VAE')
            test(
                model=model, 
                ts_loader=ts_loader, 
                epoch=epoch, 
                metadata=metadata, 
                only=only, 
                conditional=conditional, 
                monitor=monitor
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
def validate(model, vd_loader, epoch, verbose, monitor=None):

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
def test(model, ts_loader, epoch, metadata, show_original=False, only=None, conditional=False, monitor=None):

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

if __name__ == '__main__':

    #======================== Prepare params ========================#
    parser = create_parser()
    args = parser.parse_args()
    with open(args.params, 'r') as f:
        params = yaml.safe_load(f)
    model_params = params['model']
    hyperparams = params['hyperparams']
    ksize=int(model_params['encoder']['layer0']['size'] / 1000)
    
    def summary_net(exp, species, chr, params):
        if params['num_classes'] is not None:
            shape = 'C-'+params['shape']
        else:
            shape = params['shape'].capitalize()
        layer_sizes = [str(params['encoder'][layer]['size']) for layer in params['encoder'].keys()]
        name = f'[{exp}] {species.capitalize()} chr{chr}: {shape}({",".join(layer_sizes)})'
        return name
    
    summary = summary_net(args.num, args.species, args.chr, model_params)
    
    #======================== Prepare data ========================#
    system_info()
    IPATH = os.path.join(os.environ.get('IN_PATH'), f'data/{args.species}/chr{args.chr}/prepared')
    log.info('Loading data...')
    log.info('Loading TR data...')
    tr_loader, tr_metadata = loader(
        ipath=IPATH,
        batch_size=hyperparams['batch_size'], 
        split_set='train',
        ksize=ksize,
        only=args.only,
        conditional=args.conditional,
    )
    log.info(f'TR data loaded.')
    system_info()
    log.info('Loading VD data...')
    vd_loader, vd_metadata = loader(
        ipath=IPATH,
        batch_size=hyperparams['batch_size'], 
        split_set='valid',
        ksize=ksize,
        only=args.only,
        conditional=args.conditional
    )
    log.info(f'VD data loaded.')
    system_info()
    ts_loader, ts_metadata = loader(
        ipath=IPATH,
        batch_size=hyperparams['batch_size'], 
        split_set='test',
        ksize=ksize,
        only=args.only,
        conditional=args.conditional
    ) if bool(args.evolution) else None, None
    log.info('Data loaded ++')
    log.info(f"Training set of shape <= {len(tr_loader) * hyperparams['batch_size']}")
    log.info(f"Validation set of shape <= {len(vd_loader) * hyperparams['batch_size']}")
    if bool(args.evolution): log.info(f"Test set of shape <= {len(ts_loader) * hyperparams['batch_size']}")
    if ts_metadata is not None:
        if (tr_metadata['n_populations'] != vd_metadata['n_populations']) or (tr_metadata['n_populations'] != ts_metadata['n_populations']) or (vd_metadata['n_populations'] != ts_metadata['n_populations']):
            log.info(f'[WARNING] Missing populations:')
            log.info(f"\tTR SET has {tr_metadata['n_populations']} populations.")
            log.info(f"\tVD SET has {vd_metadata['n_populations']} populations.")
            log.info(f"\tTS SET has {ts_metadata['n_populations']} populations.")
    else:
        if (tr_metadata['n_populations'] != vd_metadata['n_populations']):
            log.info(f'[WARNING] Missing populations:')
            log.info(f"\tTR SET has {tr_metadata['n_populations']} populations.")
            log.info(f"\tVD SET has {vd_metadata['n_populations']} populations.")
    log.info(f'Data ready ++')
    #======================== Prepare model ========================#
    model = AEgen(
        params=model_params, 
        conditional=args.conditional,
        imputation=args.imputation,
        sample_mode=False,
    )

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f'Using device: {device}')
    model_parallel = False
    if torch.cuda.device_count() > 1:
        model, model_parallel = nn.DataParallel(model), True
    log.info(f"Using {torch.cuda.device_count()} GPU(s)")
    log.info(f'Sending model to device {torch.cuda.get_device_name()}')
    model.to(device)

    # Number of parameters used in the model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f'Number of model parameters: {num_params}')
    system_info()
    log.info('Model ready ++')
    #======================== Prepare optimizer ========================#
    optimizer = torch.optim.Adam(
        params=model.parameters(), 
        lr=hyperparams['lr'], 
        weight_decay=hyperparams['weight_decay']
    )
    log.info(f'Optimizer ready ++')
    #======================== Prepare scheduler ========================#
    if hyperparams['scheduler'] is not None:
        if hyperparams['scheduler']['method'] == 'plateau':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=hyperparams['scheduler']['factor'],
                patience=hyperparams['scheduler']['patience'],
                threshold=hyperparams['scheduler']['threshold'],
                threshold_mode=hyperparams['scheduler']['mode']
            )
        elif hyperparams['scheduler']['method'] == 'step':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=hyperparams['scheduler']['step'],
                gamma=hyperparams['scheduler']['gamma']
            )
        elif hyperparams['scheduler']['method'] == 'multistep':
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=hyperparams['scheduler']['milestones'],
                gamma=hyperparams['scheduler']['gamma']
            )
        elif hyperparams['scheduler']['method'] == 'exponential':
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=hyperparams['scheduler']['gamma']
            )
        else: raise Exception('Wrong definition for scheduler')
    else: lr_scheduler = None
    log.info('Scheduler ready ++')
    #======================== Start training ========================#
    log.info('Starting training...')
    system_info()
    train(
        model={
            'architecture': model_params['shape'] + (' AE' if not args.conditional else ' C-AE'),
            'shape':model_params['shape'],
            'distribution': model_params['distribution'],
            'body': model, 
            'parallel': model_parallel,
            'num_params': num_params,
            'imputation': args.imputation,
            'gpu': torch.cuda.get_device_name(),
        }, 
        optimizer={
            'body': optimizer,
            'scheduler': lr_scheduler
        },
        hyperparams=hyperparams,
        stats={
            'epoch': 0,
            'verbose': bool(args.verbose),
            'slide': 15,
            'best_epoch': 0,
            'best_loss': np.inf,
        },
        tr_loader=tr_loader,
        vd_loader=vd_loader,
        ts_loader=ts_loader,
        metadata={
            'tr_metadata' : tr_metadata,
            'vd_metadata' : vd_metadata,
            'ts_metadata' : ts_metadata,
        },
        summary=summary,
        num=args.num,
        only=args.only,
        monitor='wandb',
    )
    #======================== End training ========================#
