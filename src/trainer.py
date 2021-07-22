#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import gc
import time
import yaml
import torch
import logging
import numpy as np
import torch.nn as nn

from parser import create_parser

from models.aegen import aegen
from models.losses import aeloss, L1loss
from models.metrics import *

from utils.loader import loader 
from utils.loggers import system_info
from learning import train, validate, test


logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

if __name__ == '__main__':
    #======================== Prepare params ========================#
    parser = create_parser()
    args = parser.parse_args()
    with open(args.params, 'r') as f:
        params = yaml.safe_load(f)
    model_params = params['model']
    hyperparams = params['hyperparams']
    ksize=int(model_params['encoder']['layer0']['size'] / 1000)
    
    def summary_net(exp, species, chr, model_params, hyperparams):
        if (model_params['conditioning']['num_classes'] is not None) and (model_params['conditioning']['using']):
            shape = 'C-'+model_params['shape']
        else:
            shape = model_params['shape']
        layer_sizes = [str(model_params['encoder'][layer]['size']) for layer in model_params['encoder'].keys()]
        hyper = f'optimizer {hyperparams["optimizer"]["algorithm"]} with lr={hyperparams["optimizer"]["lr"]} and decay={hyperparams["optimizer"]["weight_decay"]}'
        data = f'using {hyperparams["training"]["simulation"]} simulation'
        name = f'[{exp}] {species.capitalize()} chr{chr}: {model_params["distribution"]} {shape}({",".join(layer_sizes)}), {hyper}, '
        return name
    
    summary = summary_net(args.num, args.species, args.chr, model_params, hyperparams)
    
    #======================== Prepare data ========================#
    system_info()
    IPATH = os.path.join(os.environ.get('IN_PATH'), f'data/{args.species}/chr{args.chr}/prepared')
    
    ## TR data loader.
    if hyperparams['training']['simulation'] == 'offline':
        log.info('-- USING OFFLINE SIMULATION FOR TR DATA --')
        log.info('Loading data...')
        log.info('Loading TR data...')
        tr_loader, tr_metadata = loader(
            ipath=IPATH,
            batch_size=hyperparams['batch_size'], 
            split_set='valid',
            ksize=ksize,
            only=args.only,
            conditional=model_params['conditioning']['using'],
        )
        log.info(f'TR data loaded.')
        log.info(f"Training set of shape <= {len(tr_loader) * hyperparams['batch_size']}")
        system_info()
    else: 
        log.info('-- USING ONLINE SIMULATION FOR TR DATA --')
        tr_loader, tr_metadata = None, None
    
    ## VD data loader.
    if hyperparams['validation']['simulation'] == 'offline':
        log.info('-- USING OFFLINE SIMULATION FOR VD DATA --')
        log.info('Loading VD data...')
        vd_loader, vd_metadata = loader(
            ipath=IPATH,
            batch_size=hyperparams['batch_size'], 
            split_set='valid',
            ksize=ksize,
            only=args.only,
            conditional=model_params['conditioning']['using']
        )
        log.info(f'VD data loaded.')
        log.info(f"Validation set of shape <= {len(vd_loader) * hyperparams['batch_size']}")
        system_info()
    else: raise Exception('VD data simulation can only be offline.')
    
    ## TS data loader.
    if bool(args.evolution):
        if hyperparams['testing']['simulation'] == 'offline':
            log.info('-- USING OFFLINE SIMULATION FOR TS DATA --')
            log.info('Loading TS data...')
            ts_loader, ts_metadata = loader(
                ipath=IPATH,
                batch_size=hyperparams['batch_size'], 
                split_set='test',
                ksize=ksize,
                only=args.only,
                conditional=model_params['conditioning']['using']
            )
            log.info(f'TS data loaded.')
            log.info(f"Test set of shape <= {len(ts_loader) * hyperparams['batch_size']}")
        else: raise Exception('TS data simulation can only be offline.')
    else: ts_loader, ts_metadata = None, None
        
    log.info('Data loaded ++')
    
    ## Checking coherence between datasets w.r.t. to the # of populations.
    if (ts_metadata is not None) and (tr_metadata is not None) and (vd_metadata is not None):
        if (tr_metadata['n_populations'] != vd_metadata['n_populations']) or (tr_metadata['n_populations'] != ts_metadata['n_populations']) or (vd_metadata['n_populations'] != ts_metadata['n_populations']):
            log.info(f'[WARNING] Missing populations:')
            log.info(f"\tTR SET has {tr_metadata['n_populations']} populations.")
            log.info(f"\tVD SET has {vd_metadata['n_populations']} populations.")
            log.info(f"\tTS SET has {ts_metadata['n_populations']} populations.")
    elif (ts_metadata is None) and (tr_metadata is not None) and (vd_metadata is not None):
        if (tr_metadata['n_populations'] != vd_metadata['n_populations']):
            log.info(f'[WARNING] Missing populations:')
            log.info(f"\tTR SET has {tr_metadata['n_populations']} populations.")
            log.info(f"\tVD SET has {vd_metadata['n_populations']} populations.")
    log.info(f'Data ready ++')
    #======================== Prepare model ========================#
    model = aegen(
        params=model_params, 
        conditional=model_params['conditioning']['using'],
        imputation=model_params['denoising']['using'],
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
    log.info(model)
    log.info('Model ready ++')
    #======================== Prepare optimizer ========================#
    if hyperparams['optimizer'] is not None:
        if hyperparams['optimizer']['algorithm'] == 'Adam':
            optimizer = torch.optim.Adam(
            params=model.parameters(), 
            lr=hyperparams['optimizer']['lr'], 
            weight_decay=hyperparams['optimizer']['weight_decay']
        )
        elif hyperparams['optimizer']['algorithm'] == 'AdamW':
            optimizer = torch.optim.AdamW(
            params=model.parameters(), 
            lr=hyperparams['optimizer']['lr'], 
            weight_decay=hyperparams['optimizer']['weight_decay']
        )
        # RuntimeError: SparseAdam does not support dense gradients, please consider Adam instead
        elif hyperparams['optimizer']['algorithm'] == 'SparseAdam':
            optimizer = torch.optim.SparseAdam(
            params=model.parameters(), 
            lr=hyperparams['optimizer']['lr'], # HAS NO WEIGHT DECAY!
            #weight_decay=hyperparams['optimizer']['weight_decay']
        )
        else: raise Exception('Unknown optimization algorithm')
    else: raise Exception('Missing optimizer')
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
            'architecture': model_params['shape'] + (' AE' if not model_params['conditioning']['using'] else ' C-AE'),
            'shape':model_params['shape'],
            'distribution': model_params['distribution'],
            'body': model, 
            'parallel': model_parallel,
            'num_params': num_params,
            'conditional': model_params['conditioning']['using'],
            'num_classes': model_params['conditioning']['num_classes'],
            'imputation': model_params['denoising']['using'],
            'gpu': torch.cuda.get_device_name(),
        }, 
        optimizer={
            'algorithm': hyperparams['optimizer']['algorithm'],
            'body': optimizer,
            'scheduler': lr_scheduler,
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
        device=device,
        metadata={
            'tr_metadata' : tr_metadata,
            'vd_metadata' : vd_metadata,
            'ts_metadata' : ts_metadata,
        },
        summary=summary,
        num=args.num,
        only=args.only,
        monitor='wandb',
        metrics={
            ## If the key is a function:
            ## Then define the inputs and the expected outputs
            ## of the function to store the metrics.
            aeloss: {
                'inputs' : ['input', 'output', 'mu', 'logvar'],
                'outputs': ['ae_loss', 'reconstruction_loss', 'KL_divergence'],
            },
            L1loss: {
                'inputs' : ['input', 'reconstruction'],
                'outputs': ['L1_loss', 'ones_reconstruction_loss', 'zeros_reconstruction_loss'],
            },
            residual_sparsity: {
                'inputs' : ['input', 'reconstruction', 'batch_size'],
                'outputs': ['residual_sparsity'],
            },
            varloss: {
                'inputs': ['mu'],
                'outputs': ['varloss'],
            },
            ## If the key is a metric name:
            ## Then define the inputs, the function,
            ## and the hyperparameters for the function.
            'cratio_no_shuffle' : {
                'inputs' : ['input', 'mu', 'residual', 'distribution'],
                'function': cratio_no_shuffle,
                'params' : ['lz4', 'zlib', 'zstd']
            },
            'cratio_bitshuffle' : {
                'inputs' : ['input', 'mu', 'residual', 'distribution'],
                'function': cratio_bitshuffle,
                'params' : ['lz4', 'zlib', 'zstd']
            },
            'ccratio_no_shuffle' : {
                'inputs' : ['input', 'mu', 'residual', 'distribution'],
                'function': ccratio_no_shuffle,
                'params' : ['lz4', 'zlib', 'zstd']
            },
            'cratio_bitshuffle' : {
                'inputs' : ['input', 'mu', 'residual', 'distribution'],
                'function': cratio_bitshuffle,
                'params' : ['lz4', 'zlib', 'zstd']
            },
            'ccratio_bitshuffle' : {
                'inputs' : ['input', 'mu', 'residual', 'distribution'],
                'function': ccratio_bitshuffle,
                'params' : ['lz4', 'zlib', 'zstd']
            },
            'partial_embedding_ccratio_no_shuffle' : {
                'inputs' : ['input', 'mu', 'residual', 'distribution'],
                'function': partial_embedding_ccratio_no_shuffle,
                'params' : ['lz4', 'zlib', 'zstd']
            },
            'partial_residual_ccratio_no_shuffle' : {
                'inputs' : ['input', 'mu', 'residual', 'distribution'],
                'function': partial_residual_ccratio_no_shuffle,
                'params' : ['lz4', 'zlib', 'zstd']
            },
            'partial_embedding_ccratio_bitshuffle' : {
                'inputs' : ['input', 'mu', 'residual', 'distribution'],
                'function': partial_embedding_ccratio_bitshuffle,
                'params' : ['lz4', 'zlib', 'zstd']
            },
            'partial_residual_ccratio_bitshuffle' : {
                'inputs' : ['input', 'mu', 'residual', 'distribution'],
                'function': partial_residual_ccratio_bitshuffle,
                'params' : ['lz4', 'zlib', 'zstd']
            },
        }
    )
    #======================== End training ========================#
