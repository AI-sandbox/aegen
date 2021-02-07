import os
import time
import torch
import datetime
import numpy as np
from models.VAEgen import VAEgen
from models.losses import VAEloss, L1loss
from utils.loader import loader 
from utils.decorators import timer 
from utils.loggers import progress
from tensorboardX import SummaryWriter

LOGDIR = os.path.join(os.path.join(os.environ.get('USER_PATH'), f"logs/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"))
writer = SummaryWriter(log_dir=LOGDIR)
print(LOGDIR)

def save_checkpoint(state, is_best, filename=os.path.join(os.environ.get('USER_PATH'), 'checkpoints/checkpoint.pt')):
     """Save checkpoint if a new best is achieved"""
     if is_best:
         print ("=> Saving a new best model")
         torch.save(state, filename)  # Save checkpoint
     else:
         print ("=> Loss did not improve")

@timer
def train(model, tr_loader, val_loader, hyperparams):

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

        writer.add_scalars(f'VAE_losses', {
            'VAE_loss': total_vae_loss[-1],
            'rec_loss': total_rec_loss[-1],
            'KL_div': total_KL_div[-1],
        }, epoch + 1)
		
        writer.add_scalars(f'L1_losses', {
            'L1_loss': total_L1_loss[-1],
            'zeros_loss': total_zeros_loss[-1],
            'ones_loss': total_ones_loss[-1],
        }, epoch + 1)

        print(f"Epoch [{epoch + 1} / {hyperparams['epochs']}] ({time.time()-ini}s) VAE error: {total_vae_loss[-1]}")
        
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
def validate(model, val_loader):

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
        for i, snps_array in enumerate(val_loader):

            snps_array = snps_array.to(device)

            snps_reconstruction, latent_mu, latent_logvar = model(snps_array)

            loss, rec_loss, KL_div = VAEloss(snps_array, snps_reconstruction, latent_mu, latent_logvar)
            L1_loss, zeros_loss, ones_loss = L1loss(snps_array, snps_reconstruction, partial=True, proportion=True)

            total_vae_loss.append(loss)
            total_rec_loss.append(rec_loss)
            total_KL_div.append(KL_div)

            total_L1_loss.append(L1_loss)
            total_zeros_loss.append(zeros_loss)
            total_ones_loss.append(ones_loss)

        progress(
            current=0, total=0, train=False, bar=False, time=time.time()-ini,
            vae_loss=total_vae_loss, rec_loss=total_rec_loss, KL_div=total_KL_div,
            L1_loss=total_L1_loss, zeros_loss=total_zeros_loss, ones_loss=total_ones_loss
        )
        

if __name__ == '__main__':

    hyperparams = {
        'max_limit': 100000,
        'epochs': 80,
        'batch_size': 64,
        'hidden_1_dims': 1024,
        'hidden_2_dims': 512,
        'latent_dims': 128,
        'lr': 0.01,
        'beta': 1,
        'weight_decay': 0,
    }

    dataloader = loader(DATA_PATH = os.path.join(os.environ.get('USER_PATH'), 'data/ancestry_datasets'),
                        batch_size=hyperparams['batch_size'], 
                        max_limit=hyperparams['max_limit'])

    vae = VAEgen(
        input=hyperparams['max_limit'], 
        hidden1=hyperparams['hidden_1_dims'], 
        hidden2=hyperparams['hidden_2_dims'],
        latent=hyperparams['latent_dims']
    )
    snps_array = next(iter(dataloader))
    writer.add_graph(vae, snps_array.detach(), verbose = False)

    train(model=vae, tr_loader=dataloader, hyperparams=hyperparams)