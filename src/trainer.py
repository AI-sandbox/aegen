import os
import time
import torch
from models.VAEgen import VAEgen
from models.losses import VAEloss
from utils.loader import loader 
from utils.decorators import timer 

def save_checkpoint(state, is_best, filename=os.path.join(os.environ.get('USER_PATH'), 'checkpoints/checkpoint.pt')):
     """Save checkpoint if a new best is achieved"""
     if is_best:
         print ("=> Saving a new best model")
         torch.save(state, filename)  # Save checkpoint
     else:
         print ("=> Loss did not improve")

@timer
def train(model, dataloader, hyperparams):
    optimizer = torch.optim.Adam(params=vae.parameters(), lr=hyperparams['lr'], weight_decay=hyperparams['weight_decay'])

    # This is the number of parameters used in the model
    num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f'Number of model parameters: {num_params}')

    # set to training mode
    model.train()

    train_loss_avg = 0
    datalen = len(dataloader)

    print('Training ...')
    for epoch in range(hyperparams['epochs']):
        
        num_batches = 0
        
        for i, snps_array in enumerate(dataloader):

            snps_reconstruction, latent_mu, latent_logvar = model(snps_array)
            loss, _ = VAEloss(snps_array, snps_reconstruction, latent_mu, latent_logvar)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # One step of the optimizer (using the gradients from backpropagation)
            optimizer.step()

            train_loss_avg += loss.item()

            print(f"Epoch progress: [{(i + 1)} / {datalen}] " + '='*(i + 1) + '-'*(datalen - (i + 1)))
            
        print(f"Epoch [{epoch + 1} / {hyperparams['epochs']}] average reconstruction error: {train_loss_avg / (i + 1)}")
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'beta': hyperparams['beta'],
            'lr': hyperparams['lr'],
        }, is_best = True, filename=os.path.join(os.environ.get('USER_PATH'), f'checkpoints/checkpoint_VAE_{epoch + 1}.pt'))

if __name__ == '__main__':

    hyperparams = {
        'max_limit': 100000,
        'epochs': 80,
        'batch_size': 128,
        'hidden_dims': 1024,
        'latent_dims': 512,
        'lr': 1e-3,
        'beta': 1,
        'weight_decay': 1e-5,
    }

    dataloader = loader(DATA_PATH = os.path.join(os.environ.get('USER_PATH'), 'data/ancestry_datasets'),
                        batch_size=hyperparams['batch_size'], 
                        max_limit=hyperparams['max_limit'])
    print(dataloader)

    vae = VAEgen(input=hyperparams['max_limit'], hidden=hyperparams['hidden_dims'], latent=hyperparams['latent_dims'])

    train(model=vae, dataloader=dataloader, hyperparams=hyperparams)