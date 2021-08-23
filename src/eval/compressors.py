import gc
import sys
import time
import torch
import numpy as np
import blosc2 as blosc
from abc import ABC, abstractmethod

sys.path.insert(0, '/home/geleta/aegen/src')
from models.metrics import *

def load_model(experiment):
    with open(f'/local-scratch/mrivas/experiments/exp{experiment}/params.yaml', 'r') as f: 
        params = yaml.safe_load(f)
    model_params = params['model']
    
    ae = aegen(model_params, conditional=False, sample_mode=False)
    
    state = torch.load(f'/local-scratch/mrivas/experiments/exp{experiment}/aegen_weights_{experiment}.pt')
    stats = torch.load(f'/local-scratch/mrivas/experiments/exp{experiment}/aegen_stats_{experiment}.pt')
    print(f"Model has been trained during {stats['epoch']} epochs.")
    print(f"Retrieving the  best checkpoint, from epoch {stats['best_epoch']}.")
    ae.load_state_dict(state['weights'], strict=False)
    ae.eval();
    print("Model ready for testing.")
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {torch.cuda.device_count()} GPU(s).")
    print(f'Sending model to device {torch.cuda.get_device_name()}.')
    return ae.to(device), state

## Abstract class of a compressor.
## [experiment] defines the ID of the trained ae to load.
class Compressor(ABC):
    
    ## [ae] defines the autoencoder model.
    ## [distribution] can be [Gaussian, Multi-Bernoulli, Uniform, Unknown].
    ## [isize] is the input size (the size of the data to compress).
    ## [bsize] is the bottleneck size (the size of the latent representation
    ## of the compressed data).
    ## [latent_type] is the data type of the latent factors.
    def __init__(self, experiment):
        self.ae, self.stats = load_model(experiment)
        self.distribution = self.stats['distribution']
        self.isize = self.stats['isize']
        self.wsize = self.stats['window_size']
        if self.stats['shape'] == 'window-based': 
            self.bsize = int(self.stats['isize'] / self.wsize * self.stats['bsize'])
        else: self.bsize = int(self.stats['bsize'])
        
        if self.distribution == 'Uniform':
            self.latent_type = np.dtype('B')
        elif self.distribution == 'Multi-Bernoulli':
            self.latent_type = bool
        elif self.distribution == 'Gaussian': 
            self.latent_type = float
        elif self.distribution == 'Unknown':
            self.latent_type = float
        else: raise Exception('Unknown latent type.')    
        
        print(f'{self.distribution.capitalize()} compressor ({self.isize} → {self.bsize})')
            
        super().__init__()
     
    @abstractmethod
    def compress(self):
        pass
    
    @abstractmethod
    def decompress(self):
        pass

## Definition of the Lempel-Ziv + Autoencoder compressor.
## Inherits from Compressor class.
class LZAE(Compressor):
    
    ## Compress method is used to compress input SNP data, 
    ## passed in with the [data] variable.
    ## [lz_algorithm] is the Lempel-Ziv algorithm to compress the
    ## latent representation and the residuals.
    ## [lz_algorithm] can be [zstd, zlib, lz4].
    ## [shuffle] defines whether to use shuffling.
    ## [shuffle] can be [NO_SHUFFLE, BITSHUFFLE].
    ## [batch_size] defines the size of the batches to be processed.
    ## Depending on the capabilities of the GPU, the batch size can be increased.
    def compress(self, data, lz_algorithm='zstd', shuffle=blosc.BITSHUFFLE, batch_size=512, save_z=None, save_r=None):
        
        if self.distribution == 'Uniform':
            latent = np.empty((0, int(self.isize / self.wsize)), self.latent_type)
        else: latent = np.empty((0, self.bsize), self.latent_type)
        residual = np.empty((0, self.isize), bool)

        print(f'Autoencoder processing data of shape {data.shape}...')
        ini = time.time()
        for i in range(0, data.shape[0], batch_size):
            if data.shape[1] < batch_size:
                x = torch.from_numpy(data[:,:]).float().cuda()
            else: 
                x = torch.from_numpy(data[i:i + batch_size,:]).float().cuda()
            ## Obtain latent representation
            z = self.ae.encoder(x, None)
            if self.distribution == 'Uniform': indices, z, _, _, _ = self.ae.quantizer(z)
            elif self.distribution == 'Multi-Bernoulli': z = self.ae.quantizer(z)
            elif self.distribution == 'Gaussian': z = torch.cat(z[0], axis=-1)
            ## Obtain reconstruction
            o = (self.ae.decoder(z, None) > 0.5).float().cpu().detach().numpy()
            ## Store latent representation
            z = z.cpu().detach().squeeze(0).numpy()
            if self.distribution == 'Multi-Bernoulli':
                z = 1/2*z + 1/2
            elif self.distribution == 'Uniform': 
                z = indices.float().cpu().detach().numpy()
            z = z.astype(self.latent_type)
            latent = np.vstack((latent, z))
            ## Compute residual
            x = x.cpu().detach().numpy()
            r = np.abs(x - o).astype(bool)
            ## Store residual
            residual = np.vstack((residual, r))

            del x, z, o, r
            gc.collect()
            torch.cuda.empty_cache()
        
        xbin, zbin, rbin = data.tostring(), latent.tostring(), residual.tostring()
        print(f'Compressing data of size {len(xbin)} bytes...')
        xcom = blosc.compress(xbin, typesize=1, cname=lz_algorithm, shuffle=shuffle)
        print(f'Compressed size of raw data: {len(xcom)} bytes.')
        zcom = blosc.compress(zbin, typesize=elem_tsize(latent), cname=lz_algorithm, shuffle=shuffle)
        rcom = blosc.compress(rbin, typesize=elem_tsize(residual), cname=lz_algorithm, shuffle=shuffle)
        end = time.time()
        print(f'Compressed size of ae-processed data: {len(zcom) + len(rcom)} bytes.')
        print('='*50)
        factor = len(xbin) / (len(zcom) + len(rcom))
        cfactor = len(xcom) / (len(zcom) + len(rcom))
        print(f'Average reconstruction error: {np.sum(residual.astype(float))/data.shape[1]}.')
        print(f'Compression ratio: {np.round(1/factor, 2)}.')
        print(f'Compression factor: x{np.round(factor, 2)}.')
        print(f'CC factor: x{np.round(cfactor, 2)}.')
        print(f'Compression time: {np.round(end - ini, 2)} seconds.')
        print('='*50)
        del data, latent, residual
        
        ## Save compressed latent representation, if desired.
        if save_z is not None:
            f = open(save_z, 'wb')
            f.write(zcom)
            f.close()
        
        ## Save compressed residual, if desired.
        if save_r is not None:
            f = open(save_r, 'wb')
            f.write(rcom)
            f.close()
        
        return zcom, rcom
    
    ## [zcom] is the compressed latent representation by the Compress 
    ## method of the data to be decompressed.
    ## [rcom] is the compressed residual by the Compress method of 
    ## the data to be decompressed.
    ## [batch_size] defines the size of the batches to be processed.
    ## Depending on the capabilities of the GPU, the batch size can be increased.
    def decompress(self, zcom, rcom, batch_size=512):
        zbin = blosc.decompress(zcom)
        rbin = blosc.decompress(rcom)
        
        if self.distribution == 'Uniform':
            indices = torch.from_numpy(np.frombuffer(zbin, dtype=self.latent_type).reshape(-1,int(self.isize/self.wsize))).cuda().long()
            z = self.ae.quantizer.codebook.weight.index_select(0, indices.view(-1)).view((-1,self.bsize)).cpu().detach().numpy()
            del indices
        else:
            z = np.frombuffer(zbin, dtype=self.latent_type).reshape(-1,self.bsize).astype(float)
        r = np.frombuffer(rbin, dtype=bool).reshape(-1,self.isize).astype(int)
        if self.distribution == 'Multi-Bernoulli':
            z = 2*z - 1

        reconstructed = np.empty((0, self.isize), bool)
        print(f'Autoencoder processing latent data of shape {z.shape}...')
        ini = time.time()
        for i in range(0, z.shape[0], batch_size):
            zi = torch.from_numpy(z[i:i + batch_size,:]).float().cuda()
            ri = r[i:i + batch_size,:]
            ## Obtain lossless reconstruction.
            o = (self.ae.decoder(zi, None) > 0.5).float().cpu().detach().numpy()
            ## Error correction.
            o = (np.abs(o + ri) % 2).astype(bool)
            reconstructed = np.vstack((reconstructed, o))

            del zi, ri
            gc.collect()
            torch.cuda.empty_cache()
        end = time.time()
        print(f'Losslessly reconstructed data of shape {reconstructed.shape} in {np.round(end - ini, 2)} seconds.')
        del z, r

        return reconstructed
    
    ## Decompress from files.
    ## [zcom_path] is the path where the compressed latent representation
    ## is stored.
    ## [rcom_path] is the path where the compressed residual is stored.
    ## [batch_size] defines the size of the batches to be processed.
    ## Depending on the capabilities of the GPU, the batch size can be increased.
    def decompress_from_files(self, zcom_path, rcom_path, batch_size=512):
        ## Read compressed latent representation.
        f = open(zcom_path, 'r')
        zcom = f.read()
        f.close()
        ## Read compressed residual.
        f = open(rcom_path, 'r')
        rcom = f.read()
        f.close()
        
        return self.decompress(zcom, rcom, batch_size=batch_size)