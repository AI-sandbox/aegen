import gc
import sys
import time
import torch
import numpy as np
import blosc2 as blosc
from abc import ABC, abstractmethod

sys.path.insert(0, '/home/users/geleta/aegen/src')
from models.metrics import *
from eval.utils import *

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
        self.ae, _ = load_model(experiment)
        self.distribution = _['distribution']
        self.isize = _['isize']
        if _['shape'] == 'window-based': 
            self.bsize = int(_['isize'] / _['window_size'] * _['bsize'])
        else: self.bsize = int(_['bsize'])
        
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
        
        latent = np.empty((0, self.bsize), self.latent_type)
        reconstructed = np.empty((0, self.isize), bool)

        print(f'Autoencoder processing data of shape {data.shape}...')
        ini = time.time()
        for i in range(0, data.shape[0], batch_size):
            if data.shape[1] < batch_size:
                x = torch.from_numpy(data[:,:]).float().cuda()
            else: 
                x = torch.from_numpy(data[i:i + batch_size,:]).float().cuda()
            ## Obtain latent representation
            z = self.ae.encoder(x, None)
            if self.distribution == 'Uniform': _, z, _, _, _ = self.ae.quantizer(z)
            elif self.distribution == 'Multi-Bernoulli': z = self.ae.quantizer(z)
            elif self.distribution == 'Gaussian': z = torch.cat(z[0], axis=-1)
            ## Obtain reconstruction
            o = (self.ae.decoder(z, None) > 0.5).float().cpu().detach().numpy()
            ## Store latent representation
            z = z.cpu().detach().squeeze(0).numpy().astype(self.latent_type)
            latent = np.vstack((latent, z))
            ## Compute residual
            x = x.cpu().detach().numpy()
            r = np.abs(x - o).astype(bool)
            ## Store residual
            reconstructed = np.vstack((reconstructed, r))

            del x, z, o, r
            gc.collect()
            torch.cuda.empty_cache()
        
        xbin, zbin, rbin = data.tostring(), latent.tostring(), reconstructed.tostring()
        print(f'Compressing data of size {len(xbin)} bytes...')
        zcom = blosc.compress(zbin, typesize=elem_tsize(latent), cname=lz_algorithm, shuffle=shuffle)
        rcom = blosc.compress(rbin, typesize=elem_tsize(reconstructed), cname=lz_algorithm, shuffle=shuffle)
        end = time.time()
        del data, latent, reconstructed
        print(f'Compressed size: {len(zcom) + len(rcom)} bytes.')
        print('='*50)
        factor = len(xbin) / (len(zcom) + len(rcom))
        print(f'Compression ratio: {np.round(1/factor, 2)}.')
        print(f'Compression factor: x{np.round(factor, 2)}.')
        print(f'Compression time: {np.round(end - ini, 2)} seconds.')
        print('='*50)
        
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
        
        z = np.frombuffer(zbin, dtype=self.latent_type).reshape(-1,self.bsize)
        r = np.frombuffer(rbin, dtype=bool).reshape(-1,self.isize)

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