import gc
import sys
import time
import torch
import shlex
import subprocess
import numpy as np
import pandas as pd
import blosc2 as blosc
from abc import ABC, abstractmethod

sys.path.insert(0, '/home/geleta/aegen/src')
from models.metrics import *

## Data in/out paths:
os.environ["IN_PATH"]  = "/local-scratch/mrivas"
os.environ["OUT_PATH"] = "/local-scratch/mrivas"
os.environ["GENOZIP_PATH"] = "/home/geleta/genozip-linux-x86_64"
path = lambda chm, ini, end: os.path.join(os.environ.get('IN_PATH'), 
                                     f'data/human/chr{chm}/prepared/test/test{int(end-ini)}_{ini}_{end}.h5') 
def load(path):
    with h5py.File(path, 'r') as f: snps = f['snps'][:]
    return snps

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
    def __init__(self, experiment, chm=22):
        self.chm = chm
        self.ae, self.stats = load_model(experiment)
        self.distribution = self.stats['distribution']
        self.isize = self.stats['isize']
        self.wsize = self.stats['window_size']
        if self.stats['shape'] == 'window-based': 
            self.bsize = int(self.stats['isize'] / self.wsize * self.stats['bsize'])
        else: self.bsize = int(self.stats['bsize'])
        
        if self.distribution == 'Uniform':
            self.latent_type = np.dtype('B')
            self.heads = 1#self.stats['heads'] if self.stats['heads'] is not None else 1
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
    
    ## Project data into low dimensional manifold given by AE latent space and compute
    ## the residual from reconstruction. Returns the latent representation and residual.
    def transform(self, data, latent, residual, batch_size=512, verbose=True):
        if verbose: print(f'Autoencoder processing data of shape {data.shape}...')
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
                z = indices.int().cpu().detach().numpy()
            z = z.astype(self.latent_type)
            if self.distribution != 'Uniform' or (self.distribution == 'Uniform' and self.heads <= 1): 
                latent = np.vstack((latent, z))
            else: 
                z = indices.permute(0,2,1).cpu().detach().numpy()
                latent = np.concatenate([latent, z], axis=0)
            ## Compute residual
            x = x.cpu().detach().numpy()
            r = np.abs(x - o).astype(bool)
            ## Store residual
            residual = np.vstack((residual, r))
            
            del x, z, o, r
            gc.collect()
            torch.cuda.empty_cache()
            
        if verbose: print(f'Average reconstruction error: {np.sum(residual.astype(float))/data.shape[1]}.')
        
        return latent, residual
    
    ## Encode [objbytes] into a bit stream with [algorithm] by splitting bytes by [typesize].
    ## Optionally, a [shuffle] can be used. [shuffle] can be [NO_SHUFFLE, BITSHUFFLE].
    def bitencode(self, objbytes, typesize, algorithm, shuffle=blosc.NOSHUFFLE, verbose=True):
        cobjbytes = blosc.compress(objbytes, typesize=typesize, cname=algorithm, shuffle=shuffle)
        return cobjbytes
        
    ## Compress method is used to compress input SNP data, 
    ## passed in with the [data] variable.
    ## [lz_algorithm] is the Lempel-Ziv algorithm to compress the
    ## latent representation and the residuals.
    ## [lz_algorithm] can be [zstd, zlib, lz4].
    ## [shuffle] defines whether to use shuffling.
    ## [shuffle] can be [NO_SHUFFLE, BITSHUFFLE].
    ## [batch_size] defines the size of the batches to be processed.
    ## Depending on the capabilities of the GPU, the batch size can be increased.
    def compress(self, data, algorithm='zstd', shuffle=blosc.NOSHUFFLE, batch_size=512, save_z=None, save_r=None, verbose=True):
        if (algorithm == 'genozip') and (save_z is None or save_r is None):
            raise Exception('To use genozip you must store the outputs at disk.')
        
        if self.distribution == 'Uniform':
            latent = np.empty((0, self.heads, int(self.isize / self.wsize)), self.latent_type)
            if self.heads <= 1: latent = np.squeeze(latent, axis = 1)
        else: latent = np.empty((0, self.bsize), self.latent_type)
        residual = np.empty((0, self.isize), bool)

        latent, residual = self.transform(data, latent, residual, batch_size=batch_size, verbose=verbose)
        
        xbin, zbin, rbin = data.tostring(), latent.tostring(), residual.tostring()
        if verbose: 
            print(f'Compressing data of size {len(xbin)} bytes...')
            print(f'Latent representation bytes: {len(zbin)}.')
            print(f'Residual bytes: {len(rbin)}.')
        
        if algorithm == 'genozip':
            f = open(save_z, 'wb')
            f.write(latent)
            f.close()
            
            commands =  f'--input generic --force -o /tmp/z.genozip {save_z}'
            execute = os.path.join(os.environ.get('GENOZIP_PATH'), f'genozip {commands}')
            print(execute)
            proc = subprocess.Popen(shlex.split(execute), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            proc.wait()
            
            f = open(save_r, 'wb')
            f.write(residual)
            f.close()
            
            commands =  f'--input generic --force -o /tmp/r.genozip {save_r}'
            execute = os.path.join(os.environ.get('GENOZIP_PATH'), f'genozip {commands}')
            print(execute)
            proc = subprocess.Popen(shlex.split(execute), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            proc.wait()
        
        else: 
            zcom = self.bitencode(zbin, typesize=elem_tsize(latent), algorithm=algorithm, shuffle=shuffle)
            rcom  = self.bitencode(rbin, typesize=elem_tsize(residual), algorithm=algorithm, shuffle=shuffle)
            del data, latent, residual
            
            ## Save compressed latent representation, if desired.
            if save_z is not None:
                f = open(save_z, 'wb')
                f.write(zcom)
                f.close()
                print(f'Latent z stored at {save_z}')

            ## Save compressed residual, if desired.
            if save_r is not None:
                f = open(save_r, 'wb')
                f.write(rcom)
                f.close()
                print(f'Residual z stored at {save_r}')

            if verbose:
                cfactor = len(xbin) / (len(zcom) + len(rcom))

                xcom = self.bitencode(xbin, typesize=1, algorithm=algorithm, shuffle=shuffle)
                print(f'Compressed size of raw data: {len(xcom)} bytes.') 

                ccfactor = len(xcom) / (len(zcom) + len(rcom))

                print(f'Compressed latent representation bytes: {len(zcom)}.')
                print(f'Compressed residual bytes: {len(rcom)}.')
                print(f'Compressed size of ae-processed data: {len(zcom) + len(rcom)} bytes.')
                print('='*50)
                print(f'Compression ratio: {np.round(1/cfactor, 2)}.')
                print(f'Compression factor: x{np.round(cfactor, 2)}.')
                print(f'CC factor: x{np.round(ccfactor, 2)}.')
                print('='*50)
        
        if algorithm != 'genozip': return zcom, rcom        
    
    ## [zcom] is the compressed latent representation by the Compress 
    ## method of the data to be decompressed.
    ## [rcom] is the compressed residual by the Compress method of 
    ## the data to be decompressed.
    ## [batch_size] defines the size of the batches to be processed.
    ## Depending on the capabilities of the GPU, the batch size can be increased.
    def decompress(self, zbin, rbin, batch_size=512):
        
        if self.distribution == 'Uniform':
            indices = torch.from_numpy(np.frombuffer(zbin, dtype=self.latent_type).reshape(-1,self.heads,int(self.isize/self.wsize))).cuda().long()
            if self.heads <= 1: indices = indices.squeeze(1)
            z = self.ae.quantizer.codebook.weight.index_select(0, indices.view(-1)).view((-1,self.heads,self.bsize)).cpu().detach().numpy()
            if self.heads <= 1: z = np.squeeze(z, axis=1)
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
            zi = torch.from_numpy(z[i:i + batch_size,...]).float().cuda()
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
        
        if zcom_path.endswith('.genozip'):
            commands =  f'--force -o /tmp/z_unzipped.bytes {zcom_path}'
            execute = os.path.join(os.environ.get('GENOZIP_PATH'), f'genounzip {commands}')
            print(execute)
            proc = subprocess.Popen(shlex.split(execute), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            proc.wait()
            
            ## Read compressed latent representation.
            f = open('/tmp/z_unzipped.bytes', 'rb')
            zbin = f.read()
            f.close()    
        else:
            ## Read compressed latent representation.
            f = open(zcom_path, 'rb')
            zcom = f.read()
            f.close()
            
            zbin = blosc.decompress(zcom)
        
        if rcom_path.endswith('.genozip'):
            commands =  f'--force -o /tmp/r_unzipped.bytes {rcom_path}'
            execute = os.path.join(os.environ.get('GENOZIP_PATH'), f'genounzip {commands}')
            print(execute)
            proc = subprocess.Popen(shlex.split(execute), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            proc.wait()
            
            ## Read compressed latent representation.
            f = open('/tmp/r_unzipped.bytes', 'rb')
            rbin = f.read()
            f.close()
        else:
            ## Read compressed residual.
            f = open(rcom_path, 'rb')
            rcom = f.read()
            f.close()
            rbin = blosc.decompress(rcom)
        
        return self.decompress(zbin, rbin, batch_size=batch_size)
    
    ## Benchmark test with [algorithms]. Computes the improvement over [algorithms]
    ## by adding the autoencoder into the compression pipeline.
    def benchmark(self, data, batch_size=512, algorithms=['lz4', 'zlib', 'zstd', 'genozip'], shuffle=blosc.NOSHUFFLE):
        if self.distribution == 'Uniform':
            latent = np.empty((0, self.heads, int(self.isize / self.wsize)), self.latent_type)
            if self.heads <= 1: latent = np.squeeze(latent, axis = 1)
        else: latent = np.empty((0, self.bsize), self.latent_type)
        residual = np.empty((0, self.isize), bool)
        
        print(f'Autoencoder processing data of shape {data.shape}...')
        latent, residual = self.transform(data, latent, residual, batch_size=batch_size, verbose=False)
        xbin, zbin, rbin = data.tostring(), latent.tostring(), residual.tostring()
        
        def bytes_disk(file):
            path_to_file = f'ls -ltr /tmp/{file}'
            process_list_file = subprocess.Popen(shlex.split(path_to_file), stdout=subprocess.PIPE)
            out, err = process_list_file.communicate()
            out = out.split()
            size = []
            for i in range(0,len(out),9): size.append(int(out[i:i+9][4]))
            if len(size) == 1: size = size[0]
            return size
                
        benchlist = []
        ini = time.time()
        for algorithm in algorithms: 
            print(f'Encoding into bytes with {algorithm} algorithm...')
            if algorithm == 'genozip':
                
                commands =  f'--input generic --force -o /tmp/bench_x.genozip {path(self.chm, 0, self.isize)}'
                execute = os.path.join(os.environ.get('GENOZIP_PATH'), f'genozip {commands}')
                print(execute)
                proc = subprocess.Popen(shlex.split(execute), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                proc.wait()
                
                f = open('/tmp/bench_z.bytes', 'wb')
                f.write(latent)
                f.close()
                commands =  '--input generic --force -o /tmp/bench_z.genozip /tmp/bench_z.bytes'
                execute = os.path.join(os.environ.get('GENOZIP_PATH'), f'genozip {commands}')
                print(execute)
                proc = subprocess.Popen(shlex.split(execute), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                proc.wait()
                
                f = open('/tmp/bench_r.bytes', 'wb')
                f.write(residual)
                f.close()
                commands =  '--input generic --force -o /tmp/bench_r.genozip /tmp/bench_r.bytes'
                execute = os.path.join(os.environ.get('GENOZIP_PATH'), f'genozip {commands}')
                print(execute)
                proc = subprocess.Popen(shlex.split(execute), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                proc.wait()
                
                lenxcom = bytes_disk('bench_x.genozip')
                lenzcom = bytes_disk('bench_z.genozip')
                lenrcom = bytes_disk('bench_r.genozip')   

            else:
                lenxcom = len(self.bitencode(xbin, typesize=1, algorithm=algorithm, shuffle=shuffle))
                lenzcom = len(self.bitencode(zbin, typesize=elem_tsize(latent), algorithm=algorithm, shuffle=shuffle))
                lenrcom  = len(self.bitencode(rbin, typesize=elem_tsize(residual), algorithm=algorithm, shuffle=shuffle))
                
            algcfactor = len(xbin) / lenxcom
            aecfactor = len(xbin) / (lenzcom + lenrcom)
            aeccfactor = lenxcom / (lenzcom + lenrcom)

            benchlist.append([algorithm, len(xbin), len(zbin), 
                          len(rbin), lenxcom, lenzcom, lenrcom, 
                          algcfactor, aecfactor, aeccfactor])
        end = time.time()
        print(f'Benchmark results ready. Total time: {np.round(end - ini, 4)}s')
        benchres = pd.DataFrame(benchlist, columns=['algorithm', 'xbytes', 'zbytes', 'rbytes', 
                                                    'xcbytes', 'zcbytes', 'rcbytes', 
                                                    'algcfactor', 'aecfactor', 'aeccfactor'])
        return benchres