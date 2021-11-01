import os
import allel
import torch
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

class OnlineSimulator:
    def __init__(self, batch_size, n_populations, mode='uniform', device='cpu', species='human', chm=22, split='train', balanced=True, save_vcf_pointer=False, preloaded_data_pointer=None):
        ## Hyperparams set by user.
        self.species = species
        self.chm = chm
        self.split = split
        self.batch_size = batch_size
        self.n_populations = n_populations
        self.mode = mode
        self.device = device
        self.single_ancestry = (n_populations is not None)
        self.balanced = balanced
        self.save_vcf_pointer = save_vcf_pointer
        self.preloaded_data_pointer = preloaded_data_pointer
        
        ## Pre-defined paths.
        self.reference_panel_path = os.path.join(os.environ.get('IN_PATH'),f'data/{self.species}/reference_panel_metadata_v2.tsv')
        self.genetic_map_path = os.path.join(os.environ.get('IN_PATH'), f'data/{self.species}/allchrs.b37.gmap')
        
        if isinstance(self.chm, int):
            self.vcf_file_path = os.path.join(os.environ.get('IN_PATH'), f'data/{self.species}/chr{self.chm}/ref_final_beagle_phased_1kg_hgdp_sgdp_chr{self.chm}_hg19.vcf')
            self.sample_map_path = os.path.join(os.environ.get('IN_PATH'), f'data/{self.species}/maps')
        elif isinstance(self.chm, str) and self.chm == 'all':
            self.vcf_file_path = os.path.join(os.environ.get('IN_PATH'), f'data/{self.species}/allchm/whole_filt_ld_single.vcf')
            self.sample_map_path = os.path.join(os.environ.get('IN_PATH'), f'data/{self.species}/allchm/prepared/{self.split}/sample.map')
        else: raise Exception('Unknown chromosome.')
        
        ## Prepare ancestry list.
        ## Generate single ancestry sample maps for species.
        self.ancestries = self._get_ancestries()
        if self.single_ancestry:
            assert(len(self.ancestries) == self.n_populations)
            
        ## Prepare simulator.
        self._load_founders(single_ancestry=self.single_ancestry)
        
    def _get_ancestries(self):
        ref_panel_metadata = pd.read_csv(self.reference_panel_path, sep="\t", header=0)
        ## Define ancestries (superpop)
        return ref_panel_metadata['Superpopulation code'].unique()
    
    def _load_map_file(self):
        if isinstance(self.chm, int):
            self.mapfiles = {}
            ## Store sample map files in dictionary.
            for i, ancestry in enumerate(self.ancestries):
                sample_map = pd.read_csv(os.path.join(os.environ.get('OUT_PATH'), f'data/{self.species}/chr{self.chm}/prepared/{self.split}/{ancestry}/{ancestry}.map'), sep="\t", header=None, index_col=False)
                sample_map.columns = ["sample", "ancestry"]
                self.mapfiles[ancestry] = {
                    'id' : i,
                    'samples' : sample_map['sample'],
                }
        elif isinstance(self.chm, str) and self.chm == 'all':
            sample_map = pd.read_csv(self.sample_map_path, sep="\t", header=None, index_col=False)
            sample_map.columns = ["sample", "ancestry"]
            self.mapfile = sample_map

    def _load_vcf_samples_in_maps(self):
        ## Concat all available sample names.
        self.samples, self.labels = [], []#, self.ancestry_labels, self.ancestry_names = [], [], []
        if isinstance(self.chm, int):
            for k,v in self.mapfiles.items():
                self.samples += v['samples'].tolist()
                self.labels += (len(v['samples']) * [v['id']])
        elif isinstance(self.chm, str) and self.chm == 'all':
            self.samples = self.mapfile['sample'].values
            self.labels = self.mapfile['ancestry'].values
            
            #metadata = pd.read_csv(self.reference_panel_path, sep="\t", header=0)
            #for i, sample in enumerate(self.samples):
            #    print(f'Sample {sample} has label {self.labels[i]} and is of ancestry {metadata[metadata["Sample"] == sample]["Superpopulation code"].values[0]}')
            #print('\n\n\n\n')
            #idx = np.where(self.labels == 3)[0]
            #self.samples = self.samples[idx]
            #self.labels = self.labels[idx]
            #for i, sample in enumerate(self.samples):
            #    print(f'Sample {sample} has label {self.labels[i]} and is of ancestry {metadata[metadata["Sample"] == sample]["Superpopulation code"].values[0]}')
            
            #print(self.samples)
            #print("Unique: ", np.unique(self.labels))
            
        self.samples = np.asarray(self.samples)
        self.labels = np.asarray(self.labels)
        log.info(f'Read {len(self.samples)} samples from mapfile {self.sample_map_path}')
        log.info(f'Read {len(self.labels)} labels from mapfile {self.sample_map_path}')
        #self.ancestry_names = np.asarray(self.ancestry_names)
        ## Reading VCF.
        
        if self.preloaded_data_pointer is not None:
            log.info(f'Using preloaded data from another online simulator.')
            vcf_data = self.preloaded_data_pointer
        else:
            log.info(f'Loading genotypes from VCF file: {self.vcf_file_path}')
            vcf_data = allel.read_vcf(self.vcf_file_path)
        if self.save_vcf_pointer:
            self.vcf_data = vcf_data
        ## Intersection between samples from VCF and samples from .map.
        log.info(f'Read {len(vcf_data["samples"])} samples from VCF {self.vcf_file_path}')
        samp, idx_inter_vcf, idx_inter_mapfile = np.intersect1d(vcf_data['samples'], self.samples, assume_unique=False, return_indices=True)
        log.info(f'{len(idx_inter_vcf)} samples at intersection of mapfile and VCF.')

        ## Filter only interecting samples.
        self.snps = vcf_data['calldata/GT'].transpose(1,2,0)[idx_inter_vcf,...]
        self.samples_vcf = vcf_data['samples'][idx_inter_vcf]
        log.info(f'Using {len(self.samples_vcf)} {self.species} samples for simulation.')
        
        self.samples = self.samples_vcf# self.samples[idx_inter_mapfile]
        self.labels = self.labels[idx_inter_mapfile]
        
        ## Save header info of VCF file.
        self.info = {
            'chm' : vcf_data['variants/CHROM'],
            'pos' : vcf_data['variants/POS'],
            'id'  : vcf_data['variants/ID'],
            'ref' : vcf_data['variants/REF'],
            'alt' : vcf_data['variants/ALT'],
        }
        
    def _load_founders(self, single_ancestry=True, make_haploid=True, verbose=True):
        ## Load .map files.
        if verbose: log.info('Loading vcf and .map files...')
        self._load_map_file()
        self._load_vcf_samples_in_maps()
        if verbose: log.info('Done loading vcf and .map files...')
        
        ## Map labels from map file to VCF samples
        #argidx = np.argsort(self.samples_vcf)
        #self.samples_vcf = self.samples_vcf[argidx]
        #self.snps = self.snps[argidx, ...]

        #argidx = np.argsort(self.samples)
        #log.info(f'Argidx from self.samples: {len(argidx)}: {argidx}')
        #self.samples = self.samples[argidx]
        #self.labels = self.labels[argidx, ...]

        if verbose:
            log.info(f'A total of {len(self.samples)} diploid individuals where found in the vcf and .map.')
            log.info(f'A total of {len(self.ancestries)} ancestries where found: {self.ancestries}.')

        if make_haploid:
            if isinstance(self.chm, int):
                n_samples, n_seq, n_snps = self.snps.shape
                _ = self.snps.shape
                self.snps = self.snps.reshape(n_samples * n_seq, n_snps)
                log.info(f'Reshaping {len(self.labels)} into {len(self.labels) * 2}')
                self.labels = np.repeat(self.labels, 2)
                log.info(f'Reshaping {_} into {self.snps.shape}')
            elif isinstance(self.chm, str) and self.chm == 'all': 
                n_samples, n_seq, n_snps = self.snps.shape
                _ = self.snps.shape
                log.info(n_samples, n_seq, n_snps)
                self.snps = self.snps.reshape(n_samples * n_seq, n_snps)
                log.info(f'Reshaping {len(self.labels)} into {len(self.labels) * 2}')
                self.labels = np.repeat(self.labels, 2)
                log.info(f'Reshaping {_} snps into {self.snps.shape}')
    
    def _simulate_from_pool(self, batch_snps, batch_labels, batch_size, num_generation_max, num_generation, generation_num_list, rate_per_snp, cM):
        ## If simulate_in_device, batch is moved into device before simulation, 
        ## else is moved after simulation.
        if self.device != 'cpu':
            batch_snps = batch_snps.to(self.device)
            batch_labels = batch_labels.to(self.device)

        ## Select number of generations.
        if self.mode == 'uniform':
            num_generation = np.random.randint(0, num_generation_max)
        elif self.mode == 'exponential':
            num_generation = np.exp(np.random.uniform(0, np.log(num_generation_max)))
        elif self.mode == 'pre-defined':
            num_generation = np.random.choice(generation_num_list)
        elif self.mode == 'fix':
            num_generation = num_generation # Only used for mode 'fix'
        else:
            assert False, 'Simulation mode not valid - use "uniform", "pre-defined", or "fix"'

        ## Obtain number of samples and SNPs - make sure dimensions of labels and SNPs match.
        num_snps = self.snps.shape[1]
            
        ## Obtain a list of switch point indexes
        ## if rate_per_snp (from genetic map) is available, a binomial is sampled.
        if rate_per_snp is not None:
            switch = np.random.binomial(num_generation, rate_per_snp) % 2
            split_point_list = np.flatnonzero(switch)

        ## else if cM is available, a uniform distribution is used
        elif cM is not None:
            switch_per_generation = cM/100
            split_point_list = torch.randint(num_snps, (int(num_generation*switch_per_generation),))  

        ## else if chr number is available, a hardcoded cM (for humans) is used.
        elif (self.chm is not None) and isinstance(self.chm, int) and self.species == 'human':
            cM_list = [286.279234, 268.839622, 223.361095, 214.688476, 204.089357, 192.039918, 187.2205, 168.003442, 166.359329, 181.144008, 158.21865, 174.679023, 125.706316, 120.202583, 141.860238, 134.037726, 128.490529, 117.708923, 107.733846, 108.266934, 62.786478, 74.109562]
            switch_per_generation = cM_list[(int(self.chm)-1)]/100
            split_point_list = torch.randint(num_snps, (int(num_generation*switch_per_generation),))  

        ## else 1 switch per generation is used.
        else:
            split_point_list = torch.randint(num_snps, (int(num_generation*1),))  

        # Perform the simulation    
        for split_point in split_point_list:
            rand_perm = torch.randperm(batch_size)
            batch_snps[:, split_point:] = batch_snps[rand_perm, split_point:]
            batch_labels[:, split_point:] = batch_labels[rand_perm, split_point:]

        return batch_snps, batch_labels
        
    
    def simulate(self, batch_size=None, num_generation_max=100, num_generation=None, generation_num_list=[1,2,4,8,16,32,64,128], rate_per_snp=None, cM=None):
        
        batch_size = self.batch_size if batch_size is None else batch_size
        
        with torch.no_grad():

            ## Make sure input arrays are pytorch tensors.
            if not torch.is_tensor(self.snps):
                self.snps = torch.tensor(self.snps)
            if not torch.is_tensor(self.labels):
                self.labels = torch.tensor(self.labels)

            ## Obtain number of samples and SNPs - make sure dimensions of labels and SNPs match.
            num_samples = self.snps.shape[0]
            num_snps = self.snps.shape[1]

            if len(self.labels.shape) == 1:
                self.labels = self.labels.unsqueeze(1).repeat(1, num_snps)

            ## Obtain batch of samples.
            ## Select samples randomly in the pool.
            if (not self.single_ancestry) and (not self.balanced):
                
                rand_idx = torch.randint(num_samples, (batch_size,))
                batch_snps = self.snps[rand_idx,:]
                batch_labels = self.labels[rand_idx,:]
                
                batch_snps, batch_labels = self._simulate_from_pool(
                    batch_snps=batch_snps, 
                    batch_labels=batch_labels, 
                    batch_size=batch_size, 
                    num_generation_max=num_generation_max, 
                    num_generation=num_generation, 
                    generation_num_list=generation_num_list, 
                    rate_per_snp=rate_per_snp, 
                    cM=cM
                )
            
            ## Select samples randomly in population pool uniformly.
            elif (not self.single_ancestry) and self.balanced: 
                batch_snps = torch.empty(0,num_snps).int()
                batch_labels = torch.empty(0,num_snps).int()

                if self.device != 'cpu':
                    batch_snps = batch_snps.to(self.device)
                    batch_labels = batch_labels.to(self.device)

                for i, ancestry in enumerate(self.ancestries):
                    aux_batch_size = (batch_size // len(self.ancestries)) + (0 if i < len(self.ancestries) - 1 else batch_size % len(self.ancestries))
                    ancestry_idx = np.where(self.labels[:,0] == i)[0]
                    rand_ancestry_idx = torch.randint(len(ancestry_idx), (aux_batch_size,))
                    
                    anc_batch_snps = self.snps[ancestry_idx,:][rand_ancestry_idx,:].int()
                    anc_batch_labels = self.labels[ancestry_idx,:][rand_ancestry_idx,:].int()
                    batch_snps = torch.cat([batch_snps, anc_batch_snps], axis=0)
                    batch_labels = torch.cat([batch_labels, anc_batch_labels], axis=0)
                
                ## Permute ancestries.
                permute_idx = torch.randperm(batch_size)
                batch_snps  = batch_snps[permute_idx,:]
                batch_labels = batch_labels[permute_idx,:]
                
                batch_snps, batch_labels = self._simulate_from_pool(
                    batch_snps=batch_snps, 
                    batch_labels=batch_labels, 
                    batch_size=batch_size, 
                    num_generation_max=num_generation_max, 
                    num_generation=num_generation, 
                    generation_num_list=generation_num_list, 
                    rate_per_snp=rate_per_snp, 
                    cM=cM
                )
            ## 
            elif self.single_ancestry and self.balanced:
                batch_snps = torch.empty(0,num_snps).int()
                batch_labels = torch.empty(0,num_snps).int()

                if self.device != 'cpu':
                    batch_snps = batch_snps.to(self.device)
                    batch_labels = batch_labels.to(self.device)

                for i, ancestry in enumerate(self.ancestries):
                    aux_batch_size = (batch_size // len(self.ancestries)) + (0 if i < len(self.ancestries) - 1 else batch_size % len(self.ancestries))
                    ancestry_idx = np.where(self.labels[:,0] == i)[0]
                    if len(ancestry_idx) == 0:
                        log.info(f'Skipping ancestry {ancestry} as there are no samples.')
                        continue
                    else:
                        log.info(f'Ancestry {ancestry} has {len(ancestry_idx)} indexes.')
                    rand_ancestry_idx = torch.randint(len(ancestry_idx), (aux_batch_size,))
                    
                    anc_batch_snps = self.snps[ancestry_idx,:][rand_ancestry_idx,:].int()
                    anc_batch_labels = self.labels[ancestry_idx,:][rand_ancestry_idx,:].int()
                    anc_batch_snps, anc_batch_labels = self._simulate_from_pool(
                        batch_snps=anc_batch_snps, 
                        batch_labels=anc_batch_labels, 
                        batch_size=aux_batch_size, 
                        num_generation_max=num_generation_max, 
                        num_generation=num_generation, 
                        generation_num_list=generation_num_list, 
                        rate_per_snp=rate_per_snp, 
                        cM=cM
                    )

                    #log.info(batch_snps.dtype, batch_snps.device)
                    #log.info(anc_batch_snps.dtype, anc_batch_snps.device)
                    #log.info('\n')
                    
                    batch_snps = torch.cat([batch_snps, anc_batch_snps], axis=0)
                    batch_labels = torch.cat([batch_labels, anc_batch_labels], axis=0)
            
            ##
            elif self.single_ancestry and (not self.balanced):
                raise Exception('Not implemented.')
            
            return batch_snps.float(), batch_labels.float()