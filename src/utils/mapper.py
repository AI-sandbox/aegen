import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.environ.get('USER_PATH'), 'src'))
from parser import create_parser

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def single_ancestry_sample_maps(species, reference_panel_version=2):
    if species == 'human':
        ## v2 maps Mozabites to WAS instead of AFR
        file = 'reference_panel_metadata' if reference_panel_version != 2 else 'reference_panel_metadata_v2'
        ref_panel_path = os.path.join(os.environ.get('IN_PATH'),f'data/{species}/{file}.tsv')
        ref_panel_metadata = pd.read_csv(ref_panel_path, sep="\t", header=0)
        ## Define ancestries (superpop)
        ancestries = ref_panel_metadata['Superpopulation code'].unique()
        ## Filter by single-ancestry
        ref_panel_metadata = ref_panel_metadata[ref_panel_metadata['Single_Ancestry']==1].reset_index(drop=True)
        ## For each ancestry, create a sample map
        for ancestry in ancestries:
            ancestry_metadata = ref_panel_metadata[ref_panel_metadata['Superpopulation code'] == ancestry]
            sample_map_info = ancestry_metadata[['Sample', 'Superpopulation code']]
            sample_map_info.to_csv(
                os.path.join(os.environ.get('OUT_PATH'), 
                f'data/{species}/maps/{ancestry}.map'), 
                header=False, 
                sep="\t", 
                index=False
            )
        return ancestries
    else: raise Exception(f'No data for {species} species')


if __name__ == '__main__':
    ## Parse species to create sample map files
    parser = create_parser()
    args = parser.parse_args()
    species = args.species
    chr = args.chr
    
    ## Generate single ancestry sample maps for species
    ancestries = single_ancestry_sample_maps(species, reference_panel_version=2)
    
    ## Ratios defined for TR-VD-TS split
    ratios = [0.8, 0.1, 0.1]
    for ancestry in ancestries:
        log.info(f'Splitting founders of {ancestry}')
        ancestry_df = pd.read_csv(
            os.path.join(os.environ.get('IN_PATH'), f'data/human/maps/{ancestry}.map'),
            delimiter="\t",
            header=None,
            comment="#",
            dtype=str
        )
        data = ancestry_df.sample(frac=1, random_state=123)
        proportions = np.add.accumulate(np.array(ratios) * data.shape[0]).astype(int)
        
        train, valid, test = np.split(data, proportions[:-1])

        log.info(f'\t{train.shape[0]} founders for training')
        log.info(f'\t{valid.shape[0]} founders for validation')
        log.info(f'\t{test.shape[0]} founders for test')

        train.to_csv(os.path.join(os.environ.get('OUT_PATH'), f'data/{species}/chr{chr}/prepared/train/{ancestry}/{ancestry}.map'), sep="\t", header=None, index=False)
        valid.to_csv(os.path.join(os.environ.get('OUT_PATH'), f'data/{species}/chr{chr}/prepared/valid/{ancestry}/{ancestry}.map'), sep="\t", header=None, index=False)
        test.to_csv(os.path.join(os.environ.get('OUT_PATH'), f'data/{species}/chr{chr}/prepared/test/{ancestry}/{ancestry}.map'), sep="\t", header=None, index=False)
