import os
import argparse

def parse_bool(boolean):
    if isinstance(boolean, bool):
        return keyword
    if boolean.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif boolean.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Wrong keyword.')

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--species', 
        type=str, 
        choices=['human', 'canid'],
        default='human', 
        metavar='STR',
        help='SNP data species: [human], [canid]'
    )
    parser.add_argument('--chr', 
        type=int, 
        default=22, 
        metavar='INT',
        help='Chromosome [chr] number'
    )
    parser.add_argument('--params', 
        type=str, 
        default=os.path.join(os.environ.get('USER_PATH'), 'params.yaml'), 
        metavar='STR',
        help='Path of YAML file with parameters and hyperparameters defined in [STR]'
    )
    parser.add_argument('--num', 
        type=int, 
        default=0, 
        metavar='INT',
        help='#[INT] of the experiment'
    )                 
    parser.add_argument('--verbose', 
        type=parse_bool, 
        default=False, 
        metavar='BOOL',
        help='Verbose output indicator defined by [BOOL]'
    )
    parser.add_argument('--evolution', 
        type=parse_bool, 
        default=False, 
        metavar='BOOL',
        help='PCA evolution plots at wandb defined by [BOOL]'
    )
    parser.add_argument('--only', 
        type=int, 
        default=None, 
        metavar='INT',
        help='Use data only from population by label [INT]'
    )

    return parser