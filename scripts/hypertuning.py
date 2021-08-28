import os
import yaml
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
        
parser = argparse.ArgumentParser()
parser.add_argument('--ipath', type=str, default='params.yaml')
parser.add_argument('--opath', type=str, default=None)
parser.add_argument('--optimizer', type=str, choices=['Adam', 'AdamW', 'RAdam', 'QHAdam', 'Yogi', 'DiffGrad'], default=None)
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--weight_decay', type=float, default=None)
parser.add_argument('--heads', type=int, default=None)
parser.add_argument('--vqbeta', type=float, default=None)
parser.add_argument('--bottleneck', type=int, default=None)
parser.add_argument('--latent', type=str, choices=['Unknown','Multi-Bernoulli','Uniform','Gaussian'], default=None)
parser.add_argument('--isize', type=int, default=None)
parser.add_argument('--chm', type=int, default=None)
parser.add_argument('--winshare', type=bool, default=None)

args = parser.parse_args()

IPATH = os.path.join(os.environ.get('USER_PATH'), args.ipath)
OPATH = os.path.join(os.environ.get('USER_PATH'), args.opath)
    
with open(IPATH, 'r') as f:
    params_file = yaml.safe_load(f)
    
if args.optimizer is not None:
    params_file['hyperparams']['optimizer']['algorithm'] = args.optimizer
if args.lr is not None:
    params_file['hyperparams']['optimizer']['lr'] = args.lr
if args.weight_decay is not None:
    params_file['hyperparams']['optimizer']['weight_decay'] = args.weight_decay
if args.heads is not None:
    params_file['model']['quantizer']['multi_head']['using'] = (args.heads > 1)
    params_file['model']['quantizer']['multi_head']['features'] = args.heads
if args.vqbeta is not None:
    params_file['model']['quantizer']['beta'] = args.vqbeta
if args.bottleneck is not None:
    params_file['model']['encoder']['layer2']['size'] = args.bottleneck
    params_file['model']['decoder']['layer0']['size'] = args.bottleneck
if args.latent is not None:
    params_file['model']['distribution'] = args.latent
if args.isize is not None:
    params_file['model']['arange']['end'] = args.isize
    params_file['model']['encoder']['layer0']['size'] = args.isize
    params_file['model']['decoder']['layer2']['size'] = args.isize
if args.chm is not None:
    params_file['model']['chm'] = args.chm
if args.winshare is not None: 
    params_file['model']['window_cloning'] = args.winshare   
    
with open(OPATH, 'w') as f:
    yaml.dump(params_file, f)