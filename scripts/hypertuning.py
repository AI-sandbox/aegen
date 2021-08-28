import os
import yaml
import argparse

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
    
with open(OPATH, 'w') as f:
    yaml.dump(params_file, f)