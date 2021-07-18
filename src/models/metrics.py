

def create_metrics_dict(metrics, prefix='train'):
    metrics_dict = {}
    if (prefix == 'train') or (prefix == 'tr'):
        prefix = 'tr' 
    elif (prefix == 'valid') or (prefix == 'vd'):
        prefix = 'vd'
    elif prefix == 'aux':
        prefix = 'aux'
    else: raise Exception('Prefix not valid.')
    for kmetric, meta in metrics.items():
        if callable(kmetric):
            for name in meta['outputs']:
                metrics_dict[f'{prefix}_{name}'] = []
        else:
            for p in meta['params']:
                metrics_dict[f'{prefix}_{p}_{kmetric}'] = []
    return metrics_dict

def metacompressor_metric(x,z,o,p=None):
    return 4