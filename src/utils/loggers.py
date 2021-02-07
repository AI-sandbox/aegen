import numpy as np


def progress(current, total, train=True, bar=True, time=None, **kwargs):

    if train:
        indicator = f'Progress: [{current} / {total}]'
    else:
        indicator = f'Validation:'
        
    bar_progress = ''
    metrics = ''

    if bar:
        bar_progress += '=' * (current) + '-' * (total - current) 
    
    if time is not None:
        bar_progress += f'({np.round(time, 2)}s)'

    for k, v in kwargs.items():
        metrics += f' {k}: {np.round(np.mean(v), 2)}'
    
    print(f'{indicator} {bar_progress} {metrics}')

    
