import numpy as np


def progress(current, total, batch_size=None, bar=True, time=None, **kwargs):

    indicator = f'Progress: [{current} / {total}]'
    bar_progress = ''
    metrics = ''

    if bar:
        bar_progress += '=' * (current) + '-' * (total - current) 
    
    if time is not None:
        bar_progress += f'({np.round(time, 2)}s)'

    for k, v in kwargs.items():
        metrics += f' {k}: {np.round(np.mean(v), 2)}'
    
    print(f'{indicator} {bar_progress} {metrics}')

    
