import os
import torch
import numpy as np
from torch.utils.data import Dataset

USER_PATH = os.environ.get('USER_PATH')
DATA_PATH = os.path.join(USER_PATH, 'data')

print(DATA_PATH)