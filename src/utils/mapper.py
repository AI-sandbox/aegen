import os
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

if __name__ == '__main__':
    ratios = [0.8, 0.1, 0.1]
    for ancestry in ['EUR', 'EAS', 'AMR', 'SAS', 'AFR', 'OCE', 'WAS']:
        log.info(f'Splitting founders of {ancestry}')
        ancestry_df = pd.read_csv(
            os.path.join(os.environ.get('IN_PATH'), f'data/human/chr22/{ancestry}.map'),
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

        train.to_csv(os.path.join(os.environ.get('OUT_PATH'), f'data/human/chr22/prepared/train/{ancestry}/{ancestry}.map'), sep="\t", header=None, index=False)
        valid.to_csv(os.path.join(os.environ.get('OUT_PATH'), f'data/human/chr22/prepared/valid/{ancestry}/{ancestry}.map'), sep="\t", header=None, index=False)
        test.to_csv(os.path.join(os.environ.get('OUT_PATH'), f'data/human/chr22/prepared/test/{ancestry}/{ancestry}.map'), sep="\t", header=None, index=False)