import os

import pandas as pd
from torch.utils.data import Dataset

from utils import *


class SaDataset(Dataset):
    def __init__(self, split):
        self.split = split
        assert self.split in {'train', 'valid'}

        if split == 'train':
            filename = os.path.join(train_folder, train_filename)
        elif split == 'valid':
            filename = os.path.join(valid_folder, valid_filename)
        else:
            filename = os.path.join(test_a_folder, test_a_filename)

        user_reviews = pd.read_csv(filename)
        self.samples = user_reviews['content']

    def __getitem__(self, i):
        content = self.samples[i]

        return content

    def __len__(self):
        return len(self.samples)
