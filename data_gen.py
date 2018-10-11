import itertools

import jieba
import numpy as np
from torch.utils.data import Dataset

from utils import *


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


# Meaning	    Positive	Neutral	    Negative	Not mentioned
# Old labels    1	        0	        -1	        -2
# New labels    3           2           1           0
def map_sentimental_type(value):
    return value + 2


def parse_user_reviews(user_reviews):
    samples = []
    for i in range(len(user_reviews)):
        content = user_reviews['content'][i]
        label_tensor = np.empty((num_labels,), dtype=np.int32)
        for idx, name in enumerate(label_names):
            sentimental_type = user_reviews[name][i]
            y = map_sentimental_type(sentimental_type)
            # label_tensor[:, idx] = to_categorical(y, num_classes)
            # CrossEntropyLoss does not expect a one-hot encoded vector as the target, but class indices.
            label_tensor[idx] = y
        samples.append({'content': content, 'label_tensor': label_tensor})
    return samples


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


# Returns padded input sequence tensor and lengths
def inputVar(indexes_batch):
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# Returns all items for a given batch of pairs
def batch2TrainData(pair_batch):
    pair_batch.sort(key=lambda x: len(x[0]), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch)
    output = torch.FloatTensor(output_batch)
    return inp, lengths, output


class SaDataset(Dataset):
    def __init__(self, split, voc):
        self.split = split
        self.voc = voc
        assert self.split in {'train', 'valid'}

        if split == 'train':
            filename = os.path.join(train_folder, train_filename)
        elif split == 'valid':
            filename = os.path.join(valid_folder, valid_filename)
        else:
            filename = os.path.join(test_a_folder, test_a_filename)

        user_reviews = pd.read_csv(filename)
        self.samples = parse_user_reviews(user_reviews)
        self.num_chunks = len(self.samples) // chunk_size

    def __getitem__(self, i):
        pair_batch = []

        for i_chunk in range(chunk_size):
            idx = i * chunk_size + i_chunk
            content = self.samples[idx]['content']
            content = content.strip()
            seg_list = jieba.cut(content)
            input_indexes = encode_text(self.voc.word2index, list(seg_list))
            label_tensor = self.samples[idx]['label_tensor']
            pair_batch.append((input_indexes, label_tensor))

        return batch2TrainData(pair_batch)

    def __len__(self):
        return self.num_chunks
