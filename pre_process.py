import json
import os
from collections import Counter

import pandas as pd
from tqdm import tqdm

from config import *


def extract(split):
    if split == 'train':
        filename = os.path.join(train_folder, train_filename)
    elif split == 'valid':
        filename = os.path.join(valid_folder, valid_filename)
    else:
        filename = os.path.join(test_a_folder, test_a_filename)

    user_reviews = pd.read_csv(filename)
    print(user_reviews)


def build_wordmap():
    filename = os.path.join(train_folder, train_filename)

    with open(filename, 'r') as f:
        sentences = f.readlines()

    word_freq = Counter()

    for sentence in tqdm(sentences):
        # Update word frequency
        word_freq.update(list(sentence.strip()))

    # Create word map
    words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    word_map = {k: v + 4 for v, k in enumerate(words)}
    word_map['<pad>'] = 0
    word_map['<start>'] = 1
    word_map['<end>'] = 2
    word_map['<unk>'] = 3
    print('len(word_map): ' + str(len(word_map)))
    print(words[:10])

    with open('data/WORDMAP.json', 'w') as file:
        json.dump(word_map, file, indent=4)


if __name__ == '__main__':
    extract('train')
    extract('valid')

    build_wordmap()
