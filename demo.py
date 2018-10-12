# import the necessary packages
import os
import random

import jieba
import pandas as pd
import torch

from config import device, save_folder, valid_folder, valid_filename
from data_gen import parse_user_reviews, batch2TrainData
from utils import Lang, encode_text

if __name__ == '__main__':
    voc = Lang('data/WORDMAP.json')
    print("voc.n_words: " + str(voc.n_words))

    checkpoint = '{}/BEST_checkpoint.tar'.format(save_folder)  # model checkpoint
    print('checkpoint: ' + str(checkpoint))

    # Load model
    checkpoint = torch.load(checkpoint)
    encoder = checkpoint['encoder']

    # Use appropriate device
    encoder = encoder.to(device)

    # Set dropout layers to eval mode
    encoder.eval()

    filename = os.path.join(valid_folder, valid_filename)
    user_reviews = pd.read_csv(filename)
    samples = parse_user_reviews('test_a')

    samples = random.sample(samples, 10)
    pair_batch = []
    for i, sample in enumerate(samples):
        content = sample['content']
        content = content.strip()
        seg_list = jieba.cut(content)
        input_indexes = encode_text(voc.word2index, list(seg_list))
        label_tensor = sample['label_tensor']
        pair_batch.append((input_indexes, label_tensor))

    test_data = batch2TrainData(pair_batch)
    input_variable, lengths, _ = test_data
    outputs = encoder(input_variable, lengths)
    print('outputs.size(): ' + str(outputs.size()))
