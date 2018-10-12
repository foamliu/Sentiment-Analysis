# import the necessary packages
import json
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
    samples = parse_user_reviews(user_reviews)

    samples = random.sample(samples, 10)
    pair_batch = []
    result = []
    for i, sample in enumerate(samples):
        content = sample['content']
        # print(content)
        result.append({'content': content})
        content = content.strip()
        seg_list = jieba.cut(content)
        input_indexes = encode_text(voc.word2index, list(seg_list))
        label_tensor = sample['label_tensor']
        pair_batch.append((input_indexes, label_tensor))

    test_data = batch2TrainData(pair_batch)
    input_variable, lengths, _ = test_data
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    outputs = encoder(input_variable, lengths)
    _, outputs = torch.max(outputs, 1)
    print('outputs.size(): ' + str(outputs.size()))
    outputs = outputs.cpu().numpy()

    for i in range(10):
        result[i]['labels'] = (outputs[i] - 2).tolist()

    with open('result.json', 'w') as file:
        json.dump(result, file, indent=4)
