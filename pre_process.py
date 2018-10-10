import json
import os
from collections import Counter

import jieba
import pandas as pd
from tqdm import tqdm

from config import *


def parse_user_reviews(split):
    if split == 'train':
        filename = os.path.join(train_folder, train_filename)
    elif split == 'valid':
        filename = os.path.join(valid_folder, valid_filename)
    else:
        filename = os.path.join(test_a_folder, test_a_filename)

    user_reviews = pd.read_csv(filename)
    id = user_reviews['id']
    content = user_reviews['content']
    location_traffic_convenience = user_reviews['location_traffic_convenience']
    location_distance_from_business_district = user_reviews['location_distance_from_business_district']
    location_easy_to_find = user_reviews['location_easy_to_find']
    service_wait_time = user_reviews['service_wait_time']
    service_waiters_attitude = user_reviews['service_waiters_attitude']
    service_parking_convenience = user_reviews['service_parking_convenience']
    service_serving_speed = user_reviews['service_serving_speed']
    price_level = user_reviews['price_level']
    price_cost_effective = user_reviews['price_cost_effective']
    price_discount = user_reviews['price_discount']
    environment_decoration = user_reviews['environment_decoration']
    environment_noise = user_reviews['environment_noise']
    environment_space = user_reviews['environment_space']
    environment_cleaness = user_reviews['environment_cleaness']
    dish_portion = user_reviews['dish_portion']
    dish_taste = user_reviews['dish_taste']
    dish_look = user_reviews['dish_look']
    dish_recommendation = user_reviews['dish_recommendation']
    others_overall_experience = user_reviews['others_overall_experience']
    others_willing_to_consume_again = user_reviews['others_willing_to_consume_again']

    return user_reviews


def build_wordmap(contents):
    word_freq = Counter()

    for sentence in tqdm(contents):
        seg_list = jieba.cut(sentence.strip())
        # Update word frequency
        word_freq.update(list(seg_list))

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
    user_reviews = parse_user_reviews('train')
    build_wordmap(user_reviews['content'])

    parse_user_reviews('valid')
