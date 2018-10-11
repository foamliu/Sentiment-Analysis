import jieba
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import parse_user_reviews


def analyze(contents):
    sent_lengths = []

    for sentence in tqdm(contents):
        seg_list = list(jieba.cut(sentence.strip()))
        # Update word frequency
        sent_lengths.append(len(seg_list))

    num_bins = 100
    n, bins, patches = plt.hist(sent_lengths, num_bins, facecolor='blue', alpha=0.5)
    plt.show()


if __name__ == '__main__':
    user_reviews = parse_user_reviews('train')
    analyze(user_reviews['content'])
