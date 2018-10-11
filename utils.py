import datetime
import json
import os
import time

import pandas as pd

from config import *


def encode_text(word_map, c):
    return [word_map.get(word, word_map['<unk>']) for word in c] + [word_map['<end>']]


class Lang:
    def __init__(self, filename):
        word_map = json.load(open(filename, 'r'))
        self.word2index = word_map
        self.index2word = {v: k for k, v in word_map.items()}
        self.n_words = len(word_map)


# Exponentially weighted averages
class ExpoAverageMeter(object):
    # Exponential Weighted Average Meter
    def __init__(self, beta=0.9):
        self.reset()

    def reset(self):
        self.beta = 0.9
        self.val = 0
        self.avg = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.avg = self.beta * self.avg + (1 - self.beta) * self.val


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def save_checkpoint(epoch, encoder, optimizer, val_acc, is_best):
    ensure_folder(save_folder)
    state = {'encoder': encoder,
             'optimizer': optimizer}

    if is_best:
        filename = '{0}/checkpoint_{1}_{2:.3f}.tar'.format(save_folder, epoch, val_acc)
        torch.save(state, filename)

        # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
        torch.save(state, '{}/BEST_checkpoint.tar'.format(save_folder))


def encode_text(word_map, c):
    return [word_map.get(word, word_map['<unk>']) for word in c] + [word_map['<end>']]


def accuracy(scores, targets, k=1):
    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    # print('correct: ' + str(correct))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def parse_user_reviews(split):
    if split == 'train':
        filename = os.path.join(train_folder, train_filename)
    elif split == 'valid':
        filename = os.path.join(valid_folder, valid_filename)
    else:
        filename = os.path.join(test_a_folder, test_a_filename)

    user_reviews = pd.read_csv(filename)
    return user_reviews


def timestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
