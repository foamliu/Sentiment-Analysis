import os
import zipfile

train_folder = 'data/ai_challenger_fsauor2018_trainingset_20180816'
valid_folder = 'data/ai_challenger_fsauor2018_validationset_20180816'
test_a_folder = 'data/ai_challenger_fsauor2018_testa_20180816'


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def extract(folder):
    filename = '{}.zip'.format(folder)
    print('Extracting {}...'.format(filename))
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall('data')


if __name__ == '__main__':
    ensure_folder('data')

    if not os.path.isdir(train_folder):
        extract(train_folder)

    if not os.path.isdir(valid_folder):
        extract(valid_folder)

    if not os.path.isdir(test_a_folder):
        extract(test_a_folder)
