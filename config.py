import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Configure training/optimization
learning_rate = 0.0001
min_word_freq = 3
print_every = 100
batch_size = 200
num_labels = 20
num_sentimental_types = 4
save_folder = 'models'

# Configure models
start_epoch = 0
epochs = 120
hidden_size = 500
encoder_n_layers = 2
dropout = 0.05

train_folder = 'data/ai_challenger_sentiment_analysis_trainingset_20180816'
valid_folder = 'data/ai_challenger_sentiment_analysis_validationset_20180816'
test_a_folder = 'data/ai_challenger_sentiment_analysis_testa_20180816'

train_filename = 'sentiment_analysis_trainingset.csv'
valid_filename = 'sentiment_analysis_validationset.csv'
test_a_filename = 'sentiment_analysis_testa.csv'
