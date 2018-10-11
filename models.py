import torch
import torch.nn as nn

from config import num_labels


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
        self.fc = nn.Linear(hidden_size, num_labels)

    def forward(self, input_seq, input_lengths, hidden=None):
        # input_seq = [sent len, batch size]
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # embedded = [sent len, batch size, hidden size]
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # outputs = [sent len, batch size, hidden size]
        outputs = outputs[-1]
        # outputs = [batch size, hidden size]
        outputs = self.fc(outputs)
        # outputs = [batch size, num_labels]

        # Return output
        return outputs
