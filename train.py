import time

from torch import nn
from torch import optim

from data_gen import SaDataset
from models import EncoderRNN
from utils import *


def train(epoch, train_data, encoder, optimizer):
    # Ensure dropout layers are in train mode
    encoder.train()

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    batch_time = ExpoAverageMeter()  # forward prop. + back prop. time
    losses = ExpoAverageMeter()  # loss (per word decoded)
    accs = ExpoAverageMeter()  # accuracy

    start = time.time()

    # Batches
    for i_batch, (input_variable, lengths, target_variable) in enumerate(train_data):
        # Zero gradients
        optimizer.zero_grad()

        # Set device options
        input_variable = input_variable.to(device)
        lengths = lengths.to(device)
        target_variable = target_variable.to(device)
        print('input_variable.size(): ' + str(input_variable.size()))
        print('lengths.size(): ' + str(lengths.size()))
        print('target_variable.size(): ' + str(target_variable.size()))

        # Forward pass through encoder
        encoder_outputs, encoder_hidden = encoder(input_variable, lengths)
        print('encoder_outputs.size(): ' + str(encoder_outputs.size()))

        loss = criterion(encoder_outputs, target_variable)
        loss.backward()

        optimizer.step()

        acc = accuracy(encoder_outputs, target_variable)
        # print('acc: ' + str(acc))

        # Keep track of metrics
        losses.update(loss.item())
        batch_time.update(time.time() - start)
        accs.update(acc)

        start = time.time()

        # Print status
        if i_batch % print_every == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {accs.val:.3f} ({accs.avg:.3f})'.format(epoch, i_batch, len(train_loader),
                                                                    batch_time=batch_time,
                                                                    loss=losses,
                                                                    accs=accs))


def valid(val_data, encoder):
    encoder.eval()  # eval mode (no dropout or batchnorm)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    batch_time = ExpoAverageMeter()  # forward prop. + back prop. time
    losses = ExpoAverageMeter()  # loss (per word decoded)
    accs = ExpoAverageMeter()  # accuracy

    start = time.time()

    with torch.no_grad():
        # Batches
        for i_batch, (x, y_true) in enumerate(val_data):
            # Set device options
            x = x.to(device)
            y_true = y_true.to(device)

            y_pred = encoder(x)

            loss = criterion(y_pred, y_true)

            acc = accuracy(y_pred, y_true)
            # print('acc: ' + str(acc))

            # Keep track of metrics
            losses.update(loss.item())
            batch_time.update(time.time() - start)
            accs.update(acc)

            start = time.time()

            # Print status
            if i_batch % print_every == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {accs.val:.3f} ({accs.avg:.3f})'.format(i_batch, len(val_loader),
                                                                        batch_time=batch_time,
                                                                        loss=losses,
                                                                        accs=accs))

    return accs.avg, losses.avg


def main():
    voc = Lang('data/WORDMAP.json')
    print("voc.n_words: " + str(voc.n_words))

    train_data = SaDataset('train', voc)
    val_data = SaDataset('valid', voc)

    # Initialize encoder
    encoder = EncoderRNN(voc.n_words, hidden_size, encoder_n_layers, dropout)

    # Use appropriate device
    encoder = encoder.to(device)

    # Initialize optimizers
    print('Building optimizers ...')
    optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)

    best_loss = 100000
    epochs_since_improvement = 0

    # Epochs
    for epoch in range(start_epoch, epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(optimizer, 0.8)

        # One epoch's training
        train(epoch, train_data, encoder, optimizer)

        # One epoch's validation
        val_loss = valid(val_data, encoder)
        print('\n * LOSS - {loss:.3f}\n'.format(loss=val_loss))

        # Check if there was an improvement
        is_best = val_loss < best_loss
        best_loss = max(best_loss, val_loss)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, encoder, optimizer, val_loss, is_best)


if __name__ == '__main__':
    main()
