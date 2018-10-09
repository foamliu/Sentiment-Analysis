from config import *


def encode_text(word_map, c):
    return [word_map.get(word, word_map['<unk>']) for word in c] + [word_map['<end>']]


# Since we are dealing with batches of padded sequences, we cannot simply consider all elements of
# the tensor when calculating loss. We define maskNLLLoss to calculate our loss based on our
# decoder’s output tensor, the target tensor, and a binary mask tensor describing the padding of the
# target tensor. This loss function calculates the average negative log likelihood of the elements that
# correspond to a 1 in the mask tensor.
def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(input=inp, dim=1, index=target.view(-1, 1)))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()
