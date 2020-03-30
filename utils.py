import numpy as np
import gensim
import pickle
from tqdm import tqdm
import os
import collections
import re
import matplotlib.pyplot as plt
import torch

def save_word2vec(path = "./GoogleNews-vectors-negative300.bin"):

    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary = True)

    word2idx = dict()
    Weight = model.vectors.astype("float32")
    for word in model.vocab:
        word2idx[word] = len(word2idx)

    #data = dict()
    #data["Weight"] = Weight
    #data["word2idx"] = word2idx

    with open("./pre_corpus.pickle", "wb") as f:
        pickle.dump(word2idx, f, protocol= pickle.HIGHEST_PROTOCOL)

    with open("./pre_weight.pickle", "wb") as f:
        pickle.dump(Weight, f, protocol = pickle.HIGHEST_PROTOCOL)

def gen_data(data, val_ratio = 0.1):
    np.random.shuffle(data)
    
    total = len(data)
    temp = int(val_ratio * total)
    seed = np.random.choice(total, temp)

    train_data = data[temp:, :]
    val_data = data[:temp,:]

    return train_data, val_data

def get_mini(data, batch_size):
    seed = np.random.choice(len(data), batch_size)

    length = data[seed, -2]
    max_length = max(length)

    train_data = data[seed, :max_length]
    target = data[seed, -1]

    return train_data, target

def plot(acc_stack, loss_stack, epochs):
    a = [i for i in range(epochs)]
    
    fig , ax1 = plt.subplots()
    ax2 = ax1.twinx()
    acc = ax1.plot(a, acc_stack, 'r', label = 'Accuracy')
    loss = ax2.plot(a, loss_stack, 'b', label = 'loss')
    plt.legend()
    ax1.set_xlabel('epochs')
    ax2.set_ylabel('loss')
    ax1.set_ylabel("accuracy")

    ax = acc + loss
    labels = [l.get_label() for l in ax]
    plt.legend(ax, labels, loc =2)

    plt.show()

def evaluate(val_words, val_target, model, device):
    ppl = torch.nn.CrossEntropyLoss(reduction = 'none', ignore_index = 0)
    model.eval()
    batch = 1

    total = 0
    for i in range(len(val_words) // batch):

        val_x , val_y = get_mini(val_words, val_target, batch, i)

        val_x = torch.Tensor(val_x).to(torch.long).to(device)
        val_y = torch.Tensor(val_y).to(torch.long).to(device)

        length = val_y.shape[1]

        val_y = val_y.view(batch * length)

        pred = model(val_x)
        val_loss = ppl(pred, val_y)
        val_loss = val_loss.view(batch, length)
        val_loss = torch.sum(torch.exp(torch.mean(val_loss, dim = 1)))

        total += val_loss.item()

    total /= len(val_words)

    return total



if __name__ == "__main__":
    #save_word2vec()
    print(0)