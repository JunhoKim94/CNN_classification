import numpy as np
import torch
import pickle
from LM_preprocess import *
from model.model import Conv_Classifier, Conv_LM
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time

print("\n ====================================> Training Start <=========================================")
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
#device = torch.device("cpu")
print(torch.cuda.is_available())

path = "./data/ptb.train.txt"
val_path = "./data/ptb.valid.txt"
test_path = "./data/ptb.test.txt"

word2idx, ch_corpus = raw_corpus(path)
print(len(word2idx), len(ch_corpus))
words = recall_word(path)
words, target = wordtoid(words, word2idx, ch_corpus)

val_words = recall_word(val_path)
val_words, val_target = wordtoid(val_words, word2idx, ch_corpus)

print(val_words.shape)

test_words = recall_word(test_path)
test_words, test_target = wordtoid(test_words, word2idx, ch_corpus)

print(test_words.shape)

batch_size = 20
total = len(words)
epochs = 50
embed = 15
hidden_size = 200
#h = [(1, 25), (2, 25), (3, 25), (4, 25), (5, 25), (6, 25)]
h = [(i, 25 * i) for i in range(1,7)]
lr = 0.001
drop_out = 0.5
num_layer = 2

model = Conv_LM(embed, h, len(ch_corpus), hidden_size, len(word2idx), num_layer, drop_out)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

model.to(device)
st = time.time()

def evaluate(val_words, val_target, model):
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

#val_loss = evaluate(val_words, val_target, model)

for epoch in range(epochs):
    PPL = 0
    epoch_loss = 0
    model.train()
    for iteration in range(total // batch_size):
        batch_x , batch_y = get_mini(words, target, batch_size, iteration)
        length = batch_y.shape[1]

        batch_x = torch.Tensor(batch_x).to(torch.long).to(device)
        batch_y = torch.Tensor(batch_y).to(torch.long).to(device)
        batch_y = batch_y.view(batch_size * length)

        y_pred = model(batch_x)

        loss = criterion(y_pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    #PPL /= (total//batch_size)
    epoch_loss /= (total // batch_size)

    val_loss = evaluate(val_words, val_target, model)


    scheduler.step()
    for param_group in optimizer.param_groups:
        lr = param_group['lr']

    if (epoch % 2 == 0):
        test_loss = evaluate(test_words, test_target, model)
        print(f"epoch = {epoch} |  Val_PPL = {val_loss} | epoch loss : {epoch_loss}  |  lr = {lr} | spend time : {time.time() - st}  |  test_PPL = {test_loss}")
        torch.save(model.state_dict(), "./model.pt")

