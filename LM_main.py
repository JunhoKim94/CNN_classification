import numpy as np
import torch
import pickle
from LM_preprocess import *
from model.model import Conv_Classifier, Conv_LM
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time

print("=" * 20 + "> Training Start < " + "=" * 20)
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

batch_size = 50
total = len(words)
epochs = 400
embed = 15
hidden_size = 300
#h = [(1, 25), (2, 25), (3, 25), (4, 25), (5, 25), (6, 25)]
h = [(i, 25 * i) for i in range(1,7)]
lr = 0.025
drop_out = 0.5
num_layer = 2

model = Conv_LM(embed, h, len(ch_corpus), hidden_size, len(word2idx), num_layer, drop_out)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 1e-3)
#optimizer = torch.optim.SGD(model.parameters(), lr = lr, weight_decay= 1e-3)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

'''
model.load_state_dict(torch.load("./model.pt"))
model.train()
'''

model.to(device)
st = time.time()
#val_loss = evaluate(val_words, val_target, model, device)
m_l = 40
ppl_last = 1e10

for epoch in range(epochs):
    epoch_loss = 0
    model.train()

    
    for iteration in range(total // batch_size):
        #B,S-1,ch / B,S-1
        batch_x , batch_y = get_mini(words, target, batch_size, iteration)
        length = batch_y.shape[1]
        batch_x = torch.Tensor(batch_x).to(torch.long).to(device)
        batch_y = torch.Tensor(batch_y).to(torch.long).to(device)
        hidden = (torch.zeros(2, batch_size, hidden_size).to(device), torch.zeros(2, batch_size, hidden_size).to(device))
        for i in range(length // m_l):
            
            x_train = batch_x[:, i * m_l : (i+1) * m_l,:].contiguous()
            y_train = batch_y[:, i * m_l : (i+1) * m_l]
            y_train = y_train.contiguous().view(-1)


            hidden = [state.detach() for state in hidden]
            y_pred, hidden = model(x_train, hidden)
            #print(y_pred.shape)

            loss = criterion(y_pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5, norm_type = 2)

            optimizer.step()
            epoch_loss += loss.item()
    
    #PPL /= (total//batch_size)
    epoch_loss /= (total // batch_size)

    if (epoch % 2 == 0):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
        val_loss = evaluate(val_words, val_target, model, device)
        test_loss = evaluate(test_words, test_target, model, device)

        if val_loss > ppl_last:
            scheduler.step()

        print(f"epoch = {epoch} |  Val_PPL = {val_loss} | epoch loss : {epoch_loss}  |  lr = {lr} | spend time : {time.time() - st}  |  test_PPL = {test_loss}")
        torch.save(model.state_dict(), "./model.pt")
        ppl_last = val_loss

        del val_loss, epoch_loss, test_loss
