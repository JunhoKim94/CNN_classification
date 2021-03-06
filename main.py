import numpy as np
import torch
import pickle
from preprocess import *
from model.model import Conv_Classifier
import matplotlib.pyplot as plt
import torch.nn.functional as F
from utils import *

print("=" * 20 + "> Training Start < " + "=" * 20)
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
#device = torch.device("cpu")
print(torch.cuda.is_available())
train_type = "multichannel"

#path = ["./data/TREC/TREC.train.all" , "./data/TREC/TREC.test.all"]
path = ["./data/MR/rt-polarity.txt"]
#path = ["./data/Subj/subj.all"]
#path = ["./data/SST-2/stsa.binary.dev","./data/SST-2/stsa.binary.train","./data/SST-2/stsa.binary.test" ]
#path = ["./data/SST-1/stsa.fine.dev", "./data/SST-1/stsa.fine.train", "./data/SST-1/stsa.fine.test"]

word2idx, _ = raw_corpus(path)
Weight, word2idx = load_word2vec("./preweight.pickle", "pre_corpus.pickle", word2idx)
Weight = torch.FloatTensor(Weight).to(device)
#Weight = None

words, target = recall_word(path)
words = word_id_gen(words, word2idx)

data, max_len = padding(words, target)
print(data.shape, len(word2idx), max_len)

'''
test_words , test_target = recall_word(["./data/SST-2/stsa.binary.test"])
test_words = word_id_gen(test_words, word2idx)
test_words, _ = padding(test_words, test_target)
test_words, _ = gen_data(test_words, val_ratio= 0)

x_val , y_val = get_mini(test_words, len(test_words))

print(x_val.shape , y_val.shape)
'''
train_data, val_data = gen_data(data, val_ratio = 0.1)

x_val, y_val = get_mini(val_data, len(val_data))
print(len(train_data), len(val_data))

x_val = torch.tensor(x_val).to(torch.long).to(device)
x_val = x_val.unsqueeze(1)
y_val = torch.tensor(y_val).to(torch.long).to(device)

#Hyper parameters
vocab_size = len(word2idx)
total = len(train_data)
embed_size = 300
h = [(3,100), (4,100), (5,100)]
class_num = max(target) + 1
ch = 2
batch_size = 50
learning_rate = 0.001
epochs = 15

#Model
model = Conv_Classifier(ch, class_num , embed_size, h, vocab_size, Weight, drop_out =  0.5, train_type = train_type, mode = "linear")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-3)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
#torch.nn.utils.clip_grad_norm_(model.parameters(), 3)

model.to(device)
model.train()

loss_stack = []
acc_stack = []
#(B,ch, max_len)
for epoch in range(epochs):
    epoch_loss = 0
    model.train()
    for iteration in range(total // batch_size):
        batch_train, batch_target = get_mini(train_data, batch_size)
        x_train = torch.tensor(batch_train).to(torch.long).to(device)
        x_train = x_train.unsqueeze(1)
        y_train = torch.tensor(batch_target).to(torch.long).to(device)
        y_pred = model(x_train)

        optimizer.zero_grad()
        loss = criterion(y_pred, y_train)
        '''
        l2 = torch.Tensor([0])
        for w in model.parameters():
            l2 += torch.norm(w, p = 2)
        
        loss = loss + lamb * l2
        '''
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()

        epoch_loss += loss.item()
    
        
    scheduler.step()
    epoch_loss /= total
    model.eval()
    y_v = F.log_softmax(model(x_val), dim = 1)
    y_v = torch.argmax(y_v, dim = 1)

    y_t = torch.argmax(F.log_softmax(model(x_train), dim = 1), dim = 1)
    score_train = len(y_train[y_train == y_t]) / len(y_t)
    del x_train, y_pred, y_train

    score = len(y_val[y_val == y_v]) / len(y_v)
    acc_stack.append(score)
    loss_stack.append(epoch_loss)
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    if (epoch % 1 == 0):
        
        print(f"epoch = {epoch} | loss = {epoch_loss} | val_score = {score} | lr = {lr} | train_score : {score_train}")


#plot(acc_stack, loss_stack, epochs)

