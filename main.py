import numpy as np
import torch
import pickle
from preprocess import *
from model.CNN import Convolution
import matplotlib.pyplot as plt
import torch.nn.functional as F

print("\n ==============================> Training Start <=============================")
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
#device = torch.device("cpu")
print(torch.cuda.is_available())


path = ["./data/custrev.neg", "./data/custrev.pos"]

'''
with open("./data.pickle", 'rb') as f:
    save = pickle.load(f)

word2idx = save["word2idx"]
idx2word = save["idx2word"]
count = save["count"]
file_path = save["file_path"]

'''

#Weight, word2idx = load_word2vec("./preweight.pickle", "pre_corpus.pickle")

Weight = None
word2idx, idx2word = raw_corpus(path)


words = batch_words(path)
words = word_id_gen(words, word2idx)
data, max_len = padding(words)

print(data.shape, len(word2idx), max_len)

train_data, val_data = gen_data(data, val_ratio= 0.1)

x_val, y_val = get_mini(val_data, len(val_data))

x_val = torch.tensor(x_val).to(torch.long).to(device)
x_val = x_val.unsqueeze(1)
y_val = torch.tensor(y_val).to(torch.long).to(device)

#Hyper parameters
vocab_size = len(word2idx)
total = len(train_data)
embed_size = 300
h = [3, 4, 5]
class_num = 2
kernel_num = 100
ch = 1
batch_size = 50
learning_rate = 0.001
epochs = 20

#Model
model = Convolution(ch, kernel_num, class_num , embed_size, h, max_len, vocab_size, Weight, drop_out =  0.5)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
#torch.nn.utils.clip_grad_norm_(model.parameters(), 3)

model.to(device)
model.train()

loss_stack = []
acc_stack = []
#(B,ch, max_len)
for epoch in range(epochs):
    epoch_loss = 0
    for iteration in range(total // batch_size):
        batch_train, batch_target = get_mini(train_data, batch_size)
        x_train = torch.tensor(batch_train).to(torch.long).to(device)
        x_train = x_train.unsqueeze(1)
        y_train = torch.tensor(batch_target).to(torch.long).to(device)
        y_pred = model(x_train,train = True)

        optimizer.zero_grad()
        loss = criterion(y_pred, y_train)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        
    scheduler.step()
    epoch_loss /= total
    y_v = F.log_softmax(model(x_val, False), dim = 1)
    y_v = torch.argmax(y_v, dim = 1)

    y_t = torch.argmax(F.log_softmax(model(x_train,False), dim = 1), dim = 1)
    score_train = len(y_train[y_train == y_t]) / len(y_t)


    score = len(y_val[y_val == y_v]) / len(y_v)
    acc_stack.append(score)
    loss_stack.append(epoch_loss)
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    if (epoch % 5 == 0):
        print(f"epoch = {epoch} | loss = {epoch_loss} | val_score = {score} | lr = {lr} | train_score : {score_train}")
    
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


#plot(acc_stack, loss_stack, epochs)