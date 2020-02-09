import numpy as np
import torch
import pickle
from preprocess import *
from model.CNN import Convolution


print("\n ==============================> Training Start <=============================")
#device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")
print(torch.cuda.is_available())
if torch.cuda.device_count() >= 1:
    print(f"\n ====> Training Start with GPU Number : {torch.cuda.device_count()} GPU Name: {torch.cuda.get_device_name(device=None)}")
else:
    print(f"\n ====> Training Start with CPU Number : {torch.cuda.device_count()} CPU Name: {torch.cuda.get_device_name(device=None)}")


path = ["./data/MR/rt-polarity.neg", "./data/MR/rt-polarity.pos"]

with open("./data.pickle", 'rb') as f:
    save = pickle.load(f)

word2idx = save["word2idx"]
idx2word = save["idx2word"]
count = save["count"]
file_path = save["file_path"]

words = batch_words(path)
words = word_id_gen(words, word2idx)
data, max_len = padding(words)

train_data, val_data = gen_data(data, val_ratio= 0.2)
x_val = torch.Tensor(val_data[:, :-1]).to(torch.long).to(device)
x_val = x_val.unsqueeze(1)
y_val = torch.Tensor(val_data[:, -1]).to(torch.float).to(device)
y_val = y_val.unsqueeze(1)

vocab_size = len(word2idx)
total = len(train_data)
embed_size = 300
h = [2,3,4]
class_num = 2
kernel_num = 100
ch = 1
batch_size = 50
learning_rate = 0.001
epochs = 3

model = Convolution(ch, embed_size, class_num - 1, embed_size, h, max_len, vocab_size)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr = learning_rate)
model.to(device)
model.train()

#(B,ch, max_len)
for epoch in range(epochs):

    epoch_loss = 0
    for iteration in range(total // batch_size):
        seed = np.random.choice(total, batch_size)

        batch_train = train_data[seed]
        x_train = torch.tensor(batch_train[:, :-1]).to(torch.long).to(device)
        x_train = x_train.unsqueeze(1)
        y_train = torch.tensor(batch_train[:, -1]).to(torch.float).to(device)
        y_train = y_train.unsqueeze(1)

        y_pred = model(x_train)
        optimizer.zero_grad()
        loss = criterion(y_pred, y_train)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if iteration % 100 == 0:
            print(f"iteration : {iteration}  |  loss = {epoch_loss / (iteration + 1)}")

    seed = np.random.choice(len(x_val), 100)
    x = x_val[seed]
    y = y_val[seed]

    epoch_loss /= total
    y_v = torch.nn.functional.sigmoid(model(x))
    y_v[y_v >= 0.5] = 1
    y_v[y_v < 0.5] = 0

    score = len(y[y == y_v]) / len(y_v)
    print(f"epoch = {epoch} | loss = {epoch_loss} | val_score = {score}")
    