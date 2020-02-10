import numpy as np
import gensim
import pickle
from tqdm import tqdm

def save_word2vec(path = "./GoogleNews-vectors-negative300.bin", save_path = "./pre_corpus.pickle"):

    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary = True)

    word2idx = dict()
    #Weight = model.vectors.astype("float32")
    for word in model.vocab:
        word2idx[word] = len(word2idx)

    #data = dict()
    #data["Weight"] = Weight
    #data["word2idx"] = word2idx

    with open(save_path, "wb") as f:
        pickle.dump(word2idx, f, protocol= pickle.HIGHEST_PROTOCOL)


def load_word2vec(weight_path, corpus_path):
    with open(weight_path, "rb") as f:
        weight = pickle.load(f)
    with open(corpus_path, 'rb') as f:
        corpus = pickle.load(f)
    return weight,corpus

def batch_words(paths):
    words = []
    for path in paths:
        word = recall_word(path)
        words.append(word)

    return words

def recall_word(path):

    word = []
    with open(path, 'r', encoding= 'utf-8') as f:
        lines = f.readlines()
        for line in lines:
            word.append(line.split(" "))

    return word

def word_id_gen(words, word2idx):
    '''
    words : [[class 1(lines)], [class 2(lines)], ...]
    '''
    word_ = []
    for lines in words:
        word_id = []
        for line in tqdm(lines, desc = "Changing Word to Index"):
            line_id = []
            for word in line:
                if word not in word2idx:
                    continue
                    #line_id += [word2idx["UNK"]]
                else:
                    line_id += [word2idx[word]]
            word_id.append(line_id)
        word_.append(word_id)

    return word_

def padding(words, PAD = 0):
    '''
    전체 데이터에서 max_len
    words : [[class 1(lines)], [class 2(lines)], ...]
    '''
    length = []
    for lines in words:
        length += [len(s) for s in lines]
    max_len = max(length)
    #words = sorted(words, key=lambda items: length)

    train_data = np.zeros((len(length), max_len + 1), dtype = np.int32)

    i = 0
    #zeros padding
    for label, lines in enumerate(words):
        for line in lines:
            train_data[i, :length[i]] = line
            train_data[i, -1] = label
            #train_data[i, -2] = len(line)
            i += 1

    return train_data, max_len

def gen_data(data, val_ratio = 0.3):
    np.random.shuffle(data)
    total = len(data)
    temp = int(val_ratio * total)

    train_data = data[:temp, :]
    val_data = data[temp:,:]

    return train_data, val_data

def get_mini(train_data, batch_size):
    length = train_data[:, -2]
    max_length = max(length)

    train = np.zeros((batch_size, max_length), dtype = np.int32)
    target = train_data[:,-1]

    for i, l in enumerate(i, length):
        train[i, :l] = train_data[i, :l]

