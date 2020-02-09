import numpy as np
import gensim
import pickle
from tqdm import tqdm

def save_word2vec(path, save_path):
    path = "./GoogleNews-vectors-negative300.bin"

    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary = True)

    word2idx = dict()
    word2idx["UNK"] = 0
    Weight = model.vectors.astype("float32")
    print(model.vocab)
    for word in model.vocab:
        word2idx[word] = len(word2idx)

    data = dict()
    data["Weight"] = Weight
    data["word2idx"] = word2idx

    with open("./data.pickle", "wb") as f:
        pickle.dump(data, f)

    return data

def load_word2vec(path):
    with open(path, 'r') as f:
        data = pickle.load(f)
    return data["Weight"], data["word2idx"]

def batch_words(paths):
    words = []
    for path in paths:
        word = recall_word(path)
        words.append(word)

    return words

def recall_word(path):

    word = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            word.append(line.split())

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
                    line_id += [word2idx["UNK"]]
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
    
    train_data = np.zeros((len(length), max_len + 1), dtype = np.int32)

    i = 0
    #zeros padding
    for label, lines in enumerate(words):
        for line in lines:
            train_data[i, :length[i]] = line
            train_data[i, length[i] :] = PAD
            train_data[i, -1] = label
            i += 1

    return train_data, max_len

def gen_data(data, val_ratio = 0.3):
    np.random.shuffle(data)
    total = len(data)
    temp = int(val_ratio * total)

    train_data = data[:temp, :]
    val_data = data[temp:,:]

    return train_data, val_data
