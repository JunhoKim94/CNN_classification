import numpy as np
import gensim
import pickle
from tqdm import tqdm
import os
import collections
import re


def load_word2vec(weight_path, corpus_path, word2idx):
    with open(weight_path, "rb") as f:
        weight = pickle.load(f)
    with open(corpus_path, 'rb') as f:
        corpus = pickle.load(f)

    print("Load Complete")

    word_list = word2idx.keys()
    word2idx = dict()
    id_list = []
    for word in word_list:
        if word not in corpus:
            continue
        id_list.append(corpus[word])
        word2idx[word] = len(id_list) - 1

    return weight[id_list], word2idx

#대용량 데이터를 나눠서 corpus를 생성
def raw_corpus(path):
    '''
    file_path = list()
    for path, dir , files in os.walk(path):
        for filename in files:
            file_path.append(path + "/" + filename)

    '''
    char_vocab = dict()
    char_vocab["PAD"] = 0
    word2idx = dict()
    word2idx["UNK"] = 0
    for p in path:
        with open(p, 'rt', encoding= 'latin-1') as f:
            lines = f.readlines()
            for line in lines:
                line = line[2:].strip()
                line = clean_str(line, True)
                for w in line.split(" "):
                    if w not in word2idx:
                        word2idx[w] = len(word2idx)

                    for ch in w:
                        if ch not in char_vocab:
                            char_vocab[ch] = len(char_vocab)

    return word2idx, char_vocab

def batch_words(paths):
    words = []
    for path in paths:
        word = recall_word(path)
        words.append(word)

    return words

def recall_word(path):

    word = []
    target = []
    for p in path:
        with open(p, 'rt', encoding= 'latin-1') as f:
            lines = f.readlines()
            for line in lines:
                target.append(int(line[:1]))
                line = line[2:].strip()
                line = clean_str(line, True)
                word.append(line.split(" "))

    return word, target

def word_id_gen(words, word2idx):
    '''
    words : [lines]
    '''
    word_ = []
    for line in tqdm(words, desc = "Changing Word to Index"):
        line_id = []
        for word in line:
            if word not in word2idx:
                continue
                    #line_id += [word2idx["UNK"]]
            else:
                line_id += [word2idx[word]]
        word_.append(line_id)

    return word_

def padding(words, target, PAD = 0):
    '''
    전체 데이터에서 max_len
    words : [lines] , target 
    '''
    length = [len(s) for s in words]
    max_len = max(length)
    #words = sorted(words, key=lambda items: length)

    train_data = np.zeros((len(length), max_len + 2), dtype = np.int32)
    print(np.mean(length))
    #zeros padding
    for i, line in enumerate(words):
        train_data[i, :length[i]] = line
        train_data[i, -1] = target[i]
        train_data[i, -2] = length[i]

    return train_data, max_len

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()