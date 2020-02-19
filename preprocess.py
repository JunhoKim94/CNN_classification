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
    collect = collections.Counter()
    
    file_path = path

    word_token = []
    for path in file_path:
        word = []
        with open(path, 'r', encoding= 'utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                line = clean_str(line, True)
                word += line.split(" ")
        word_token += word
        collect.update(word_token)
    
    temp = collect.most_common()
    count = [["UNK", 1]]
    count.extend(temp)

    word2idx = dict()
    data = []
    for word, freq in count:
        data.append(freq)
        word2idx[word] = len(word2idx)
        
    count = data

    return word2idx

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
            line = line.strip()
            line = clean_str(line, True)
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
                if word  not in word2idx:
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

    train_data = np.zeros((len(length), max_len + 2), dtype = np.int32)
    print(np.mean(length))
    i = 0
    #zeros padding
    for label, lines in enumerate(words):
        for line in lines:
            train_data[i, :length[i]] = line
            train_data[i, -1] = label
            train_data[i, -2] = len(line)
            i += 1

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