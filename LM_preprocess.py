import numpy as np
import pickle
from tqdm import tqdm
import re

def wordtoalpha(word, subcorpus, max_len):
    ret = []
    ret.append(subcorpus["BOS"])
    for ch in word:
        if ch not in subcorpus:
            continue
        ret.append(subcorpus[ch])

    ret.append(subcorpus["EOS"])
    while(len(ret) < max_len):
        ret.append(0)

    return [ret]

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
    word2idx["<unk>"] = 0
    with open(path, 'r', encoding= 'utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            #line = clean_str(line, True)
            for w in line.split(" "):
                if w not in word2idx:
                    word2idx[w] = len(word2idx)

                for ch in w:
                    if ch not in char_vocab:
                        char_vocab[ch] = len(char_vocab)

    char_vocab["BOS"] = len(char_vocab)
    char_vocab["EOS"] = len(char_vocab)

    return word2idx, char_vocab

def batch_words(paths):
    words = []
    for path in paths:
        word = recall_word(path)
        words += word

    return words

def recall_word(path):

    word = []
    with open(path, 'r', encoding= 'utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            #line = clean_str(line, True)
            word.append(line.split(" "))

    return word

def wordtoid(words,word2idx,sub_corpus):
    '''
    words = [sen1, sen2, sen3 ....]
    '''
    
    length = [len(k) for k in word2idx.keys()]
    max_len = max(length) + 2
    sen_length = [len(sen) for sen in words]
    max_sen = max(sen_length)
    batch_size = len(words)

    word_id = []
    target_id = []
    for line in tqdm(words, desc = "Changing Word to Index"):
        line_id = []
        stack = []
        for word in line:
            if word  not in word2idx:
                continue
            else:
                line_id += wordtoalpha(word, sub_corpus, max_len)
                stack += [word2idx[word]]
        word_id.append(line_id)
        target_id.append(stack)

    #Batch x sen x Max_len
    
    
    train = np.zeros((batch_size, max_sen , max_len), dtype = np.int32)
    label = np.zeros((batch_size, max_sen + 1), dtype = np.int32)
    for i, sent in enumerate(word_id):
        train[i,:sen_length[i],:] = sent
        label[i,:sen_length[i]] = target_id[i]
        label[i, -1] = sen_length[i]
    return train, label

def get_mini(data, label, batch_size):
    #data = data[:batch_size *len(data)//batch_size]
    #label = label[:batch_size * len(label) // batch_size]
    seed = np.random.choice(len(data), batch_size, replace = False)

    length = label[seed, -1]
    max_length = max(length)

    train_data = data[seed, :(max_length - 1)]
    target = label[seed, 1 :max_length]

    return train_data, target

def clean_str(string, TREC = True):
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