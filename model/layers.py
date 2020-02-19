import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, embed, vocab_size, pre_weight, train_type ="rand", padding_idx = 0):
        super(Embedding, self).__init__()
        '''
        embed = embed_size of embeding
        vocab_size : total word(n-grams) size
        pre_weight : pre_trained weight
        train_type : "rand", "static", "nonstatic","multichannel"
        '''
        self.embed_size = embed
        self.vocab_size = vocab_size

        self.embedding = nn.ModuleList()
        if train_type.lower() == "rand":
            self.embedding.append(nn.Embedding(self.vocab_size, self.embed_size, padding_idx = padding_idx))

        elif train_type.lower() == "static":
            emb = nn.Embedding(self.vocab_size, self.embed_size, _weight = pre_weight, padding_idx = padding_idx)
            #emb.weight.data = nn.Parameter(pre_weight)
            for params in emb.parameters():
                params.requires_grad = False
            self.embedding.append(emb)

        elif train_type.lower() == "nonstatic":
            emb = nn.Embedding(self.vocab_size, self.embed_size, padding_idx = padding_idx)
            emb.weight.data = nn.Parameter(pre_weight)
            self.embedding.append(emb)

        elif train_type.lower() == "multichannel":
            emb = nn.Embedding(self.vocab_size, self.embed_size, padding_idx = padding_idx)
            emb.weight.data = nn.Parameter(pre_weight)
            for params in emb.parameters():
                params.requires_grad = False
            self.embedding.append(nn.Embedding(self.vocab_size, self.embed_size, padding_idx = padding_idx))
            self.embedding.append(emb)
            
        else:
            print(f"please write right train_type")
            exit(1)

        self.embed_size = 2 * embed if train_type.lower() == "multichannel" else embed