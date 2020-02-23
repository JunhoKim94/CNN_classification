import torch
import numpy as np
import torch.nn.functional as F
from model.layers import Embedding, Highway
import torch.nn as nn


class Conv_LM(Conv_Classifier):
    def __init__(self, output_ch, embed, kernel, vocab_size, hidden, output, num_layers, dropout):
        super(Conv_LM, self).__init__(1,0,embed,kernel,vocab_size,None, dropout,"raw",0,"highway")
        self.hidden = hidden
        self.output = output
        self.input = self.conv.output_ch * len(self.conv.kernel)

        self.rnn = nn.LSTM(self.input, hidden, num_layers, dropout = dropout)
        self.out_linear = nn.Linear(self.hidden, self.output)

    
    def forward(self, x, h):
        '''
        x = (B, ch, S, word_len)
        '''
        sentence = x.shape[2]

        #B,ch, S, Word_len, embed
        out = self.embed(x)
        for i in range(sentence):
            x_ = out[:,:,i,:,:]
            output = [F.tanh(conv(x_)).squeeze(3) for conv in self.conv]
            #(B,output_ch)
            output = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in output]

            #(B, 3 * out_ch)
            out = torch.cat(output, dim = 1)
            out = self.dropout(out)
            out = self.linear(out)

            out, h = self.rnn(out, h)
            out = self.out_linear(out)
        
    


class Conv_Classifier(Embedding):
    def __init__(self, input_ch, output_ch, out, embed, kernel_window, vocab_size, pre_weight, drop_out, train_type = "raw", padding_idx = 0, mode = "linear"):
        super(Conv_Classifier, self).__init__(embed, vocab_size, pre_weight, train_type, padding_idx)
        '''
        input_ch : input data channel
        output_ch : output_ch after Convolution(filter numbers)
        out : class of train_data
        embed : word embedding dimension
        kernel_window = [h1, h2 .. hN] list of convolution window size(h)
        max_len : max_length of input data batch
        '''
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.kernel = kernel_window
        self.out = out
        #(B,C,X,Y) 가 인풋
        
        self.conv = nn.ModuleList([nn.Conv2d(self.input_ch, self.output_ch, (h, self.embed_size), padding = (h-1,0)) for h in self.kernel])
        
        if mode.lower() == "highway":
            self.linear = Highway(self.output_ch * len(self.kernel))
        else:
            self.linear = nn.Linear(self.output_ch * len(self.kernel), self.out)
        self.dropout = nn.Dropout(drop_out)

        self.init_weight()

    def init_weight(self):
        #for layer in self.embedding:
        #    layer.weight.data.uniform_(-0.01,0.01)
        
        for layer in self.conv:
            layer.weight.data.uniform_(-0.01, 0.01)
            #torch.nn.init.kaiming_uniform_(layer.weight)
        self.linear.weight.data.uniform_(-0.01, 0.01)
        #torch.nn.init.kaiming_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0)
        
    def forward(self, x):
        '''
        x = (Batch, Channel, Sentence(max_len))
        '''
        #(B,ch, S, embed_size)
        out = [layer(x) for layer in self.embedding]
        out = torch.cat(out, dim = 1)
        print(out.shape)
        #(B, output_ch, S-k+1, 1) -->. (B,output_ch, S-k+1)
        output = [F.relu(conv(out)).squeeze(3) for conv in self.conv]
        #(B,output_ch)
        output = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in output]

        #(B, 3 * out_ch)
        out = torch.cat(output, dim = 1)
        out = self.dropout(out)
        out = self.linear(out)
            

        return out
