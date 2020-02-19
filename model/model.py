import torch
import numpy as np
import torch.nn.functional as F
from model.layers import Embedding

class Convolution(Embedding):
    def __init__(self, input_ch, output_ch, out, embed, kernel_window, vocab_size, pre_weight, drop_out, train_type = "raw", padding_idx = 0):
        super(Convolution, self).__init__(embed, vocab_size, pre_weight, train_type, padding_idx)
        '''
        input_ch : input data channel
        output_ch : output_ch after Convolution(filter numbers)
        out : class of train_data (if you want Binary classification out = 1)
        embed : word embedding dimension
        kernel_window = [h1, h2 .. hN] list of convolution window size(h)
        max_len : max_length of input data batch
        '''
        self.input_ch = input_ch
        self.output_ch = output_ch
        self.kernel = kernel_window
        self.out = out
        #(B,C,X,Y) 가 인풋
        
        self.conv = torch.nn.ModuleList([torch.nn.Conv2d(self.input_ch, self.output_ch, (h, self.embed_size), padding = (h-1,0)) for h in self.kernel])
        self.linear = torch.nn.Linear(self.output_ch * len(self.kernel), self.out)
        self.dropout = torch.nn.Dropout(drop_out)

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
        
    def forward(self, x, train = True):
        '''
        x = (Batch, Channel, Sentence(max_len))
        '''
        #(B,ch, S, embed_size)
        out = [layer(x) for layer in self.embedding]
        out = torch.cat(out, dim = 3)
        #(B, output_ch, S-k+1, 1) -->. (B,output_ch, S-k+1)
        output = [F.relu(conv(out)).squeeze(3) for conv in self.conv]
        #(B,output_ch)
        output = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in output]

        #(B, 3 * out_ch)
        out = torch.cat(output, dim = 1)
        if train:
            out = self.dropout(out)
        out = self.linear(out)

        return out
