import torch
import numpy as np
import torch.nn.functional as F

class Convolution(torch.nn.Module):

    def __init__(self, input_ch, output_ch, out, embed, kernel_window, max_len, vocab_size, pre_weight, drop_out):
        super(Convolution, self).__init__()
        '''
        input_ch : input data channel
        output_ch : output_ch after Convolution(filter numbers)
        out : class of train_data (if you want Binary classification out = 1)
        embed : word embedding dimension
        kernel_window = [h1, h2 .. hN] list of convolution window size(h)
        max_len : max_length of input data batch
        '''
        self.input_ch = input_ch
        self.max_len = max_len
        self.output_ch = output_ch
        self.kernel = kernel_window
        self.vocab_size = vocab_size
        self.out = out
        self.emb_size = embed

        if pre_weight is not None:
            self.embed = torch.nn.Embedding(self.vocab_size, embed, _weight = pre_weight, padding_idx = 0)
        else:
            self.embed = torch.nn.Embedding(self.vocab_size, embed, padding_idx = 0)

        #UNK token & padding 모두 0으로 처리
        
        #(B,C,X,Y) 가 인풋
         
        self.conv = torch.nn.ModuleList([torch.nn.Conv2d(self.input_ch, self.output_ch, (h, embed), padding=(h-1,0)) for h in self.kernel])
        self.linear = torch.nn.Linear(self.output_ch * len(self.kernel), self.out)
        self.dropout = torch.nn.Dropout(drop_out)

        self.init_weight()

    def init_weight(self):
        self.embed.weight.data.uniform_(-0.01,0.01)
        #self.embed.weight = torch.nn.Parameter(torch.FloatTensor(np.random.uniform(-0.01,0.01,size = (self.vocab_size,self.emb_size))))
        

        for layer in self.conv:
            #layer.weight = torch.nn.Parameter(torch.FloatTensor(np.random.uniform(-0.01, 0.01,(self.output_ch, 1, self.kernel[i],self.emb_size))))
            layer.weight.data.uniform_(-0.25, 0.25)

        self.linear.weight.data.uniform_(-0.01, 0.01)
        self.linear.bias.data.fill_(0)
        #self.linear.weight = torch.nn.Parameter(torch.FloatTensor(np.random.uniform(-0.01,0.01, (self.out,self.output_ch * len(self.kernel)))))


    def forward(self, x, train = True):
        '''
        x = (Batch, Channel, Sentence(max_len))
        '''
        batch_size = x.shape[0]
        #sentence = x.shape[2]
        #(B,ch, S, embed_size)
        out = self.embed(x)
        #out = out.unsqueeze(1)

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
