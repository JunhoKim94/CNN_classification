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

        if pre_weight is not None:
            self.pre_weight = torch.Tensor(pre_weight)
        else:
            self.pre_weight = None

        #UNK token & padding 모두 0으로 처리
        self.embed = torch.nn.Embedding(self.vocab_size, embed, _weight = self.pre_weight)
        #(B,C,X,Y) 가 인풋
        
        self.conv = []
        
        for h in self.kernel:
            layer = torch.nn.Sequential(
            torch.nn.Conv2d(self.input_ch, self.output_ch, (h, embed)),
            torch.nn.ReLU(),
            )
            
            self.conv.append(layer)
        
        self.conv = torch.nn.ModuleList(self.conv)
        
        self.linear = torch.nn.Sequential(
        torch.nn.Linear(self.output_ch * len(self.kernel), out),
        torch.nn.Dropout(p = drop_out)
        )

    def forward(self, x):
        '''
        x = (Batch, Channel, Sentence(max_len))
        '''
        batch_size = x.shape[0]
        sentence = x.shape[2]
        #(B,ch, S, embed_size)
        out = self.embed(x)

        output = []
        for i, conv in enumerate(self.conv):
            #(B,out_ch, S - h + 1)
            temp = conv(out)
            temp = temp.squeeze(3)
            temp = F.max_pool2d(temp, (1,  sentence - self.kernel[i] + 1))
            temp = temp.squeeze(2)
            output.append(temp)

        #(B, 3 * out_ch, 1)
        out = torch.cat([output[0], output[1], output[2]], dim = 1)


        return self.linear(out)
