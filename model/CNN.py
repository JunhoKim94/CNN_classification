import torch
import numpy as np

class Convolution(torch.nn.Module):

    def __init__(self, input_ch, output_ch, out, embed, kernel_window, max_len, vocab_size, pre_weight):
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
        self.embed = torch.nn.Embedding(self.vocab_size, embed, padding_idx= 0, _weight = self.pre_weight)
        #(B,C,X,Y) 가 인풋
        
        self.conv = []
        
        for h in self.kernel:
            layer = torch.nn.Sequential(
                torch.nn.Conv2d(self.input_ch, self.output_ch, (h, embed)),
                torch.nn.MaxPool2d((self.max_len - h + 1,1))
            )
            self.conv.append(layer)
        
        self.conv = torch.nn.ModuleList(self.conv)
        '''

        self.conv1 = torch.nn.Sequential(
                torch.nn.Conv2d(self.input_ch, self.output_ch, (2, embed)),
                torch.nn.MaxPool2d((self.max_len - 2 + 1,1))
            )
        self.conv2 = torch.nn.Sequential(
                torch.nn.Conv2d(self.input_ch, self.output_ch, (3, embed)),
                torch.nn.MaxPool2d((self.max_len - 3 + 1,1))
            )
        self.conv3 = torch.nn.Sequential(
                torch.nn.Conv2d(self.input_ch, self.output_ch, (4, embed)),
                torch.nn.MaxPool2d((self.max_len - 4 + 1,1))
            )
        '''   
        
        self.linear = torch.nn.Sequential(
        torch.nn.Linear(self.output_ch * len(self.kernel), out),
        torch.nn.Dropout(p = 0.5)
        )

    def forward(self, x):
        '''
        x = (Batch, Channel, Sentence(max_len))
        '''
        #(B,ch, S, embed_size)
        out = self.embed(x)
        
        for conv in self.conv:
            #(B,out_ch, S - h + 1)
            temp = conv(out)
            temp = temp.squeeze(3)
            temp = temp.squeeze(2)
            output.append(temp)
        '''
        out1 = self.conv1(out)
        print(out1.shape)
        out2 = self.conv2(out)
        print(out2.shape)
        out3 = self.conv3(out)
        print(out3.shape)
        '''
        #(B, 3 * out_ch, 1)
        out = torch.cat([output[0], output[1], output[2]], dim = 1)
        '''
        out = torch.cat([out1,out2,out3], dim = 1)
        out = out.squeeze(3)
        out = out.squeeze(2)
        print(out.shape)
        '''

        return self.linear(out)