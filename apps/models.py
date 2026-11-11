import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
from needle import ops
np.random.seed(0)


class ConvBN(ndl.nn.Module):
    def __init__(self, in_c, out_c, k, s, device=None, dtype="float32"):
        super().__init__()
        self.conv = nn.Conv(in_c, out_c, k, stride=s, device=device, dtype=dtype) 
        self.bn = nn.BatchNorm2d(out_c, device=device, dtype=dtype)  
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x) 
        x = self.bn(x) 
        x = self.relu(x) 
        return x 

class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        # raise NotImplementedError() ###
        self.l1 = ConvBN(3, 16, 7, 4, device=device, dtype=dtype) 
        self.l2 = ConvBN(16, 32, 3, 2, device=device, dtype=dtype) 
        self.l3 = ConvBN(32, 32, 3, 1, device=device, dtype=dtype) 
        self.l4 = ConvBN(32, 32, 3, 1, device=device, dtype=dtype)
        self.l5 = ConvBN(32, 64, 3, 2, device=device, dtype=dtype) 
        self.l6 = ConvBN(64, 128, 3, 2, device=device, dtype=dtype) 
        self.l7 = ConvBN(128, 128, 3, 1, device=device, dtype=dtype) 
        self.l8 = ConvBN(128, 128, 3, 1, device=device, dtype=dtype) 
        self.linear = nn.Linear(128, 128, device=device, dtype=dtype) 
        self.relu = nn.ReLU()
        self.final = nn.Linear(128, 10, device=device, dtype=dtype) 
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        x = self.l1(x) 
        ck1 = self.l2(x) 
        x = self.l3(ck1) 
        x = self.l4(x) 
        x = x + ck1 
        x = self.l5(x) 
        ck2 = self.l6(x) 
        x = self.l7(ck2) 
        x = self.l8(x) 
        x = x + ck2 
        x = nn.Flatten()(x) 
        x = self.linear(x) 
        x = self.relu(x) 
        x = self.final(x) 
        return x 
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.embedding_layer = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        if seq_model == 'rnn':
            self.sequence_model = nn.RNN(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        elif seq_model == 'lstm':
            self.sequence_model = nn.LSTM(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        elif seq_model == 'transformer':
            self.sequence_model = nn.Transformer(embedding_size, hidden_size, num_layers=num_layers, device=device, dtype=dtype)
        if seq_model == 'transformer':
            self.final_proj = nn.Linear(embedding_size, output_size, device=device, dtype=dtype)
        else:
            self.final_proj = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        self.is_transformer = seq_model == 'transformer'
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        seq_len, bs = x.shape 
        x = self.embedding_layer(x) 
        out, h = self.sequence_model(x, h)  
        if self.is_transformer:
            out = ops.reshape(out, (seq_len * bs, self.embedding_size)) 
        else:
            out = ops.reshape(out, (seq_len * bs, self.hidden_size)) 
        out = self.final_proj(out) 
        return out, h 
        ### END YOUR SOLUTION

