"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return (1 + ops.exp(-x)) ** (-1)
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.nonlinearity = nonlinearity
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.device = device
        self.dtype = dtype 
        k = 1 / hidden_size 
        bound = np.sqrt(k)
        self.W_ih = Parameter(
                init.rand(
                    input_size, hidden_size, 
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                )
            )
        self.W_hh = Parameter(
                init.rand(
                    hidden_size, hidden_size, 
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                )
            )
        if bias: 
            self.bias_ih = Parameter(
                    init.rand(
                        hidden_size, 
                        low=-bound,
                        high=bound,
                        device=device,
                        dtype=dtype,
                    )
                )
            self.bias_hh = Parameter(
                    init.rand(
                        hidden_size, 
                        low=-bound,
                        high=bound,
                        device=device,
                        dtype=dtype,
                    )
                )
        else:
            self.bias_ih = None 
            self.bias_hh = None 
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        bs, input_size = X.shape 
        if h is None: 
            # Allocate on the input's device: with device=None the parameters
            # live on init's default device, and Tensor(np...) would otherwise
            # land on a different backend.
            h = init.zeros(bs, self.hidden_size, device=X.device, dtype=self.dtype)
        if self.bias_hh is not None: 
            bias = self.bias_hh + self.bias_ih
            bias = ops.reshape(
                bias, (1, self.hidden_size)
            )

            bias = ops.broadcast_to(
                bias, (bs, self.hidden_size)
            )
            result = h @ self.W_hh + X @ self.W_ih + bias
        else:
            result = h @ self.W_hh + X @ self.W_ih
        if self.nonlinearity == "tanh":
            return ops.tanh(result) 
        elif self.nonlinearity == "relu":
            return ops.relu(result) 
        else:
            raise ValueError("Unsupported nonlinearity")
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.num_layers = num_layers
        self.rnn_cells = [] 
        self.rnn_cells.append(RNNCell(input_size, hidden_size, bias, nonlinearity, device, dtype))
        for i in range(1, num_layers):
            self.rnn_cells.append(RNNCell(hidden_size, hidden_size, bias, nonlinearity, device, dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        seq_len = X.shape[0]
        X_seq = ops.split(X, axis=0) 
        if h0 is not None: 
            h0_layer = list(ops.split(h0, axis=0))
        else:
            h0_layer = [None for i in range(self.num_layers)]
        outputs = []

        for t in range(seq_len):
            x_t = X_seq[t]
            for l in range(self.num_layers):
                h_t = self.rnn_cells[l](x_t, h0_layer[l])
                h0_layer[l] = h_t
                x_t = h_t

            outputs.append(x_t)

        output = ops.stack(outputs, axis=0)
        h_n = ops.stack(h0_layer, axis=0)

        return output, h_n
            
             
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.hidden_size = hidden_size 
        self.device = device 
        self.dtype = dtype 
        k = 1 / hidden_size 
        bound = np.sqrt(k)
        self.W_ih = Parameter(
                init.rand(
                    input_size, 4 * hidden_size, 
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                )
            )
        self.W_hh = Parameter(
                init.rand(
                    hidden_size, 4 * hidden_size, 
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                )
            )
        if bias: 
            self.bias_ih = Parameter(
                    init.rand(
                        4 * hidden_size, 
                        low=-bound,
                        high=bound,
                        device=device,
                        dtype=dtype,
                    )
                )
            self.bias_hh = Parameter(
                    init.rand(
                        4 * hidden_size, 
                        low=-bound,
                        high=bound,
                        device=device,
                        dtype=dtype,
                    )
                )
        else:
            self.bias_hh = None 
            self.bias_ih = None 
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        sig = Sigmoid() 
        bs, input_size = X.shape 
        if h is None: 
            h0 = init.zeros(bs, self.hidden_size, device=X.device, dtype=self.dtype)
            c0 = init.zeros(bs, self.hidden_size, device=X.device, dtype=self.dtype)
        else:
            h0, c0 = h 
        if self.bias_hh is not None: 
            bias = self.bias_hh + self.bias_ih
            bias = ops.reshape(
                bias, (1, 4 * self.hidden_size)
            )

            bias = ops.broadcast_to(
                bias, (bs, 4 * self.hidden_size)
            )
            result = h0 @ self.W_hh + X @ self.W_ih + bias
        else:
            result = h0 @ self.W_hh + X @ self.W_ih
        result = ops.reshape(result, (bs, 4, self.hidden_size)) 
        i, f, g, o = ops.split(result, axis=1) 
        i, f, g, o, = sig(i), sig(f), ops.tanh(g), sig(o) 
        new_c = f * c0 + i * g 
        new_h = o * ops.tanh(new_c) 
        return new_h, new_c
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.num_layers = num_layers
        self.lstm_cells = [] 
        self.lstm_cells.append(LSTMCell(input_size, hidden_size, bias, device, dtype))
        for i in range(1, num_layers):
            self.lstm_cells.append(LSTMCell(hidden_size, hidden_size, bias, device, dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        seq_len = X.shape[0]
        X_seq = ops.split(X, axis=0) 
        if h is not None: 
            h0, c0 = h 
            states = list(zip(list(ops.split(h0, axis=0)), list(ops.split(c0, axis=0))))
        else:
            states = [None for i in range(self.num_layers)]
        outputs = []

        for t in range(seq_len):
            x_t = X_seq[t]
            for l in range(self.num_layers):
                h_t, c_t = self.lstm_cells[l](x_t, states[l])
                states[l] = (h_t, c_t)
                x_t = h_t

            outputs.append(x_t)

        output = ops.stack(outputs, axis=0)
        h_n = ops.stack([states[i][0] for i in range(len(states))], axis=0)
        c_n = ops.stack([states[i][1] for i in range(len(states))], axis=0)

        return output, (h_n, c_n)
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.weight = Parameter(
                init.randn(
                    num_embeddings, embedding_dim, 
                    mean=0,
                    std=1,
                    device=device,
                    dtype=dtype,
                )
            )
        self.num_embeddings = num_embeddings 
        self.embedding_dim = embedding_dim
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        seq_len, bs = x.shape 
        one_hot = init.one_hot(
            self.num_embeddings,
            x,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        one_hot = ops.reshape(
            one_hot,
            (seq_len * bs, self.num_embeddings),
        )
        embeddings = one_hot @ self.weight 
        return ops.reshape(embeddings, (seq_len, bs, self.embedding_dim))
        ### END YOUR SOLUTION