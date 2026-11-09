from typing import List
from needle.autograd import Tensor
import needle.backend_ndarray.ndarray as ndarray
from needle import ops
import needle.init as init
import numpy as np
from .nn_sequence import Embedding
from .nn_basic import (
    Parameter, 
    Module, 
    ReLU,
    Dropout,
    LayerNorm1d,
    Linear,
    Sequential
)


class MultiHeadAttention(Module):
    """
    The multi-head self attention module.
    """
    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        self.causal = causal
        self.dropout = Dropout(dropout)

    @staticmethod
    def causal_mask_value(dtype="float32"):
        """Additive value for masked positions in `dtype`: half the largest finite
        value. It is still "minus infinity" for softmax (exp underflows to exactly
        0), but unlike -finfo(float32).max it is finite in float16 (that becomes
        -inf there, and a row of -inf gives NaN after the max subtraction), and the
        factor 1/2 leaves headroom so adding it to a negative logit cannot
        overflow to -inf either."""
        return -np.finfo(dtype).max / 2

    def create_causal_mask(self, i, j, device, dtype="float32"):
        """
        return a triangular causal mask.
        Input: i, j: the shape of the mask to be created
               dtype: dtype of the attention logits the mask is added to
        """
        mask = self.causal_mask_value(dtype) * np.triu(
            np.ones((1, 1, i, j), dtype=dtype), j - i + 1)

        return ndarray.array(
            mask, device=device, dtype=dtype)

    def matmul(self, a, b_transpose):
        """
        batched matrix multiplication a @ b_transpose^T over the last two axes;
        (..., m, d) x (..., n, d) -> (..., m, n).

        Uses the batched MatMul op. (The previous version broadcast both inputs to
        (..., m, n, d), multiplied and summed over d, materializing an m * n * d
        tensor per batch entry.)
        """
        return ops.matmul(a, ops.transpose(b_transpose))

    def softmax(self, logit):
        """
        The softmax function; 
        """
        max_val = Tensor(
            logit.realize_cached_data().max(axis=3),
            device=logit.device,
            dtype=logit.dtype,
            requires_grad=False
        )

        max_val = max_val.reshape((*logit.shape[:-1], 1))
        max_val = max_val.broadcast_to(logit.shape)

        probs = ops.exp(logit - max_val)

        denom = probs.sum(axes=3)
        denom = denom.reshape((*logit.shape[:-1], 1))
        denom = denom.broadcast_to(logit.shape)

        return probs / denom

    def forward(
        self,
        q, k, v,
    ):
        """
        The forward function of the MultiHeadAttention activation function.
        Input: three states q, k, v, with shape (batch_size, num_head, seq_len, dim_head)
        Output: the activation output `result` and attention softmax probability `probs` (with dropout applied)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, keys_values_len, k_dim = k.shape
        _, _, _, v_dim = v.shape

        assert q_dim == k_dim == v_dim

        result = None
        probs = None

        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        qkt = self.matmul(q, k) * (1 / np.sqrt(q_dim)) 
        if self.causal:
            # build the mask in the logits' dtype (float16 under a float16 model,
            # float32 under autocast or a float32 model) so it never mixes dtypes
            mask = Tensor(
                self.create_causal_mask(
                    queries_len, keys_values_len, q.device, dtype=qkt.dtype
                ),
                device=q.device,
                dtype=qkt.dtype,
                requires_grad=False,
            )
            mask = ops.broadcast_to(mask, qkt.shape) 
            qkt += mask 
        a = self.softmax(qkt) 
        probs = self.dropout(a) 
        result = ops.matmul(probs, v) 

        ### END YOUR SOLUTION

        return result, probs


class AttentionLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        *,
        k_features: int = None,
        v_features: int = None,
        out_features: int = None,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        if k_features is None:
            k_features = q_features
        if v_features is None:
            v_features = q_features
        if out_features is None:
            out_features = q_features

        self.q_features = q_features
        self.k_features = k_features
        self.v_features = v_features
        self.out_features = out_features

        self.num_head = num_head
        self.dim_head = dim_head

        self.prenorm_q = LayerNorm1d(
            q_features, device=device, dtype=dtype)
        self.prenorm_k = LayerNorm1d(
            k_features, device=device, dtype=dtype)
        self.prenorm_v = LayerNorm1d(
            v_features, device=device, dtype=dtype)

        inner_dim = num_head * dim_head
        
        self.q_projection = Linear(
            q_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.k_projection = Linear(
            k_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.v_projection = Linear(
            v_features, inner_dim, bias=False,
            device=device, dtype=dtype)

        self.attn = MultiHeadAttention(
            dropout=dropout, causal=causal,
            device=device, dtype=dtype)

        self.out_projection = Linear(
            inner_dim, out_features, bias=False,
            device=device, dtype=dtype)

    def forward(
        self,
        q, k=None, v=None,
    ):
        """
        The forward function of the self-attention layer.
        Input: `q` with shape (batch_size, q_len, q_dim)
               `k` (if not None) with shape (batch_size, kv_len, k_dim)
               `v` (if not None) with shape (batch_size, kv_len, v_dim)
        Output: the output `result` with shape (batch_size, kv_len, out_features)
        """

        if k is None:
            k = q
        if v is None:
            v = q

        batch_size, queries_len, q_dim = q.shape
        _, keys_values_len, k_dim = k.shape
        _, _, v_dim = v.shape

        result = None

        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        q = ops.reshape(q, (batch_size * queries_len, q_dim))
        k = ops.reshape(k, (batch_size * keys_values_len, k_dim))
        v = ops.reshape(v, (batch_size * keys_values_len, v_dim))

        Q = self.q_projection(self.prenorm_q(q)) 
        K = self.k_projection(self.prenorm_k(k)) 
        V = self.v_projection(self.prenorm_v(v)) 

        Q = ops.reshape(Q, (batch_size, queries_len, self.num_head, self.dim_head)) 
        K = ops.reshape(K, (batch_size, keys_values_len, self.num_head, self.dim_head)) 
        V = ops.reshape(V, (batch_size, keys_values_len, self.num_head, self.dim_head)) 

        Q, K, V = ops.transpose(Q, (1, 2)), ops.transpose(K, (1, 2)), ops.transpose(V, (1, 2))

        attn_result, prob = self.attn(Q, K, V)
        # attn_result (B, N, Q_LEN, H_DIM) 
        attn_result = ops.transpose(attn_result, (1, 2)) 
        attn_result = attn_result = ops.reshape(attn_result, (batch_size * queries_len, self.num_head * self.dim_head)
)
        result = self.out_projection(attn_result)
        result = ops.reshape(result, (batch_size, queries_len, self.out_features))
        ### END YOUR SOLUTION

        return result


class TransformerLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        hidden_size: int,
        *,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.dropout = Dropout(dropout) 
        self.attn_layer = AttentionLayer(q_features, num_head, dim_head, causal=causal, dropout=dropout, device=device, dtype=dtype) 
        self.linear1 = Linear(q_features, hidden_size, device=device, dtype=dtype) 
        self.linear2 = Linear(hidden_size, q_features, device=device, dtype=dtype) 
        self.ln = LayerNorm1d(q_features, device=device, dtype=dtype) 
        self.relu = ReLU()
        ### END YOUR SOLUTION

    def forward(
        self,
        x
    ):
        """
        The forward function of a Transformer Layer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """

        batch_size, seq_len, x_dim = x.shape

        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        after_attn = x + self.dropout(self.attn_layer(x)) 
        h = ops.reshape(after_attn, (batch_size * seq_len, x_dim)) 
        h = self.ln(h)
        h = self.linear1(h)
        h = self.relu(h)
        h = self.dropout(h)
        h = self.linear2(h)
        h = self.dropout(h)
        h = ops.reshape(h, (batch_size, seq_len, x_dim))
        x = h + after_attn
        ### END YOUR SOLUTION

        return x


class Transformer(Module):

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int, 
        *,
        num_head: int = 8,
        dim_head: int = 32,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
        batch_first = False,
        sequence_len = 2048
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.batch_first = batch_first

        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.pos_embed = Embedding(sequence_len, embedding_size, device=device, dtype=dtype) 
        self.layers = Sequential(*[
            TransformerLayer(
                embedding_size,
                num_head,
                dim_head,
                hidden_size,
                dropout=dropout,
                causal=causal,
                device=device,
                dtype=dtype
            )
            for _ in range(num_layers)
        ])
        ### END YOUR SOLUTION

    def forward(
        self,
        x, h=None
    ):

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        batch, seq_len, dim = x.shape
        position_ids = np.arange(seq_len, dtype=np.float32)

        position_ids = Tensor(
            position_ids,
            device=x.device,
            dtype=self.dtype,
            requires_grad=False
        )

        position_ids = ops.reshape(position_ids, (1, seq_len))
        position_ids = ops.broadcast_to(
            position_ids, (batch, seq_len)
        )
        pos = self.pos_embed(position_ids) 
        x = x + pos 
        x = self.layers(x)
        ### END YOUR SOLUTION

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        return x, init.zeros_like(x)