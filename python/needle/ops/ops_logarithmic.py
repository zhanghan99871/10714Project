from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        max_z = array_api.max(Z, axis=1, keepdims=True)

        Z_sub = Z - max_z

        log_sum_exp = array_api.log(
            array_api.sum(
                array_api.exp(Z_sub),
                axis=1,
                keepdims=True
            )
        )

        return Z_sub - log_sum_exp
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        input_shape = node.inputs[0].cached_data.shape
        softmax = exp(node) 
        
        grad_sum = summation(out_grad, axes=(1, ))
        grad_sum = reshape(grad_sum, (input_shape[0], 1))
        grad_sum = broadcast_to(grad_sum, input_shape) 
        
        return out_grad - softmax * grad_sum 
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        max_z = array_api.max(Z, axis=self.axes, keepdims=True) 
        Z_sub = Z - max_z 
        Z_sumexp = array_api.sum(array_api.exp(Z_sub), axis=self.axes) 
        max_z = array_api.squeeze(max_z, axis=self.axes)
        return array_api.log(Z_sumexp) + max_z 
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        input_shape = node.inputs[0].cached_data.shape 
        # pad the shape
        if self.axes is None: 
            padded_shape = (1, ) * len(input_shape)
        else:
            padded_shape = tuple(
                1 if i in self.axes else input_shape[i] 
                for i in range(len(input_shape))
            )
        out_grad = reshape(out_grad, padded_shape)
        out_grad = broadcast_to(out_grad, input_shape)
        
        Z = node.inputs[0] 
        result = reshape(node, padded_shape) 
        result = broadcast_to(result, input_shape) 
        
        softmax = exp(Z - result) 
        return out_grad * softmax
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)