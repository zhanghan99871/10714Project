from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # max_z = array_api.max(Z, axis=1, keepdims=True)

        # Z_sub = Z - max_z

        # log_sum_exp = array_api.log(
        #     array_api.sum(
        #         array_api.exp(Z_sub),
        #         axis=1,
        #         keepdims=True
        #     )
        # )

        # return Z_sub - log_sum_exp
        max_z = Z.max(axis=1, keepdims=True)

        # Explicit broadcasting
        max_z = max_z.broadcast_to(Z.shape)

        Z_sub = Z - max_z

        sum_exp = Z_sub.exp().sum(axis=1, keepdims=True)

        log_sum_exp = sum_exp.log()

        # Explicit broadcasting again
        log_sum_exp = log_sum_exp.broadcast_to(Z.shape)

        return Z_sub - log_sum_exp
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        input_shape = node.inputs[0].shape
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
        # max_z = array_api.max(Z, axis=self.axes, keepdims=True) 
        # Z_sub = Z - max_z 
        # Z_sumexp = array_api.sum(array_api.exp(Z_sub), axis=self.axes) 
        # max_z = array_api.squeeze(max_z, axis=self.axes)
        # return array_api.log(Z_sumexp) + max_z 
        ndim = Z.ndim

        if self.axes is None:
            axes = tuple(range(ndim))
        elif isinstance(self.axes, int):
            axes = (self.axes % ndim,)
        else:
            axes = tuple(ax % ndim for ax in self.axes)

        # Step 1: Compute maximum along reduction axes
        max_z = Z

        for ax in sorted(axes, reverse=True):
            max_z = max_z.max(axis=ax, keepdims=True)

        # Step 2: Broadcast maximum to input shape
        max_z_broadcast = max_z.broadcast_to(Z.shape).compact()

        # Step 3: Stable exponential
        exp_z = (Z - max_z_broadcast).exp()

        # Step 4: Sum over reduction axes
        sum_exp = exp_z

        for ax in sorted(axes, reverse=True):
            sum_exp = sum_exp.sum(axis=ax, keepdims=True)

        # Step 5: log(sum(exp(...))) + max
        result = sum_exp.log() + max_z

        # Step 6: Remove reduced dimensions
        output_shape = tuple(
            dim for i, dim in enumerate(Z.shape)
            if i not in axes
        )

        return result.compact().reshape(output_shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        input_shape = node.inputs[0].shape 
        ndim = len(input_shape)
        # pad the shape
        if self.axes is None:
            axes = tuple(range(ndim))
        elif isinstance(self.axes, int):
            axes = (self.axes % ndim,)
        else:
            axes = tuple(ax % ndim for ax in self.axes)
        
        padded_shape = tuple(
                1 if i in axes else input_shape[i] 
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