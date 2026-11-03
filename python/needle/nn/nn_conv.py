"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.padding = kernel_size // 2
        fan_in = in_channels * kernel_size ** 2
        fan_out = out_channels * kernel_size ** 2
        self.weight = Parameter(
            init.kaiming_uniform(fan_in, fan_out, shape=(kernel_size, kernel_size, in_channels, out_channels), device=device, 
                        dtype=dtype))
        if bias: 
            bound = 1.0 / np.sqrt(fan_in)

            self.bias = Parameter(
                init.rand(
                    out_channels,
                    low=-bound,
                    high=bound,
                    device=device,
                    dtype=dtype,
                )
            )
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # x (N, C, H, W) 
        x = ops.transpose(ops.transpose(x, (1, 3)), (1, 2)) 
        # x (N, H, W, C) 
        result = ops.conv(x, self.weight, stride=self.stride, padding=self.padding) 
        # result (N, H_o, W_o, C_o) 
        if self.bias is not None: 
            N, H_out, W_out, Cout = result.shape
            bias = ops.reshape(
                self.bias,
                (1, 1, 1, Cout)
            )

            bias = ops.broadcast_to(
                bias,
                result.shape
            )

            result = result + bias
        return ops.transpose(ops.transpose(result, (1, 2)), (1, 3))
        ### END YOUR SOLUTION