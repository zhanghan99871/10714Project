"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        self.need_bias = bias 
        if bias: 
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).reshape((out_features,)))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        res = X @ self.weight
        if self.need_bias: 
            bias_shape = (1,) * (len(res.shape) - 1) + (self.out_features,)
            bias = self.bias.reshape(bias_shape)
            bias = bias.broadcast_to(res.shape)
            return res + bias 
        else:
            return res 
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        total = 1 
        batch = X.shape[0] 
        for each in X.shape[1:]:
            total *= each 
        return X.reshape((batch,total))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return ops.relu(x) 
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        for each in self.modules:
            x = each(x) 
        return x 
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        batch, cls = logits.shape
        y_one_hot = init.one_hot(cls, y) 
        new_y = ops.summation(y_one_hot * logits, axes=(1, ))
        per_batch_loss = ops.logsumexp(logits, axes=(1,)) - new_y
        return ops.summation(per_batch_loss, axes=(0, )) / batch
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True)) 
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True)) 
        self.running_var = init.ones(dim, device=device, dtype=dtype, requires_grad=False)
        self.running_mean = init.zeros(dim, device=device, dtype=dtype, requires_grad=False)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        batch, dim = x.shape 
        w = self.weight.reshape((1, dim)).broadcast_to(x.shape) 
        b = self.bias.reshape((1, dim)).broadcast_to(x.shape) 
        if self.training:
            mean = ops.summation(x, axes=(0, )) / batch 
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            mean = mean.reshape((1, dim)).broadcast_to(x.shape) 
            var = ops.summation((x - mean) ** 2, axes=(0, )) / batch
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data
            var = var.reshape((1, dim)).broadcast_to(x.shape) 
            return w * (x - mean) / ((var + self.eps) ** 0.5) + b
        else:
            mean = self.running_mean.reshape((1, dim)).broadcast_to(x.shape) 
            var = self.running_var.reshape((1, dim)).broadcast_to(x.shape) 
            return w * (x - mean) / ((var + self.eps) ** 0.5) + b
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True)) 
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True)) 
        
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        batch, dim = x.shape 
        mean = ops.summation(x, axes=(1, )) / dim 
        mean = mean.reshape((batch, 1)).broadcast_to(x.shape) 
        var = ops.summation((x - mean) ** 2, axes=(1, )) / dim 
        var = var.reshape((batch, 1)).broadcast_to(x.shape) 
        w = self.weight.reshape((1, dim)).broadcast_to(x.shape) 
        b = self.bias.reshape((1, dim)).broadcast_to(x.shape) 
        return w * (x - mean) / ((var + self.eps) ** 0.5) + b
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        if self.training:
            mask = init.randb(*x.shape, p=(1-self.p), device=x.device, dtype=x.dtype, requires_grad=False) 
            return x * mask / (1 - self.p) 
        else:
            return x 
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return x + self.fn(x) 
        ### END YOUR SOLUTION
