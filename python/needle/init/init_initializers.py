import math
from .init_basic import *
from typing import Any


def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return randn(fan_in, fan_out, mean=0, std=std, **kwargs) 
    ### END YOUR SOLUTION

def kaiming_uniform(fan_in: int, fan_out: int, nonlinearity: str = "relu", shape=None, **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    if nonlinearity == "relu":
        gain = math.sqrt(2.0) 
    else:
        raise NotImplementedError()
    a = gain * math.sqrt(3.0 / fan_in)
    if shape is not None:
        return rand(*shape, low=-a, high=a, **kwargs)
    else:
        return rand(fan_in, fan_out, low=-a, high=a, **kwargs)
    ### END YOUR SOLUTION



def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu", shape=None, **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    if nonlinearity == "relu":
        gain = math.sqrt(2.0) 
    else:
        raise NotImplementedError()
    std = gain / math.sqrt(fan_in) 
    if shape is not None: 
        return randn(*shape, mean=0, std=std, **kwargs) 
    else:
        return randn(fan_in, fan_out, mean=0, std=std, **kwargs) 
    ### END YOUR SOLUTION