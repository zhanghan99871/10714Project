"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        for param in self.params:
            grad = param.grad.data + self.weight_decay * param.data 
            if param not in self.u: 
                ut = 0
            else:
                ut = self.u[param]
            ut = self.momentum * ut + (1 - self.momentum) * grad 
            param.data -= self.lr * ut 
            self.u[param] = ut 
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        Note: This does not need to be implemented for HW2 and can be skipped.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        self.t += 1
        for param in self.params: 
            grad = param.grad.data + self.weight_decay * param.data 
            if param not in self.m:
                mt = 0
            else:
                mt = self.m[param]
            if param not in self.v:
                vt = 0
            else:
                vt = self.v[param] 
            mt = self.beta1 * mt + (1 - self.beta1) * grad 
            vt = self.beta2 * vt + (1 - self.beta2) * (grad ** 2) 
            self.v[param] = vt
            self.m[param] = mt 
            vt /= 1 - self.beta2 ** self.t 
            mt /= 1 - self.beta1 ** self.t  
            param.data -= self.lr * mt / (vt**0.5 + self.eps) 
        ### END YOUR SOLUTION
