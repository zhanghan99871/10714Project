"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
from ..autograd import canonical_dtype
from ..autograd import AUTOCAST_LOWER, AUTOCAST_FP32, AUTOCAST_SKIP, autocast_promote

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *

class Cast(TensorOp):
    """Convert a tensor to another floating point dtype, e.g. "float16" / "float32".

    float32 -> float16 rounds to nearest even (and saturates to +-inf above 65504);
    float16 -> float32 is exact. The gradient is treated as the identity and cast
    back to the *input's* dtype, so a float32 leaf used in a float16 computation
    (the "master weight" pattern of mixed precision) still gets a float32 .grad.

    Releasing the input (release_input=True, the default)
    ------------------------------------------------------
    The output node keeps a reference to its input for backward, which would keep
    the input's full-precision buffer alive next to the new copy. Cast's own
    gradient only needs the input's dtype (metadata that survives release), so
    after the cast the input's cached data is released when it is a graph
    intermediate (non-leaf). The memory is returned once nothing else references
    the buffer (an NDArray view of another tensor shares its memory).

    The input still works afterwards: if anything later needs its value (another
    op in forward, or a gradient such as tanh's that reads its output), it is
    recomputed from its own inputs on demand, which costs time. Pass
    release_input=False to keep it, e.g. when the input feeds several ops.

    Never released: leaf tensors (parameters, master weights, tensors you created,
    every tensor in inference / requires_grad=False code, since those cannot be
    recomputed) and anything in LAZY_MODE, where the cast has not run yet.
    """
    autocast_policy = AUTOCAST_SKIP

    def __init__(self, dtype, release_input: bool = True):
        self.dtype = canonical_dtype(dtype)
        self.release_input = release_input

    def __call__(self, a):
        out = super().__call__(a)
        if self.release_input:
            _release_cast_input(a, out)
        return out

    def compute(self, a: NDArray):
        return a.astype(self.dtype)

    def gradient(self, out_grad: Tensor, node: Tensor):
        # gradient tensors are never released: they are what .grad ends up holding
        return cast(out_grad, node.inputs[0].dtype, release_input=False)


def _release_cast_input(a, out) -> None:
    """Free the cast input's buffer once the (eagerly computed) output exists."""
    if (
        isinstance(a, Tensor)
        and not a.is_leaf()
        and a.cached_data is not None
        and out.cached_data is not None  # not LAZY_MODE: the cast has been computed
    ):
        a.release_cached_data()


def cast(a, dtype, release_input: bool = True):
    """Cast `a` to `dtype`; returns `a` itself when it already has that dtype.

    By default a non-leaf input's cached data is released after the cast (see
    Cast); pass release_input=False to keep it.
    """
    if a.dtype == canonical_dtype(dtype):
        return a
    return Cast(dtype, release_input=release_input)(a)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""
    autocast_policy = AUTOCAST_FP32

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # Direct elementwise power, so negative bases with integer
        # exponents work (exp(b * log(a)) is NaN for a < 0).
        return a ** b
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        lhs, rhs = node.inputs
        # d/da a^b = b * a^(b-1); avoids dividing by `lhs` (inf/NaN at a == 0)
        grad_lhs = out_grad * rhs * power(lhs, rhs - 1)
        grad_rhs = out_grad * node * log(lhs)

        return grad_lhs, grad_rhs
        ### END YOUR SOLUTION

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""
    autocast_policy = AUTOCAST_FP32

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        a = node.inputs[0]
        return out_grad * self.scalar * (a ** (self.scalar - 1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return a / b 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        lhs, rhs = node.inputs 
        grad_lhs = out_grad / rhs 
        grad_rhs = out_grad * ( - lhs / power_scalar(rhs, 2))
        return grad_lhs, grad_rhs
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # suppose axes only contain (a,b), where a and b are index to transpose 
        ndim = a.ndim

        if self.axes is None:
            axis1, axis2 = ndim - 2, ndim - 1
        else:
            axis1, axis2 = self.axes

        axis1 %= ndim
        axis2 %= ndim

        # Construct permutation
        perm = list(range(ndim))
        perm[axis1], perm[axis2] = perm[axis2], perm[axis1]

        return a.permute(tuple(perm))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return transpose(out_grad, self.axes) 
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.reshape(a.compact(), self.shape) 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        a_shape = node.inputs[0].shape 
        return reshape(out_grad, a_shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        input_shape = node.inputs[0].shape
        output_shape = self.shape 
        # align input with output 
        
        padded_shape = (1, ) * (len(output_shape) - len(input_shape)) + tuple(input_shape)
        axes = tuple(i for i, (in_dim, out_dim) in enumerate(zip(padded_shape, output_shape)) 
                     if in_dim == 1 and out_dim != 1) 
        res = summation(out_grad, axes) if axes else out_grad
        # in case input has dim 1
        return reshape(res, input_shape) 
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    autocast_policy = AUTOCAST_FP32

    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        ndim = a.ndim

        if self.axes is None:
            return a.sum(axis=None)

        if isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            axes = tuple(self.axes)

        axes = tuple(ax % ndim for ax in axes)
        result = a

        # Descending order prevents axis indices from shifting
        for ax in sorted(axes, reverse=True):
            result = result.sum(axis=ax)

        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
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
        grad = reshape(out_grad, padded_shape)
        return broadcast_to(grad, input_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    autocast_policy = AUTOCAST_LOWER

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return a @ b 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # def sum_to_shape(grad, shape):
        #     cur_shape = grad.cached_data.shape 
        #     padded_shape = (1, ) * (len(cur_shape) - len(shape)) + tuple(shape)
        #     axes = tuple(i for i, (in_dim, out_dim) in enumerate(zip(padded_shape, cur_shape)) 
        #                     if in_dim == 1 and out_dim != 1) 
        #     res = summation(grad, axes) 
        #     return reshape(res, shape) 
        # lhs, rhs = node.inputs 
        # grad_lhs = matmul(out_grad, transpose(rhs))
        # grad_rhs = matmul(transpose(lhs), out_grad)
        # grad_lhs = sum_to_shape(grad_lhs, lhs.cached_data.shape)
        # grad_rhs = sum_to_shape(grad_rhs, rhs.cached_data.shape) 
        # return grad_lhs, grad_rhs
        lhs, rhs = node.inputs

        grad_lhs = matmul(out_grad, transpose(rhs))
        grad_rhs = matmul(transpose(lhs), out_grad)

        return grad_lhs, grad_rhs
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return -a 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return negate(out_grad)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    autocast_policy = AUTOCAST_FP32

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        a = node.inputs[0]
        return out_grad / a
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    autocast_policy = AUTOCAST_FP32

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        a = node.inputs[0] 
        return out_grad * exp(a)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        a = node.inputs[0] 
        mask = Tensor(
            a.realize_cached_data() > 0, 
            device=a.device, 
            dtype=a.dtype, 
            requires_grad=False 
        )
        return out_grad * mask 
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.tanh(a) 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return out_grad * (-(node * node) + 1) 
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # args is a tuple of NDArrays
        shape = args[0].shape
        ndim = len(shape) + 1
        axis = self.axis % ndim
        dtypes = {str(a.dtype) for a in args}
        if len(dtypes) > 1:
            raise TypeError(
                f"stack needs tensors of a single dtype, got {sorted(dtypes)}; "
                "use .astype() to convert"
            )

        # Insert a new dimension
        new_shape = shape[:axis] + (len(args),) + shape[axis:]

        out = array_api.empty(
            new_shape, dtype=args[0].dtype, device=args[0].device
        )
        for i, a in enumerate(args):
            slices = (
                (slice(None),) * axis
                + (slice(i, i + 1),)
                + (slice(None),) * (ndim - axis - 1)
            )
            out[slices] = a.compact().reshape(
                shape[:axis] + (1,) + shape[axis:]
            )

        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return split(out_grad, self.axis) 
        ### END YOUR SOLUTION


def stack(args, axis):
    # under autocast, tensors of mixed dtypes are promoted to the widest one
    return Stack(axis)(make_tuple(*autocast_promote(args)))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        ndim = A.ndim
        axis = self.axis % ndim

        out_shape = A.shape[:axis] + A.shape[axis + 1:]

        outputs = []

        for i in range(A.shape[axis]):
            slices = (
                (slice(None),) * axis
                + (slice(i, i + 1),)
                + (slice(None),) * (ndim - axis - 1)
            )

            # Slicing returns a view with a singleton dimension.
            # Compact before reshape.
            part = A[slices].compact().reshape(out_shape)
            outputs.append(part)

        return tuple(outputs)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return stack(out_grad, self.axis) 
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        ndim = len(a.shape)
        new_shape = tuple([a.shape[i] * (self.dilation + 1) if i in self.axes else a.shape[i] for i in range(ndim)])
        result = array_api.empty(
            new_shape, dtype=a.dtype, device=a.device
        )
        result.fill(0.0)
        new_slice = tuple([slice(None, None, self.dilation+1) if i in self.axes else slice(None) for i in range(ndim)])
        result[new_slice] = a 
        return result 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return undilate(out_grad, self.axes, self.dilation) 
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        ndim = len(a.shape)
        new_slice = tuple([slice(None, None, self.dilation+1) if i in self.axes else slice(None) for i in range(ndim)])
        return a[new_slice].compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)



class Pad(TensorOp):
    """Zero-pad a tensor. `axes` gives (before, after) amounts for every axis."""

    def __init__(self, axes: tuple):
        self.axes = tuple(tuple(a) for a in axes)

    def compute(self, a):
        return a.pad(self.axes)

    def gradient(self, out_grad, node):
        return crop(out_grad, self.axes)


def pad(a, axes):
    return Pad(axes)(a)


class Crop(TensorOp):
    """Remove (before, after) entries from every axis; the inverse of Pad."""

    def __init__(self, axes: tuple):
        self.axes = tuple(tuple(a) for a in axes)

    def compute(self, a):
        slices = tuple(
            slice(lo, dim - hi) for dim, (lo, hi) in zip(a.shape, self.axes)
        )
        return a[slices].compact()

    def gradient(self, out_grad, node):
        return pad(out_grad, self.axes)


def crop(a, axes):
    return Crop(axes)(a)


def im2col(A, kernel_h, kernel_w, stride=1, padding=0):
    """
    A: NDArray with shape (N, H, W, Cin)

    Returns:
        cols: (N * H_out * W_out, Kh * Kw * Cin)
        H_out, W_out
    """
    N, H, W, Cin = A.shape

    if padding > 0:
        A = A.pad((
            (0, 0),
            (padding, padding),
            (padding, padding),
            (0, 0),
        ))

    H_pad = H + 2 * padding
    W_pad = W + 2 * padding

    H_out = (H_pad - kernel_h) // stride + 1
    W_out = (W_pad - kernel_w) // stride + 1

    sN, sH, sW, sC = A.strides

    shape = (
        N, H_out, W_out,
        kernel_h, kernel_w, Cin
    )

    strides = (
        sN,
        sH * stride,
        sW * stride,
        sH,
        sW,
        sC,
    )

    patches = A.as_strided(shape, strides)

    cols = patches.compact().reshape((
        N * H_out * W_out,
        kernel_h * kernel_w * Cin
    ))

    return cols, H_out, W_out


class Conv(TensorOp):
    autocast_policy = AUTOCAST_LOWER

    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        N, H, W, Cin = A.shape
        Kh, Kw, Cin_B, Cout = B.shape
        # Cin == Cin_B
        cols, H_out, W_out = im2col(
            A,
            Kh,
            Kw,
            stride=self.stride,
            padding=self.padding,
        )
        kernel = B.compact().reshape((Kh * Kw * Cin, Cout)) 
        result = cols @ kernel 
        return result.reshape((N, H_out, W_out, Cout)) 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # A (N, H, W, Cin), B (Kh, Kw, Cin, Cout), out_grad (N, H_out, W_out, Cout)
        A, B = node.inputs
        N, H, W, Cin = A.shape
        Kh, Kw, _, Cout = B.shape
        s, p = self.stride, self.padding

        # Undo the stride: put the output gradient back on the stride-1 grid.
        G = dilate(out_grad, (1, 2), s - 1) if s > 1 else out_grad

        # A stride-1 conv over the padded input would produce Hn x Wn outputs.
        # Dilation yields H_out * s rows, which can exceed Hn when
        # (H + 2p - Kh) is not divisible by s; the surplus rows are all zeros
        # (they correspond to positions the strided conv never visited), so
        # crop them off.
        Hn = H + 2 * p - Kh + 1
        Wn = W + 2 * p - Kw + 1
        extra_h, extra_w = G.shape[1] - Hn, G.shape[2] - Wn
        if extra_h > 0 or extra_w > 0:
            G = crop(G, ((0, 0), (0, extra_h), (0, extra_w), (0, 0)))

        # grad wrt A: full convolution of G with the flipped kernel.
        B_flip = flip(transpose(B), (0, 1))  # (Kh, Kw, Cout, Cin)
        if Kh == Kw and Kh - 1 - p >= 0:
            A_grad = conv(G, B_flip, padding=Kh - 1 - p)
        else:
            # Non-square kernel (or padding >= kernel size): pad each spatial
            # axis by its own kernel size, then crop the input padding away.
            G_full = pad(G, ((0, 0), (Kh - 1, Kh - 1), (Kw - 1, Kw - 1), (0, 0)))
            A_grad = conv(G_full, B_flip, padding=0)  # (N, H + 2p, W + 2p, Cin)
            if p > 0:
                A_grad = crop(A_grad, ((0, 0), (p, p), (p, p), (0, 0)))

        # grad wrt B: correlate the (padded) input with G.
        # (Cin, H, W, N) conv (Hn, Wn, N, Cout) -> (Cin, Kh, Kw, Cout)
        B_grad = conv(
            transpose(A, (0, 3)),
            transpose(transpose(G, (0, 2)), (0, 1)),
            padding=p,
        )
        B_grad = transpose(transpose(B_grad, (0, 1)), (1, 2))  # (Kh, Kw, Cin, Cout)

        return A_grad, B_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


