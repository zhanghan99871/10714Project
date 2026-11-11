"""Core data structures."""
import needle
from .backend_numpy import Device, all_devices
from typing import List, Optional, NamedTuple, Tuple, Union, Dict, TYPE_CHECKING
from collections import namedtuple
import numpy
import contextlib

from needle import init

# needle version
LAZY_MODE = False
TENSOR_COUNTER = 0

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

import numpy as array_api
NDArray = numpy.ndarray

from .backend_selection import array_api, NDArray, default_device, cpu


def canonical_dtype(dtype) -> Optional[str]:
    """Canonical dtype name ("float32", "float16", ...); None stays None.

    Accepts strings, numpy dtypes and numpy scalar types, so "float16",
    numpy.float16 and numpy.dtype("float16") all compare equal afterwards.
    """
    if dtype is None:
        return None
    return numpy.dtype(dtype).name


def _check_input_dtypes(op: "Op", inputs: List["Value"]) -> None:
    """Ops compute on a single dtype; mixing float16 and float32 inputs must be
    an explicit cast rather than a backend assertion deep inside compute()."""
    dtypes = {x.dtype for x in inputs if isinstance(x, Tensor)}
    if len(dtypes) > 1:
        raise TypeError(
            f"{type(op).__name__} got inputs with different dtypes {sorted(dtypes)}; "
            "cast explicitly with tensor.astype(dtype) or ops.cast(tensor, dtype)"
        )


##############################
####### Autocast state #######
##############################

# Per-op autocast policies (the class attribute Op.autocast_policy):
AUTOCAST_LOWER = "lower"      # run in the autocast dtype (float16): MatMul, Conv
AUTOCAST_FP32 = "float32"     # run in float32: reductions, exp/log, powers, softmax
AUTOCAST_PROMOTE = "promote"  # default: mixed inputs are cast to the widest dtype
AUTOCAST_SKIP = "skip"        # never touched (Cast itself)


class _AutocastState:
    """Global autocast configuration; managed by needle.amp.autocast."""

    def __init__(self):
        self.enabled = False
        self.dtype = "float16"
        self.cache_enabled = True
        self.release_inputs = False
        # (id(leaf), dtype) -> (leaf, leaf.cached_data at cast time, cast tensor)
        self.cache = {}


_AUTOCAST = _AutocastState()


def _widest_dtype(dtypes) -> str:
    return max(dtypes, key=lambda d: (numpy.dtype(d).itemsize, d))


def _autocast_cast(x: "Tensor", dtype: str) -> "Tensor":
    """Cast for an autocast region. Parameters (leaves that require grad) are cast
    once per region and reused, so e.g. an RNN weight used at every time step is
    not converted again for each step. The cache entry is invalidated when the
    leaf's data is replaced (e.g. by an optimizer step inside the region)."""
    if _AUTOCAST.cache_enabled and x.is_leaf() and x.requires_grad:
        # grad mode is part of the key: a cast made under no_grad is a detached
        # constant, and reusing it with grad enabled would cut the parameter off
        key = (id(x), dtype, _GRAD_MODE.enabled)
        hit = _AUTOCAST.cache.get(key)
        if hit is not None and hit[0] is x and hit[1] is x.cached_data:
            return hit[2]
        out = needle.ops.cast(x, dtype, release_input=False)
        _AUTOCAST.cache[key] = (x, x.cached_data, out)
        return out
    return needle.ops.cast(x, dtype, release_input=_AUTOCAST.release_inputs)


def _autocast_inputs(op: "Op", inputs: List["Value"]) -> List["Value"]:
    """Cast an op's tensor inputs according to op.autocast_policy."""
    policy = getattr(op, "autocast_policy", AUTOCAST_PROMOTE)
    if policy == AUTOCAST_SKIP:
        return inputs
    dtypes = {x.dtype for x in inputs if isinstance(x, Tensor)}
    if not dtypes:
        return inputs
    if policy == AUTOCAST_LOWER:
        target = _AUTOCAST.dtype
    elif policy == AUTOCAST_FP32:
        target = "float32"
    else:  # promote
        if len(dtypes) == 1:
            return inputs
        target = _widest_dtype(dtypes)
    return [
        _autocast_cast(x, target) if isinstance(x, Tensor) and x.dtype != target else x
        for x in inputs
    ]


def autocast_promote(tensors: List["Tensor"]) -> List["Tensor"]:
    """Under autocast, cast a list of tensors to their widest dtype (used by ops
    that take a tuple, e.g. stack); outside autocast the list is returned as is."""
    if not _AUTOCAST.enabled or LAZY_MODE:
        return list(tensors)
    dtypes = {t.dtype for t in tensors}
    if len(dtypes) <= 1:
        return list(tensors)
    target = _widest_dtype(dtypes)
    return [_autocast_cast(t, target) if t.dtype != target else t for t in tensors]


##############################
######### Grad mode ##########
##############################


class _GradMode:
    """Whether ops record the graph (managed by no_grad / enable_grad / set_grad_enabled)."""

    def __init__(self):
        self.enabled = True


_GRAD_MODE = _GradMode()


def is_grad_enabled() -> bool:
    """True when ops record the computational graph (the default)."""
    return _GRAD_MODE.enabled


class set_grad_enabled(contextlib.ContextDecorator):
    """Context manager / decorator that turns graph recording on (True) or off (False).

    With recording off, every op result is a detached constant (requires_grad=False,
    no inputs), so nothing keeps its inputs alive: activations are freed as soon as
    the code no longer references them. Leaf tensors created explicitly keep the
    requires_grad they are given. Nestable and re-entrant; the previous mode is
    restored on exit, also when an exception is raised.
    """

    def __init__(self, mode: bool):
        self.mode = bool(mode)
        self._saved = []

    def __enter__(self):
        self._saved.append(_GRAD_MODE.enabled)
        _GRAD_MODE.enabled = self.mode
        return self

    def __exit__(self, *exc):
        _GRAD_MODE.enabled = self._saved.pop()
        return False


class no_grad(set_grad_enabled):
    """`with ndl.no_grad():` / `@ndl.no_grad()` -- inference and evaluation without
    building a graph."""

    def __init__(self):
        super().__init__(False)


class enable_grad(set_grad_enabled):
    """Turn graph recording back on inside a no_grad region."""

    def __init__(self):
        super().__init__(True)


class _AutocastDisabled:
    """Internal: suspend autocast (backward always runs in the forward dtypes)."""

    def __enter__(self):
        self.prev = _AUTOCAST.enabled
        _AUTOCAST.enabled = False

    def __exit__(self, *exc):
        _AUTOCAST.enabled = self.prev
        return False

class Op:
    """Operator definition."""

    # how autocast treats this op's inputs (see AUTOCAST_* above)
    autocast_policy = AUTOCAST_PROMOTE

    def __call__(self, *args):
        raise NotImplementedError()

    def compute(self, *args: Tuple[NDArray]):
        """Calculate forward pass of operator.

        Parameters
        ----------
        input: np.ndarray
            A list of input arrays to the function

        Returns
        -------
        output: nd.array
            Array output of the operation

        """
        raise NotImplementedError()

    def gradient(
        self, out_grad: "Value", node: "Value"
    ) -> Union["Value", Tuple["Value"]]:
        """Compute partial adjoint for each input value for a given output adjoint.

        Parameters
        ----------
        out_grad: Value
            The adjoint wrt to the output value.

        node: Value
            The value node of forward evaluation.

        Returns
        -------
        input_grads: Value or Tuple[Value]
            A list containing partial gradient adjoints to be propagated to
            each of the input node.
        """
        raise NotImplementedError()

    def gradient_as_tuple(self, out_grad: "Value", node: "Value") -> Tuple["Value"]:
        """Convenience method to always return a tuple from gradient call"""
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        else:
            return (output,)


class TensorOp(Op):
    """Op class specialized to output tensors, will be alternate subclasses for other structures"""

    def __call__(self, *args):
        return Tensor.make_from_op(self, args)


class TensorTupleOp(Op):
    """Op class specialized to output TensorTuple"""

    def __call__(self, *args):
        return TensorTuple.make_from_op(self, args)


class Value:
    """A value in the computational graph."""

    # trace of computational graph
    op: Optional[Op]
    inputs: List["Value"]
    # The following fields are cached fields for
    # dynamic computation
    cached_data: NDArray
    _shape: Optional[tuple]
    _dtype: Optional[str]
    _device: Optional[Device]
    requires_grad: bool

    def realize_cached_data(self):
        if self.cached_data is not None:
            return self.cached_data

        if self.op is None:
            raise RuntimeError(
                "Cannot recompute a leaf tensor whose data was released."
            )

        data = self.op.compute(
            *[x.realize_cached_data() for x in self.inputs]
        )

        self.cached_data = data

        if isinstance(self, Tensor):
            self._update_metadata(data)

        return self.cached_data

    def release_cached_data(self):
        if self.op is None:
            raise RuntimeError(
                "Cannot release leaf tensor data."
            )

        if self._shape is None:
            raise RuntimeError(
                "Cannot release an uninitialized tensor."
            )

        self.cached_data = None

    def is_leaf(self):
        return self.op is None

    def __del__(self):
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    def _update_metadata(self, data):
        self._shape = tuple(data.shape)
        self._dtype = str(data.dtype)

        if array_api is numpy:
            self._device = cpu()
        else:
            self._device = data.device

    def _init(
        self,
        op: Optional[Op],
        inputs: List["Tensor"],
        *,
        num_outputs: int = 1,
        cached_data: List[object] = None,
        requires_grad: Optional[bool] = None
    ):
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        if requires_grad is None:
            # under no_grad nothing created by an op requires grad (leaves created
            # explicitly keep the requires_grad they are given)
            requires_grad = _GRAD_MODE.enabled and any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad
        self._shape = None
        self._dtype = None
        self._device = None

        if cached_data is not None and isinstance(self, Tensor):
            self._update_metadata(cached_data)

    @classmethod
    def make_const(cls, data, *, requires_grad=False):
        value = cls.__new__(cls)
        value._init(
            None,
            [],
            cached_data=data,
            requires_grad=requires_grad,
        )
        return value

    @classmethod
    def make_from_op(cls, op: Op, inputs: List["Value"]):
        value = cls.__new__(cls)
        value._init(op, inputs)

        if not LAZY_MODE:
            if not value.requires_grad:
                return value.detach()
            value.realize_cached_data()
        return value


### Not needed in HW1
class TensorTuple(Value):
    """Represent a tuple of tensors.

    To keep things simple, we do not support nested tuples.
    """

    def __len__(self):
        cdata = self.realize_cached_data()
        return len(cdata)

    def __getitem__(self, index: int):
        return needle.ops.tuple_get_item(self, index)

    def tuple(self):
        return tuple([x for x in self])

    def __repr__(self):
        return "needle.TensorTuple" + str(self.tuple())

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        assert isinstance(other, TensorTuple)
        assert len(self) == len(other)
        return needle.ops.make_tuple(*[self[i] + other[i] for i in range(len(self))])

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return TensorTuple.make_const(self.realize_cached_data())


class Tensor(Value):
    grad: "Tensor"

    def __init__(
        self,
        array,
        *,
        device: Optional[Device] = None,
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        dtype = canonical_dtype(dtype)
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            elif device == array.device:
                # same device, new dtype: convert on the device
                cached_data = array.realize_cached_data().astype(dtype)
            else:
                # fall back, copy through numpy conversion
                cached_data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = device if device else default_device()
            cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)

        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        if array_api is numpy:
            return numpy.array(numpy_array, dtype=dtype)
        return array_api.array(numpy_array, device=device, dtype=dtype)

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        if not LAZY_MODE:
            if _AUTOCAST.enabled:
                inputs = _autocast_inputs(op, list(inputs))
            _check_input_dtypes(op, inputs)
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if not LAZY_MODE:
            if not tensor.requires_grad:
                return tensor.detach()
            tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)

        assert value.dtype == self.dtype
        assert value.shape == self.shape
        assert value.device == self.device

        self.cached_data = value.realize_cached_data()

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self):
        if self._shape is None:
            self.realize_cached_data()
        return self._shape

    @property
    def dtype(self):
        if self._dtype is None:
            self.realize_cached_data()
        return self._dtype

    @property
    def device(self):
        # numpy array always sits on cpu
        if array_api is numpy:
            return cpu()
        if self._device is None:
            self.realize_cached_data()
        return self._device

    def backward(self, out_grad=None, create_graph=False):
        """Compute the gradient of this tensor w.r.t. every tensor in its graph
        that requires grad, and store it in their .grad.

        create_graph=False (default): the gradient computation itself is not
        recorded, so every .grad is a plain tensor that does not reference (and
        keep alive) the forward activations. create_graph=True records it, which
        is only needed for higher-order derivatives (differentiating a .grad).
        """
        if not self.requires_grad:
            raise RuntimeError(
                "backward() on a tensor that does not require grad: it was created "
                "under no_grad, or from tensors none of which require grad"
            )
        if out_grad is None:
            out_grad = init.ones(*self.shape, dtype=self.dtype, device=self.device)
        elif out_grad.dtype != self.dtype:
            # the adjoint of a tensor always has the tensor's dtype
            out_grad = needle.ops.cast(out_grad, self.dtype, release_input=False)
        compute_gradient_of_variables(self, out_grad, create_graph=create_graph)

    def __repr__(self):
        return "needle.Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    def numpy(self):
        data = self.realize_cached_data()
        if array_api is numpy:
            return numpy.array(data)
        return data.numpy()

    def __add__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(self, other)
        else:
            return needle.ops.AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseMul()(self, other)
        else:
            return needle.ops.MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWisePow()(self, other)
        else:
            return needle.ops.PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(self, needle.ops.Negate()(other))
        else:
            return needle.ops.AddScalar(-other)(self)

    def __rsub__(self, other):
        return needle.ops.AddScalar(other)(needle.ops.Negate()(self))

    def __rtruediv__(self, other):
        return needle.ops.RDivScalar(other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseDiv()(self, other)
        else:
            return needle.ops.DivScalar(other)(self)

    def __matmul__(self, other):
        return needle.ops.MatMul()(self, other)

    def matmul(self, other):
        return needle.ops.MatMul()(self, other)

    def sum(self, axes=None):
        return needle.ops.Summation(axes)(self)

    def broadcast_to(self, shape):
        return needle.ops.BroadcastTo(shape)(self)

    def reshape(self, shape):
        return needle.ops.Reshape(shape)(self)

    def __neg__(self):
        return needle.ops.Negate()(self)

    def transpose(self, axes=None):
        return needle.ops.Transpose(axes)(self)




    __radd__ = __add__
    __rmul__ = __mul__

    ### dtype conversion (differentiable, see ops.Cast)
    def astype(self, dtype, release_input: bool = True):
        """Differentiable cast; returns self when the dtype already matches.

        By default the full-precision data of a non-leaf tensor is released after
        the cast (it is recomputed on demand if needed again); see ops.Cast.
        """
        return needle.ops.cast(self, dtype, release_input=release_input)

    def half(self, release_input: bool = True):
        return self.astype("float16", release_input=release_input)

    def float(self, release_input: bool = True):
        return self.astype("float32", release_input=release_input)

def compute_gradient_of_variables(output_tensor, out_grad, create_graph=False):
    """Take gradient of output node with respect to each node in node_list.

    Store the computed result in the grad field of each Variable.

    Autocast is suspended while the gradient graph is built: every gradient op
    runs in the dtypes the forward pass chose, whether or not backward() is
    called inside an autocast region.

    Gradient ops are recorded only with create_graph=True (independently of the
    surrounding grad mode); by default every gradient is a detached tensor.
    """
    with _AutocastDisabled(), set_grad_enabled(create_graph):
        _compute_gradient_of_variables(output_tensor, out_grad)


def _sum_adjoints(grads: List["Value"]) -> "Value":
    """Sum the partial adjoints of one node. float16 adjoints are accumulated in
    float32 and rounded once (a float16 running sum loses precision quickly, e.g.
    a weight cast once per autocast region and used at every RNN time step)."""
    if (
        len(grads) > 1
        and isinstance(grads[0], Tensor)
        and grads[0].dtype == "float16"
    ):
        total = sum_node_list(
            [needle.ops.cast(g, "float32", release_input=False) for g in grads]
        )
        return needle.ops.cast(total, "float16", release_input=False)
    return sum_node_list(grads)


def _compute_gradient_of_variables(output_tensor, out_grad):
    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
    # Special note on initializing gradient of
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_tensor] = [out_grad]

    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    for each in reverse_topo_order:
        if not each.requires_grad:
            continue
        each.grad = _sum_adjoints(node_to_output_grads_list[each])
        if each.op is None:
            continue
        prior_grads = each.op.gradient_as_tuple(each.grad, each)
        for i in range(len(prior_grads)):
            prior = each.inputs[i] 
            prior_grad = prior_grads[i] 
            # Invariant for mixed precision: a node's adjoint has the node's dtype
            # (so float32 master weights always get float32 grads and partial
            # adjoints of one node can be summed). Ops' gradients normally ensure
            # this already; cast any stray contribution instead of failing later.
            if (
                isinstance(prior, Tensor)
                and isinstance(prior_grad, Tensor)
                and prior_grad.dtype != prior.dtype
            ):
                prior_grad = needle.ops.cast(prior_grad, prior.dtype, release_input=False)
            if prior in node_to_output_grads_list:
                node_to_output_grads_list[prior].append(prior_grad)
            else:
                node_to_output_grads_list[prior] = [prior_grad]
    
    
    ### END YOUR SOLUTION


def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """
    ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    result = []
    visited = set()
    for each in node_list:
        topo_sort_dfs(each, visited, result)  
    return result
    ### END YOUR SOLUTION


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    ### BEGIN YOUR SOLUTION
    # raise NotImplementedError()
    if node in visited:
        return 
    visited.add(node) 
    for each in node.inputs:
        topo_sort_dfs(each, visited, topo_order) 
    topo_order.append(node) 
    ### END YOUR SOLUTION


##############################
####### Helper Methods #######
##############################


def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce

    return reduce(add, node_list)
