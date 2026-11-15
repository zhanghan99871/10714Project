"""memory_allocated / max_memory_allocated / reset_peak_memory_stats count the bytes
held by NDArray buffers, per device."""
import gc
import sys

sys.path.append("./python")
import numpy as np
import pytest

import needle as ndl
from needle import backend_ndarray as nd

_DEVICES = [
    ndl.cpu_numpy(),
    ndl.cpu(),
    pytest.param(ndl.cuda(), marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU")),
]
_IDS = ["numpy", "cpu", "cuda"]


@pytest.fixture(autouse=True)
def no_cycle_collection():
    """A cycle collection during a test would free buffers of earlier tests."""
    gc.collect()
    gc.disable()
    yield
    gc.enable()


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_allocation_and_free(device):
    base = nd.memory_allocated(device)
    a = nd.NDArray.make((16, 8), device=device)
    assert nd.memory_allocated(device) == base + 16 * 8 * 4
    h = nd.NDArray.make((10,), device=device, dtype="float16")
    assert nd.memory_allocated(device) == base + 16 * 8 * 4 + 10 * 2
    del a
    assert nd.memory_allocated(device) == base + 10 * 2
    del h
    assert nd.memory_allocated(device) == base


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_bfloat16_counts_two_bytes(device):
    if not device.supports_dtype("bfloat16"):
        pytest.skip(f"{device} has no bfloat16")
    base = nd.memory_allocated(device)
    a = nd.NDArray.make((7,), device=device, dtype="bfloat16")
    assert nd.memory_allocated(device) == base + 7 * 2
    del a


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_views_share_their_buffer(device):
    a = nd.NDArray.make((4, 6), device=device)
    base = nd.memory_allocated(device)
    v = a.permute((1, 0))
    r = a.reshape((24,))
    assert nd.memory_allocated(device) == base
    del a, r
    assert nd.memory_allocated(device) == base  # v still holds the buffer
    del v
    assert nd.memory_allocated(device) == base - 24 * 4


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_peak_and_reset(device):
    base = nd.memory_allocated(device)
    nd.reset_peak_memory_stats(device)
    assert nd.max_memory_allocated(device) == base
    a = nd.NDArray.make((256,), device=device)
    b = nd.NDArray.make((128,), device=device)
    del a, b
    assert nd.memory_allocated(device) == base
    assert nd.max_memory_allocated(device) == base + 384 * 4
    nd.reset_peak_memory_stats(device)
    assert nd.max_memory_allocated(device) == base


@pytest.mark.parametrize("device", _DEVICES, ids=_IDS)
def test_tensor_ops_are_counted(device):
    x = ndl.Tensor(np.ones((32, 32), dtype=np.float32), device=device)
    base = nd.memory_allocated(device)
    y = x * 2.0
    assert nd.memory_allocated(device) == base + 32 * 32 * 4
    y.release_cached_data()
    assert nd.memory_allocated(device) == base
    del y


def test_devices_are_counted_separately():
    base_numpy = nd.memory_allocated(ndl.cpu_numpy())
    a = nd.NDArray.make((100,), device=ndl.cpu())
    assert nd.memory_allocated(ndl.cpu_numpy()) == base_numpy
    del a
