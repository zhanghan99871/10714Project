import numpy as np
import sys
sys.path.append('./python')
sys.path.append('./apps')
import needle as ndl

def test_metadata_survives_release():
    x = ndl.Tensor(
        np.ones((4, 8), dtype=np.float32)
    )

    W = ndl.Tensor(
        np.ones((8, 16), dtype=np.float32)
    )

    y = x @ W

    original_shape = y.shape
    original_dtype = y.dtype
    original_device = y.device
    original_value = y.numpy().copy()

    # Release only the intermediate
    y.release_cached_data()

    assert y.cached_data is None

    # Metadata access must not trigger recomputation
    assert y.shape == original_shape
    assert y.dtype == original_dtype
    assert y.device == original_device

    assert y.cached_data is None

    # Numerical access should recompute
    recomputed = y.numpy()

    np.testing.assert_allclose(
        recomputed,
        original_value,
        rtol=1e-5,
        atol=1e-5,
    )

    assert y.cached_data is not None