"""Runs anywhere (no GPU needed): a numpy emulation of the tensor core matmul's
tiling and indexing (MatmulMmaBody in src/ndarray_backend_cuda.cu).

The emulation reads the tile constants from the CUDA source and mirrors the
kernel step by step: zero-padded staging of the A / B tiles into flat shared
buffers with padded leading dimensions, WMMA fragment loads at the kernel's
offsets, the 2 x 2 warp / 2 x 2 fragment layout, the store of the float32
accumulators, the guarded write-back and the grid-stride loop over the batch.
Shared buffers start as NaN, so reading a slot the kernel never wrote shows up.
The 16 x 16 fragment products themselves are numpy (that is the tensor cores' job).
"""
import re
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "ndarray_backend_cuda.cu"


def _constants():
    text = SRC.read_text()
    vals = {}
    for name in ("MMA_BM", "MMA_BN", "MMA_BK", "MMA_THREADS"):
        vals[name] = int(re.search(rf"#define {name} (\d+)", text).group(1))
    for name in ("MMA_A_LD", "MMA_B_LD", "MMA_C_LD"):
        expr = re.search(rf"#define {name} \(([^)]*)\)", text).group(1)
        vals[name] = eval(expr, {}, dict(vals))
    return vals


C = _constants()
BM, BN, BK, THREADS = C["MMA_BM"], C["MMA_BN"], C["MMA_BK"], C["MMA_THREADS"]
A_LD, B_LD, C_LD = C["MMA_A_LD"], C["MMA_B_LD"], C["MMA_C_LD"]
FRAG = 16


def load_frag(buf, offset, ld):
    """wmma::load_matrix_sync (row major)."""
    idx = offset + np.arange(FRAG)[:, None] * ld + np.arange(FRAG)[None, :]
    return buf[idx]


def store_frag(buf, offset, frag, ld):
    """wmma::store_matrix_sync (mem_row_major)."""
    idx = offset + np.arange(FRAG)[:, None] * ld + np.arange(FRAG)[None, :]
    buf[idx] = frag


def emulate(a, b, batch, M, N, P, a_bs, b_bs, grid_z):
    out = np.full(batch * M * P, np.nan)
    warps = THREADS // 32
    for bx in range((M + BM - 1) // BM):
        for by in range((P + BN - 1) // BN):
            for bz in range(grid_z):
                As = np.full(BM * A_LD, np.nan)
                Bs = np.full(BK * B_LD, np.nan)
                Cs = np.full(BM * C_LD, np.nan)
                row0, col0 = bx * BM, by * BN
                z = bz
                while z < batch:
                    acc = np.zeros((warps, 2, 2, FRAG, FRAG))
                    for k0 in range(0, N, BK):
                        for e in range(BM * BK):  # the threads together cover every e once
                            r, c = divmod(e, BK)
                            gr, gc = row0 + r, k0 + c
                            As[r * A_LD + c] = a[z * a_bs + gr * N + gc] if gr < M and gc < N else 0.0
                        for e in range(BK * BN):
                            r, c = divmod(e, BN)
                            gr, gc = k0 + r, col0 + c
                            Bs[r * B_LD + c] = b[z * b_bs + gr * P + gc] if gr < N and gc < P else 0.0
                        for w in range(warps):
                            wr, wc = w // 2, w % 2
                            af = [load_frag(As, (wr * 32 + i * 16) * A_LD, A_LD) for i in range(2)]
                            bf = [load_frag(Bs, wc * 32 + j * 16, B_LD) for j in range(2)]
                            for i in range(2):
                                for j in range(2):
                                    acc[w, i, j] += af[i] @ bf[j]
                    for w in range(warps):
                        wr, wc = w // 2, w % 2
                        for i in range(2):
                            for j in range(2):
                                store_frag(Cs, (wr * 32 + i * 16) * C_LD + wc * 32 + j * 16, acc[w, i, j], C_LD)
                    for e in range(BM * BN):
                        r, c = divmod(e, BN)
                        gr, gc = row0 + r, col0 + c
                        if gr < M and gc < P:
                            out[z * M * P + gr * P + gc] = Cs[r * C_LD + c]
                    z += grid_z
    return out.reshape(batch, M, P)


def test_constants_satisfy_wmma_requirements():
    """2 x 2 warps of 2 x 2 fragments cover the block tile; leading dimensions are
    multiples of 8 (16-bit) / 4 (float) elements; every fragment start is 32-byte
    aligned in shared memory."""
    assert THREADS // 32 == 4 and BM == 2 * 32 and BN == 2 * 32 and BK == FRAG
    assert A_LD % 8 == 0 and B_LD % 8 == 0 and C_LD % 4 == 0
    for i in range(2):
        for w in range(2):
            assert ((w * 32 + i * 16) * A_LD * 2) % 32 == 0  # A fragment rows, 2-byte elements
            assert ((w * 32 + i * 16) * 2) % 32 == 0  # B fragment columns
            assert ((w * 32 + i * 16) * C_LD * 4) % 32 == 0  # C fragment rows, float
            assert ((w * 32 + i * 16) * 4) % 32 == 0  # C fragment columns
    shared = BM * A_LD * 2 + BK * B_LD * 2 + BM * C_LD * 4
    assert shared <= 48 * 1024  # static shared memory limit


CASES = [  # (batch, M, N, P, a_shared, b_shared, grid_z)
    (1, 1, 1, 1, False, False, 1),
    (1, 5, 7, 3, False, False, 1),
    (1, 16, 16, 16, False, False, 1),
    (1, 64, 16, 64, False, False, 1),
    (1, 65, 17, 63, False, False, 1),
    (1, 130, 40, 70, False, False, 1),
    (3, 20, 33, 18, False, True, 3),  # shared right operand (stride 0)
    (4, 9, 5, 66, True, False, 4),  # shared left operand
    (5, 17, 31, 12, False, False, 2),  # batch > grid z: grid-stride loop
]


@pytest.mark.parametrize("batch,M,N,P,a_shared,b_shared,grid_z", CASES)
def test_emulated_kernel_matches_matmul(batch, M, N, P, a_shared, b_shared, grid_z):
    rng = np.random.default_rng(M * 1000 + N * 10 + P)
    A = rng.standard_normal((1 if a_shared else batch, M, N))
    B = rng.standard_normal((1 if b_shared else batch, N, P))
    out = emulate(A.ravel(), B.ravel(), batch, M, N, P,
                  0 if a_shared else M * N, 0 if b_shared else N * P, grid_z)
    assert not np.isnan(out).any()
    np.testing.assert_allclose(out, np.broadcast_to(A @ B, (batch, M, P)), rtol=1e-12, atol=1e-12)
