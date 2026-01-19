import os
import numpy as np
import pytest

from find_bug import load_lib, torch_mha, auto_debug, run_kernel

def test_auto_debug_smoke():
    """
    Smoke test for auto_debug.
    Set environment variable LIBMHA_SO to override the shared lib path if needed.
    """
    lib_path = os.environ.get("LIBMHA_SO")
    lib = load_lib(lib_path) if lib_path else load_lib()

    # deterministic small case
    rng = np.random.default_rng(1234)
    B, H, S, D = 1, 2, 8, 16
    Q = (rng.normal(scale=0.02, size=(B, H, S, D))).astype(np.float16)
    K = (rng.normal(scale=0.02, size=(B, H, S, D))).astype(np.float16)
    V = (rng.normal(scale=0.02, size=(B, H, S, D))).astype(np.float16)

    # reference with PyTorch
    ref = torch_mha(Q, K, V)

    # run auto-debug
    bad_token, bad_dim, stage, index_issue = auto_debug(lib, Q, K, V, ref,
                                                        batch_idx=0, head_idx=0, tol=1e-2)

    # basic structural assertions — the test is tolerant: kernel may be correct (Nones) or report an issue
    assert isinstance((bad_token, bad_dim, stage, index_issue), tuple)
    assert stage in (None, 'scores_or_softmax', 'final', 'unknown')
    assert index_issue in (None, True, False)

    if bad_token is not None:
        assert 0 <= bad_token < S
    if bad_dim is not None:
        assert 0 <= bad_dim < D