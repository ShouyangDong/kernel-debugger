import ctypes
import numpy as np
import torch
import os

LIB_PATH = os.path.abspath("../build/libmha.so")

def load_lib(path=LIB_PATH):
    lib = ctypes.cdll.LoadLibrary(path)
    lib.mha_kernel.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.mha_kernel.restype = None
    return lib

def run_kernel(lib, Q, K, V, batch_size, num_heads, seq_len, head_dim):
    assert Q.dtype == np.float16
    out = np.empty_like(Q)
    Qc = np.ascontiguousarray(Q)
    Kc = np.ascontiguousarray(K)
    Vc = np.ascontiguousarray(V)
    outc = np.ascontiguousarray(out)

    lib.mha_kernel(
        Qc.ctypes.data_as(ctypes.c_void_p),
        Kc.ctypes.data_as(ctypes.c_void_p),
        Vc.ctypes.data_as(ctypes.c_void_p),
        outc.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(batch_size),
        ctypes.c_int(num_heads),
        ctypes.c_int(seq_len),
        ctypes.c_int(head_dim)
    )
    return outc

def torch_mha(Q, K, V):
    Qt = torch.from_numpy(Q).to(torch.float32)
    Kt = torch.from_numpy(K).to(torch.float32)
    Vt = torch.from_numpy(V).to(torch.float32)
    scale = 1.0 / float(np.sqrt(Q.shape[-1]))
    B, H, S, D = Q.shape
    Q2 = Qt.view(B * H, S, D)
    K2 = Kt.view(B * H, S, D)
    V2 = Vt.view(B * H, S, D)
    scores = torch.bmm(Q2, K2.transpose(1, 2)) * scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.bmm(attn, V2)
    out = out.view(B, H, S, D).to(torch.float16).numpy()
    return out

def max_abs_diff(a, b):
    return float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))

def mask_region(Q, K, V, b_idx, h_idx, seq_slice=None, dim_slice=None, fill=0.0):
    Qm = Q.copy()
    Km = K.copy()
    Vm = V.copy()
    if seq_slice is None:
        seq_slice = slice(0, Q.shape[2])
    if dim_slice is None:
        dim_slice = slice(0, Q.shape[3])
    Qm[b_idx, h_idx, seq_slice, dim_slice] = fill
    Km[b_idx, h_idx, seq_slice, dim_slice] = fill
    Vm[b_idx, h_idx, seq_slice, dim_slice] = fill
    return Qm, Km, Vm

def find_bad_token_range(lib, Q, K, V, ref_out, batch_idx=0, head_idx=0, tol=1e-2):
    B, H, S, D = Q.shape
    lo, hi = 0, S
    orig_out = run_kernel(lib, Q, K, V, B, H, S, D)
    if max_abs_diff(orig_out[batch_idx, head_idx], ref_out[batch_idx, head_idx]) <= tol:
        return None
    # binary search on token index where masking left half removes mismatch
    while hi - lo > 1:
        mid = (lo + hi) // 2
        Qm, Km, Vm = mask_region(Q, K, V, batch_idx, head_idx, seq_slice=slice(lo, mid))
        outm = run_kernel(lib, Qm, Km, Vm, B, H, S, D)
        diff_left = max_abs_diff(outm[batch_idx, head_idx], ref_out[batch_idx, head_idx])
        if diff_left <= tol:
            hi = mid
        else:
            lo = mid
    return lo

def find_bad_dim_range(lib, Q, K, V, ref_out, batch_idx=0, head_idx=0, token_idx=0, tol=1e-2):
    B, H, S, D = Q.shape
    lo, hi = 0, D
    orig_out = run_kernel(lib, Q, K, V, B, H, S, D)
    if max_abs_diff(orig_out[batch_idx, head_idx], ref_out[batch_idx, head_idx]) <= tol:
        return None
    # binary search on feature dimension
    while hi - lo > 1:
        mid = (lo + hi) // 2
        Qm, Km, Vm = mask_region(Q, K, V, batch_idx, head_idx,
                                 seq_slice=slice(token_idx, token_idx + 1),
                                 dim_slice=slice(lo, mid))
        outm = run_kernel(lib, Qm, Km, Vm, B, H, S, D)
        diff = max_abs_diff(outm[batch_idx, head_idx], ref_out[batch_idx, head_idx])
        if diff <= tol:
            hi = mid
        else:
            lo = mid
    return lo

def auto_debug(lib, Q, K, V, ref_out, batch_idx=0, head_idx=0, tol=1e-2):
    """
    返回 (bad_token, bad_dim, stage, index_issue)
    stage: one of None / 'scores_or_softmax' / 'final' / 'unknown'
    index_issue: True/False/None
    """
    bad_token = find_bad_token_range(lib, Q, K, V, ref_out, batch_idx, head_idx, tol)
    if bad_token is None:
        return None, None, None, None

    bad_dim = find_bad_dim_range(lib, Q, K, V, ref_out, batch_idx, head_idx, bad_token, tol)

    stage, index_issue = detect_error_stage(lib, Q, K, V, ref_out,
                                            batch_idx=batch_idx, head_idx=head_idx,
                                            token_idx=bad_token, dim_idx=bad_dim, tol=tol)
    return bad_token, bad_dim, stage, index_issue

def detect_error_stage(lib, Q, K, V, ref_out, batch_idx=0, head_idx=0,
                       token_idx=0, dim_idx=0, tol=1e-2):
    """
    用局部试验判断错误发生在 scores/softmax/final（attn@V）哪一阶段，并做简单的 index 错误检测。
    返回 (stage, index_issue_flag)
    """
    B, H, S, D = Q.shape
    orig_out = run_kernel(lib, Q, K, V, B, H, S, D)
    base_diff = max_abs_diff(orig_out[batch_idx, head_idx], ref_out[batch_idx, head_idx])

    # run kernel with one-cell masked (token_idx, dim_idx) for Q/K/V
    def run_mask(kind):
        Qm, Km, Vm = Q.copy(), K.copy(), V.copy()
        sl = slice(token_idx, token_idx + 1)
        ds = slice(dim_idx, dim_idx + 1)
        if kind == 'Q':
            Qm[batch_idx, head_idx, sl, ds] = 0.0
        elif kind == 'K':
            Km[batch_idx, head_idx, sl, ds] = 0.0
        elif kind == 'V':
            Vm[batch_idx, head_idx, sl, ds] = 0.0
        outm = run_kernel(lib, Qm, Km, Vm, B, H, S, D)
        return max_abs_diff(outm[batch_idx, head_idx], ref_out[batch_idx, head_idx])

    diff_mask_Q = run_mask('Q')
    diff_mask_K = run_mask('K')
    diff_mask_V = run_mask('V')

    # Decide stage
    if base_diff <= tol:
        stage = None
    elif diff_mask_V <= tol:
        stage = 'final'
    elif diff_mask_Q <= tol or diff_mask_K <= tol:
        stage = 'scores_or_softmax'
    else:
        stage = 'unknown'

    # index issue detection via token swap: compare how PyTorch and kernel change when swapping two tokens
    def swap_tokens_and_test(i, j):
        Ksw = K.copy()
        Ksw[:, :, [i, j], :] = Ksw[:, :, [j, i], :]
        ref_sw = torch_mha(Q, Ksw, V)
        out_sw = run_kernel(lib, Q, Ksw, V, B, H, S, D)
        ref_diff = max_abs_diff(ref_sw[batch_idx, head_idx], ref_out[batch_idx, head_idx])
        out_diff = max_abs_diff(out_sw[batch_idx, head_idx], orig_out[batch_idx, head_idx])
        return ref_diff, out_diff

    index_issue = None
    # choose neighbors around token_idx for swap test
    i = max(0, token_idx - 1)
    j = min(S - 1, token_idx + 1)
    if j != i:
        ref_diff, out_diff = swap_tokens_and_test(i, j)
        # 如果 PyTorch 有明显变化但 kernel 未能反映相应变化 -> 可能是 index bug
        if ref_diff > tol and out_diff < (ref_diff * 0.5):
            index_issue = True
        else:
            index_issue = False

    return stage, index_issue