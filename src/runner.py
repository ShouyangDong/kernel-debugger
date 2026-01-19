import os
import ctypes
import numpy as np
import torch
from analyzer import analyze_outputs
from modifier import modify_cuda_code
from find_bug import load_lib, run_kernel, torch_mha

def execute_debugging_process():
    lib = load_lib()
    B, H, S, D = 1, 4, 16, 64  # Adjust to your configuration
    Q = (np.random.randn(B, H, S, D) * 0.02).astype(np.float16)
    K = (np.random.randn(B, H, S, D) * 0.02).astype(np.float16)
    V = (np.random.randn(B, H, S, D) * 0.02).astype(np.float16)
    
    ref_output = torch_mha(Q, K, V)
    orig_output = run_kernel(lib, Q, K, V, B, H, S, D)

    discrepancies = analyze_outputs(orig_output, ref_output)
    
    if discrepancies:
        print("Discrepancies found:", discrepancies)
        for discrepancy in discrepancies:
            bad_token = discrepancy['bad_token']
            bad_dim = discrepancy['bad_dim']
            print(f"Suspected bad token index: {bad_token}")
            print(f"Suspected bad dimension index for token: {bad_dim}")
            modify_cuda_code(bad_token, bad_dim)

if __name__ == "__main__":
    execute_debugging_process()