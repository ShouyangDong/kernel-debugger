import numpy as np
import torch
from src.find_bug import load_lib, run_kernel, torch_mha, find_bad_token_range, find_bad_dim_range

def example_run():
    lib = load_lib()
    B, H, S, D = 1, 4, 16, 64  # Adjust to your configuration
    # Random test input
    Q = (np.random.randn(B, H, S, D) * 0.02).astype(np.float16)
    K = (np.random.randn(B, H, S, D) * 0.02).astype(np.float16)
    V = (np.random.randn(B, H, S, D) * 0.02).astype(np.float16)
    
    # Compute reference output using PyTorch
    ref = torch_mha(Q, K, V)
    
    # Find bad token
    bad_token = find_bad_token_range(lib, Q, K, V, ref, batch_idx=0, head_idx=0)
    print("Suspected bad token index:", bad_token)
    
    if bad_token is not None:
        # Find bad dimension for the suspected bad token
        bad_dim = find_bad_dim_range(lib, Q, K, V, ref, batch_idx=0, head_idx=0, token_idx=bad_token)
        print("Suspected bad dimension index for token:", bad_dim)

if __name__ == "__main__":
    example_run()