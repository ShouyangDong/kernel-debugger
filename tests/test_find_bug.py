import numpy as np
import torch
import unittest
from src.find_bug import load_lib, run_kernel, torch_mha, find_bad_token_range, find_bad_dim_range

class TestFindBug(unittest.TestCase):
    def setUp(self):
        self.lib = load_lib()
        self.B, self.H, self.S, self.D = 1, 4, 16, 64
        self.Q = (np.random.randn(self.B, self.H, self.S, self.D) * 0.02).astype(np.float16)
        self.K = (np.random.randn(self.B, self.H, self.S, self.D) * 0.02).astype(np.float16)
        self.V = (np.random.randn(self.B, self.H, self.S, self.D) * 0.02).astype(np.float16)
        self.ref = torch_mha(self.Q, self.K, self.V)

    def test_find_bad_token_range(self):
        bad_token = find_bad_token_range(self.lib, self.Q, self.K, self.V, self.ref)
        self.assertIsInstance(bad_token, (int, type(None)), "Expected bad token index to be an integer or None")

    def test_find_bad_dim_range(self):
        bad_token = find_bad_token_range(self.lib, self.Q, self.K, self.V, self.ref)
        if bad_token is not None:
            bad_dim = find_bad_dim_range(self.lib, self.Q, self.K, self.V, self.ref, batch_idx=0, head_idx=0, token_idx=bad_token)
            self.assertIsInstance(bad_dim, (int, type(None)), "Expected bad dimension index to be an integer or None")

if __name__ == "__main__":
    unittest.main()