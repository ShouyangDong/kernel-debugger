import numpy as np
import torch

class Analyzer:
    def __init__(self, tol=1e-2):
        self.tol = tol

    def max_abs_diff(self, a, b):
        return float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))

    def analyze_outputs(self, orig_out, ref_out, batch_idx=0, head_idx=0):
        if self.max_abs_diff(orig_out[batch_idx, head_idx], ref_out[batch_idx, head_idx]) > self.tol:
            return True  # Discrepancy found
        return False  # No discrepancy

    def suggest_fixes(self, orig_out, ref_out, batch_idx=0, head_idx=0):
        discrepancies = []
        if self.analyze_outputs(orig_out, ref_out, batch_idx, head_idx):
            discrepancies.append("Discrepancy found in batch {}, head {}".format(batch_idx, head_idx))
            # Additional logic to suggest specific fixes can be added here
        return discrepancies

    def run_analysis(self, orig_out, ref_out):
        suggestions = []
        B, H, _, _ = orig_out.shape
        for b in range(B):
            for h in range(H):
                suggestions.extend(self.suggest_fixes(orig_out, ref_out, b, h))
        return suggestions

def example_analysis():
    # Example usage of the Analyzer class
    B, H, S, D = 1, 4, 16, 64  # Example dimensions
    orig_out = np.random.randn(B, H, S, D).astype(np.float16)  # Simulated original output
    ref_out = np.random.randn(B, H, S, D).astype(np.float16)  # Simulated reference output

    analyzer = Analyzer()
    suggestions = analyzer.run_analysis(orig_out, ref_out)
    for suggestion in suggestions:
        print(suggestion)

if __name__ == "__main__":
    example_analysis()