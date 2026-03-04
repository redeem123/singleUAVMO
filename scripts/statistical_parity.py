from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.stats import ranksums


def compute_vda(data1: np.ndarray, data2: np.ndarray) -> float:
    """Compute Vargha-Delaney A effect size."""
    m, n = len(data1), len(data2)
    r = ranksums(data1, data2).statistic
    # Approx A from z-score: A = (z / sqrt(m+n+1) * sqrt(m*n/12) + m*n/2) / (m*n)
    # But better to use direct rank sum:
    from scipy.stats import mannwhitneyu
    u, _ = mannwhitneyu(data1, data2, alternative="two-sided")
    return u / (m * n)


def compare_hv(python_results: Path, matlab_results: Path):
    """Compare Hypervolume distributions between Python and MATLAB."""
    print(f"Comparing: {python_results} vs {matlab_results}")
    
    # Load bestScores (runs x 2: HV, IGD)
    py_data = loadmat(str(python_results / "final_hv.mat"))["bestScores"][:, 0]
    mat_data = loadmat(str(matlab_results / "final_hv.mat"))["bestScores"][:, 0]
    
    stat, p = ranksums(py_data, mat_data)
    vda = compute_vda(py_data, mat_data)
    
    print(f"Python HV Mean: {np.mean(py_data):.4f} (±{np.std(py_data):.4f})")
    print(f"MATLAB HV Mean: {np.mean(mat_data):.4f} (±{np.std(mat_data):.4f})")
    print(f"Wilcoxon p-value: {p:.4f}")
    print(f"Vargha-Delaney A: {vda:.4f}")
    
    if p < 0.05:
        print("RESULT: Statistically significant difference detected.")
    else:
        print("RESULT: No statistically significant difference (Parity maintained).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--py", type=Path, required=True, help="Python problem result dir")
    parser.add_argument("--mat", type=Path, required=True, help="MATLAB problem result dir")
    args = parser.parse_args()
    
    compare_hv(args.py, args.mat)


if __name__ == "__main__":
    main()
