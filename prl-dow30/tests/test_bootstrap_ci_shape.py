import numpy as np

from scripts.analyze_paper_results import bootstrap_ci


def test_bootstrap_ci_shape():
    values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    ci_low, ci_high = bootstrap_ci(values, n_boot=200, alpha=0.05, seed=123)
    assert ci_low <= ci_high
