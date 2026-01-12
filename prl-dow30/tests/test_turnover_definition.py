import numpy as np

from prl.metrics import turnover_l1


def test_turnover_definition_full_l1():
    w_prev = np.array([0.6, 0.3, 0.1], dtype=np.float32)
    w_new = np.array([0.2, 0.5, 0.3], dtype=np.float32)
    expected = float(np.abs(w_new - w_prev).sum())
    assert turnover_l1(w_prev, w_new) == expected
