import numpy as np
import pytest

from prl.metrics import post_return_weights


def test_post_return_weights_basic_case():
    w_prev = np.array([0.5, 0.5], dtype=np.float64)
    r_arith = np.array([0.1, 0.0], dtype=np.float64)

    w_post = post_return_weights(w_prev, r_arith)

    expected = np.array([0.5238095238, 0.4761904762], dtype=np.float64)
    assert w_post == pytest.approx(expected)
    assert float(w_post.sum()) == pytest.approx(1.0)
    assert np.all(w_post >= 0.0)


def test_post_return_weights_wipeout_fallback():
    w_prev = np.array([0.2, 0.8], dtype=np.float64)
    r_arith = np.array([-1.0, -1.0], dtype=np.float64)

    w_post = post_return_weights(w_prev, r_arith)

    expected = w_prev / w_prev.sum()
    assert w_post == pytest.approx(expected)
    assert np.isfinite(w_post).all()
