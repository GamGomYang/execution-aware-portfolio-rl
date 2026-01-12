import numpy as np

from prl.envs import stable_softmax


def test_action_expressivity_changes_with_logit_scale():
    z = np.array([1.0] + [-1.0] * 29, dtype=np.float32)
    w_low = stable_softmax(z, scale=1.0)
    w_high = stable_softmax(z, scale=10.0)

    assert np.isclose(w_low.sum(), 1.0)
    assert np.isclose(w_high.sum(), 1.0)
    assert np.all(w_low >= 0) and np.all(w_high >= 0)

    assert w_low.max() < 0.4  # near-uniform at low scale
    assert w_high.max() > 0.7  # concentrated at high scale
    assert w_high.max() > w_low.max()
