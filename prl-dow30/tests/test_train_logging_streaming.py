import pandas as pd
from types import SimpleNamespace

from prl.train import TrainLoggingCallback


def test_train_logging_streams_to_csv(tmp_path):
    log_path = tmp_path / "log.csv"
    cb = TrainLoggingCallback(log_path, run_id="run123", model_type="baseline", seed=0, log_interval=2)

    class DummyLogger:
        def __init__(self):
            self.name_to_value = {
                "train/actor_loss": 1.0,
                "train/critic_loss": 2.0,
                "train/entropy_loss": 3.0,
            }

    cb.model = SimpleNamespace(logger=DummyLogger())
    # simulate steps
    for t in range(5):
        cb.num_timesteps = t + 1
        cb._on_step()

    cb._on_training_end()

    assert log_path.exists()
    df = pd.read_csv(log_path)
    # Expect multiple flushes: rows should be present and not all buffered
    assert len(df) == 5
    for col in [
        "schema_version",
        "run_id",
        "model_type",
        "seed",
        "timesteps",
        "actor_loss",
        "critic_loss",
        "entropy_loss",
        "ent_coef",
        "ent_coef_loss",
        "alpha_obs_mean",
        "alpha_next_mean",
        "prl_prob_mean",
        "vz_mean",
        "alpha_raw_mean",
        "alpha_clamped_mean",
        "emergency_rate",
        "beta_effective_mean",
    ]:
        assert col in df.columns
    # Ensure buffer cleared
    assert cb.buffer == []
