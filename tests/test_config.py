import sys
import pytest

sys.path.append("..")
from torch_runner import TrainerConfig


default_config = {
    "seed": 0,
    "batch_size": 1,
    "experiment_name": "model",
    "device": "cpu",
    "scheduler_step": "end",
    "scheduler_step_metric": "loss",
    "early_stop": False,
    "early_stop_params": {
        "patience": 5,
        "mode": "min",
        "delta": 0.0,
        "metric": "loss",
    },
    "use_wandb": False,
}

config1 = {
    "seed": 100,
    "batch_size": 32,
    "experiment_name": "test_model",
    "device": "cuda",
    "scheduler_step": "end",
    "scheduler_step_metric": "accuracy",
    "early_stop": False,
    "early_stop_params": {
        "patience": 2,
        "mode": "max",
        "delta": 0.01,
        "metric": "accuracy",
    },
    "use_wandb": True,
}

config2 = {
    "seed": 2,
    "batch_size": 12,
    "experiment_name": "test_model",
    "device": "cpu",
    "scheduler_step": "end",
    "scheduler_step_metric": "loss",
    "early_stop": True,
    "early_stop_params": {
        "patience": 2,
        "mode": "max",
        "delta": 0.01,
        "metric": "loss",
    },
    "use_wandb": True,
}

config3 = {
    "seed": 2,
    "batch_size": 1,
    "experiment_name": "test_model",
    "device": "cpu",
    "scheduler_step": "end",
    "scheduler_step_metric": "loss",
    "early_stop": True,
    "early_stop_params": {"patience": 2},
    "use_wandb": True,
}


def test_config_default():
    config = TrainerConfig()
    for k, v in config.__dict__.items():
        assert default_config[k] == v


@pytest.mark.parametrize("input_config", [pytest.param(config1), pytest.param(config2)])
def test_configs(input_config):
    config = TrainerConfig(**input_config)
    for k, v in config.__dict__.items():
        assert input_config[k] == v


def test_early_stop_missing_params():
    config = TrainerConfig(**config3)
    for k, v in config.__dict__.items():
        if k == "early_stop_params":
            assert v["patience"] == 2
            assert v["mode"] == "min"
            assert v["delta"] == 0.0
            assert v["metric"] == "loss"
            continue
        assert config3[k] == v
