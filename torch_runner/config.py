from dataclasses import dataclass, field
import torch


@dataclass
class TrainerConfig:
    seed: int = 0
    batch_size: int = 1
    experiment_name: str = "model"
    device: str = "cpu"
    scheduler_step: str = "end"
    scheduler_step_metric: str = "loss"
    early_stop: bool = False
    early_stop_params: dict = field(
        default_factory=lambda: {
            "patience": 5,
            "mode": "min",
            "delta": 0.0,
            "metric": "loss",
        }
    )
    use_wandb: bool = False

    def __set_early_stop_missing_params(self):
        self.early_stop_params["patience"] = self.early_stop_params.get("patience", 5)
        self.early_stop_params["mode"] = self.early_stop_params.get("mode", "min")
        self.early_stop_params["delta"] = self.early_stop_params.get("delta", 0.0)
        self.early_stop_params["metric"] = self.early_stop_params.get("metric", "loss")

    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.early_stop:
            self.__set_early_stop_missing_params()
