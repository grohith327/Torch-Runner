import sys
import pytest
import torch
import torch.nn as nn

sys.path.append("..")
from torch_runner import TrainerConfig, TrainerModule


class TestTrainer(TrainerModule):
    __test__ = False

    def train_one_step(self, batch, batch_id):
        _ = self.model(batch["data"])
        return {"loss": 0}

    def valid_one_step(self, batch, batch_id):
        _ = self.model(batch["data"])
        return {"loss": 0}


class TestDataset(torch.utils.data.Dataset):
    __test__ = False

    def __getitem__(self, i):
        return {"data": torch.randn(8).float()}

    def __len__(self):
        return 10


@pytest.fixture
def model():
    return nn.Sequential(nn.Linear(8, 8, bias=False))


@pytest.fixture
def optimizer(model):
    return torch.optim.Adam(model.parameters())


@pytest.fixture
def train_dataloader():
    dataset = TestDataset()
    return torch.utils.data.DataLoader(dataset, batch_size=1)


@pytest.fixture
def val_dataloader():
    dataset = TestDataset()
    return torch.utils.data.DataLoader(dataset, batch_size=1)


@pytest.fixture
def trainer(tmp_path, model, optimizer):
    exp_name = tmp_path / "test_model"
    config = TrainerConfig(experiment_name=str(exp_name))
    return TestTrainer(model, optimizer, config)


@pytest.fixture
def trainer_wandb(tmp_path, model, optimizer):
    exp_name = tmp_path / "test_model_wandb"
    config = TrainerConfig(experiment_name=str(exp_name), use_wandb=True)
    return TestTrainer(model, optimizer, config)


def test_save_hparams(tmp_path, model, optimizer):
    config = TrainerConfig()
    trainer = TrainerModule(model, optimizer, config)
    trainer.save_hparams(tmp_path)


def test_fit(trainer_wandb, train_dataloader, val_dataloader):
    trainer_wandb.fit(
        train_dataloader, val_dataloader, 10, wanb_project_name="torch-runner-tests"
    )
