import sys
import pytest
import torch
import torch.nn as nn

sys.path.append("..")
from torch_runner import EarlyStopping, AverageMeter

TEST_SCORE = 100


@pytest.fixture
def model():
    return nn.Sequential(nn.Linear(8, 8, bias=False))


@pytest.fixture
def optimizer(model):
    return torch.optim.Adam(model.parameters())


def test_average_meter():
    meter = AverageMeter()
    assert meter.val == 0
    assert meter.avg == 0
    assert meter.sum == 0
    assert meter.count == 0

    meter.update(val=10, n=1)
    assert meter.val == 10
    assert meter.avg == 10
    assert meter.sum == 10
    assert meter.count == 1

    meter.update(val=10, n=2)
    assert meter.val == 10
    assert meter.avg == 10
    assert meter.sum == 30
    assert meter.count == 3


def test_early_stopping(tmp_path, model, optimizer):
    es = EarlyStopping()
    assert es.patience == 5
    assert es.counter == 0
    assert es.mode == "min"
    assert es.best_score == None
    assert es.early_stop == False
    assert es.delta == 0.0

    ## test for different args
    es = EarlyStopping(metric="loss")


def test_min_early_stopping(tmp_path, model, optimizer):
    save_path = tmp_path / "test_model.pth"
    es = EarlyStopping(patience=1, mode="min")

    score_not_improved, best_score = es(save_path, TEST_SCORE, model, optimizer)
    assert es.early_stop == False
    assert best_score == TEST_SCORE
    assert score_not_improved == False

    score_not_improved, best_score = es(save_path, TEST_SCORE + 100, model, optimizer)
    assert es.early_stop == True
    assert best_score == TEST_SCORE
    assert score_not_improved == True


def test_max_early_stopping(tmp_path, model, optimizer):
    save_path = tmp_path / "test_model.pth"
    es = EarlyStopping(patience=1, mode="max")

    score_not_improved, best_score = es(save_path, TEST_SCORE, model, optimizer)
    assert es.early_stop == False
    assert best_score == TEST_SCORE
    assert score_not_improved == False

    score_not_improved, best_score = es(save_path, TEST_SCORE - 100, model, optimizer)
    assert es.early_stop == True
    assert best_score == TEST_SCORE
    assert score_not_improved == True