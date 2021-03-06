import torch
import random
import os
import numpy as np


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    def __init__(self, patience=5, mode="min", delta=0.0, *args, **kwargs):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf

    def __call__(self, model_path, epoch_score, model, optimizer, scheduler=None):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        score_not_improved = False
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, optimizer, scheduler, model_path)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            score_not_improved = True
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, optimizer, scheduler, model_path)
            self.counter = 0

        if self.mode == "min":
            return score_not_improved, -1.0 * self.best_score

        return score_not_improved, self.best_score

    def save_checkpoint(self, epoch_score, model, optimizer, scheduler, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            print(
                "Validation score improved ({} --> {}). Saving model!".format(
                    self.val_score, epoch_score
                )
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else scheduler,
                    "score": epoch_score,
                },
                model_path,
            )
        self.val_score = epoch_score
