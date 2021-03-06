import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import logging
import datetime
import os
import yaml
import copy
from .utils import seed_everything, EarlyStopping, AverageMeter

try:
    _wandb_available = True
    import wandb
except ImportError:
    _wandb_available = False


class TrainerModule:
    def __init__(
        self,
        model,
        optimizer,
        config,
        scheduler=None,
    ):
        self.config = config

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.scheduler_step = config.scheduler_step
        self.scheduler_step_metric = config.scheduler_step_metric

        self.early_stop = config.early_stop
        self.early_stop_params = config.early_stop_params
        self.early_stop_metric = config.early_stop_params["metric"]

        self.device = config.device
        self.experiment_name = config.experiment_name
        self.seed = config.seed

        self.epochs = None
        self.batch_size = config.batch_size
        self.hparams = None
        seed_everything(self.seed)

        if config.use_wandb and not _wandb_available:
            raise ModuleNotFoundError(
                "module wandb not found. use command 'pip install wandb'"
            )
        self.use_wandb = config.use_wandb

    def save_hparams(self, save_path):
        hparams = copy.deepcopy(self.__dict__)
        hparams["config"] = hparams["config"].__dict__
        hparams["model"] = hparams["model"].__class__.__name__
        hparams["optimizer"] = hparams["optimizer"].__class__.__name__
        hparams["optimizer_params"] = self.get_optim_params()
        hparams["scheduler"] = hparams["scheduler"].__class__.__name__
        hparams["device"] = hparams["device"]
        self.hparams = hparams

        with open(f"{save_path}/hparams.yml", "w") as outfile:
            yaml.safe_dump(hparams, outfile, default_flow_style=False)

    def get_optim_params(self):
        optim_dict = {}
        for group in self.optimizer.param_groups:
            for key in sorted(group.keys()):
                if key != "params":
                    optim_dict[key] = group[key]
        return optim_dict

    def calc_metric(self, **kwargs):
        raise NotImplementedError

    def loss_fct(self, **kwargs):
        raise NotImplementedError

    def train_one_step(self, batch, batch_id):
        raise NotImplementedError

    def valid_one_step(self, batch, batch_id):
        raise NotImplementedError

    def train_one_epoch(self, dataloader):
        self.model.train()
        meters = None
        pbar_params = None
        pbar = tqdm(dataloader, desc="Training")

        for batch_id, batch in enumerate(pbar):
            metrics = self.train_one_step(batch, batch_id)
            if meters is None:
                meters = dict(
                    zip(
                        metrics.keys(),
                        [AverageMeter() for _ in range(len(metrics.keys()))],
                    )
                )
                pbar_params = dict(zip(metrics.keys(), [0.0] * len(metrics.keys())))
            for key, value in metrics.items():
                meters[key].update(value)
                pbar_params[key] = meters[key].avg

            pbar.set_postfix(**pbar_params)

        return pbar_params

    @torch.no_grad()
    def validate_one_epoch(self, dataloader):
        self.model.eval()
        meters = None
        pbar_params = None
        pbar = tqdm(dataloader, desc="Validation")

        for batch_id, batch in enumerate(pbar):
            metrics = self.valid_one_step(batch, batch_id)
            if meters is None:
                meters = dict(
                    zip(
                        metrics.keys(),
                        [AverageMeter() for _ in range(len(metrics.keys()))],
                    )
                )
                pbar_params = dict(zip(metrics.keys(), [0.0] * len(metrics.keys())))
            for key, value in metrics.items():
                meters[key].update(value)
                pbar_params[key] = meters[key].avg

            pbar.set_postfix(**pbar_params)

        return pbar_params

    def fit(self, train_dataloader, val_dataloader, epochs, wanb_project_name=None):
        self.epochs = epochs

        time = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
        dir_name = f"{self.experiment_name}_{time}"
        if os.path.exists(dir_name):
            raise FileExistsError
        else:
            os.mkdir(dir_name)
        logging.basicConfig(
            filename=f"{dir_name}/log_file.log",
            level=logging.INFO,
            format="%(message)s",
        )

        self.save_hparams(dir_name)

        if self.use_wandb:
            assert wanb_project_name is not None, "Provide project name to use wandb"
            wandb.init(project=wanb_project_name, name=dir_name, config=self.hparams)
            wandb.watch(self.model)

        es = EarlyStopping(**self.early_stop_params)
        for epoch in range(epochs):
            print(f"Epoch: {epoch+1}/{epochs}")
            train_metrics = self.train_one_epoch(train_dataloader)
            val_metrics = self.validate_one_epoch(val_dataloader)

            logging_line = f"Epoch: {epoch+1}"
            for key, value in train_metrics.items():
                logging_line += f", train_{key}: {value}"
                if self.use_wandb:
                    wandb.log({f"train_{key}": value}, step=epoch)
            for key, value in val_metrics.items():
                logging_line += f", val_{key}: {value}"
                if self.use_wandb:
                    wandb.log({f"val_{key}": value}, step=epoch)
            logging.info(logging_line)

            if self.scheduler and self.scheduler_step == "end":
                if self.scheduler.__class__.__name__ == "ReduceLROnPlateau":
                    self.scheduler.step(val_metrics[self.scheduler_step_metric])
                else:
                    self.scheduler.step()
            score_not_improved, best_score = es(
                f"{dir_name}/model.pth",
                val_metrics[self.early_stop_metric],
                self.model,
                self.optimizer,
                self.scheduler,
            )
            if score_not_improved:
                if self.early_stop:
                    print(
                        "EarlyStopping counter: {} out of {}, Best Score: {}".format(
                            es.counter, es.patience, best_score
                        )
                    )
                else:
                    print("Score not improved, Best Score: {}".format(best_score))
            if es.early_stop and self.early_stop:
                print("Early Stopping")
                break