# Torch Runner
### A minimal wrapper that removes some of the overhead code in training pytorch models
Note: If you are looking for something more extensive, checkout [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning). This is mostly designed for my personal use. 

## Requirements
- torch
- tqdm

## Installation

```
pip install torch-runner
```

## Features
- seed all variables
- text logger
- early stopping
- save hyperparameters
- [weights & biases](https://wandb.ai/) support

## Example
Checkout the [examples](https://github.com/grohith327/Torch-Runner/blob/main/examples/Torch_Runner_CIFAR10.ipynb) folder which contains a jupyter notebook to train a resnet50 using torch_runner. 
 
```python
import torch
import torch_runner as T


class myTrainer(T.TrainerModule):
    def calc_metric(self, preds, target):
        ## Calc metrics such as accuracy etc.
    
    def loss_fct(self, preds, target):
        ## Calc loss
    
    def train_one_step(self, batch, batch_id):
        ## Get batch data from dataloader and perform one update
    
    def valid_one_step(self, batch, batch_id):
        ## Perform validation step

config = T.TrainerConfig()
model = myModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

train_dataloader = ## pytorch dataloader
val_dataloader = ## pytorch dataloader

Trainer = myTrainer(model, optimizer, config)
Trainer.fit(train_dataloader, val_dataloader, epochs=10)
```
