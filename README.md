# Torch Runner
### A minimal wrapper that removes some of the overhead code in training pytorch models
Note: If you are looking for something more extensive, checkout [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning). This is mostly designed for my personal use. 

## Requirements
- torch
- tqdm

## Features
- seed all variables
- text logger
- early stopping
- save hyperparameters

## Example

```python
import torch
import torch_runner as T


class myTrainer(T.TrainerModule):
    def __init__(self, model, optimizer):
        super(myTrainer, self).__init__(model, optimizer)
    
    def calc_metric(self, preds, target):
        ## Calc metrics such as accuracy etc.
    
    def loss_fct(self, preds, target):
        ## Calc loss
    
    def train_one_step(self, batch, batch_id):
        ## Get batch data from dataloader and perform one update
    
    def valid_one_step(self, batch, batch_id):
        ## Perform validation step

model = myModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

train_dataloader = ## pytorch dataloader
val_dataloader = ## pytorch dataloader

Trainer = myTrainer(model, optimizer)
Trainer.fit(train_dataloader, val_dataloader, epochs=10, batch_size=32)
```
