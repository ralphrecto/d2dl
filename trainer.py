import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Union
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
import torch.nn as nn

class ModelingDataset:

    def __init__(self, train: Dataset, val: Dataset):
        self.train = train
        self.val = val

    def get_dataloaders(self, batch_size):
        return (
            DataLoader(self.train, batch_size=batch_size, shuffle=True),
            DataLoader(self.val, batch_size=batch_size, shuffle=True)
        )

class Hyperparameters:
    loss: Dict[str, Any] = field(default_factory=dict)
    opt: Dict[str, Any] = field(default_factory=dict)
    general: Dict[str, Any] = field(default_factory=dict)

class Trainer:

    def __init__(self, *,
        model: nn.Module,
        dataset: ModelingDataset,
        loss: Callable[..., nn.Module], 
        opt: Callable[..., Optimizer],
        hyperparameters: Hyperparameters
    ):
        self.model = model
        self.dataset = dataset
        self.loss_fn = loss(**hyperparameters.loss)
        self.opt = opt(self.model.parameters(), **hyperparameters.opt)
        self.hyperparams = hyperparameters

    def train(self, plot_cadence):
        train_dataloader, val_dataloader = self.dataset.get_dataloaders(self.hyperparams.general["batch_size"])

        train_loss_hist = []
        val_loss_hist = []
        for epoch_num in range(self.hyperparams.general["num_epochs"]):
            for batch_num, (train_data, val_data) in enumerate(zip(train_dataloader, val_dataloader)):
                train_X, train_y = train_data
                val_X, val_y = val_data

                pred_y = self.model(train_X)
                loss = self.loss_fn(pred_y, train_y)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                val_pred_y = self.model(val_X)
                val_loss = self.loss_fn(val_pred_y, val_y)
                
                if batch_num % plot_cadence == 0:
                    train_loss_hist.append(loss.item())
                    val_loss_hist.append(val_loss.item())

        return pd.DataFrame(dict(train=train_loss_hist, val=val_loss_hist))