import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, Any, Callable 
from itertools import product 
from torch.optim import Optimizer
import torch.nn as nn

from data import ModelingDataset

@dataclass
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

def grid_search(hyperparam_grid, trainer_provider):
    results = []
    for i, hyperparams in enumerate(hyperparam_grid):
        print(f"Progress: {i}/{len(hyperparam_grid)}")
        trainer = trainer_provider(hyperparams)

        train_res = trainer.train(10)
        results.append(train_res)

    hyperparam_i, min_val_idx, min_val_loss = min((
        (i, result["val"].idxmin(), result["val"].min())
        for i, result in enumerate(results)
    ), key=lambda t: t[2])

    return hyperparam_i, min_val_loss

def grid_search_params(hyperparam_grids: Hyperparameters): 
    keys = []
    grid_vals = []
    for namespace, hyperparams in hyperparam_grids.__dict__.items():
        for k, k_grid_vals in hyperparams.items():
            keys.append((namespace, k))
            grid_vals.append(k_grid_vals)

    all_val_combos = product(*grid_vals)

    all_hyperparams = []
    for all_val_combo in all_val_combos:
        hyperparams = Hyperparameters()
        for ((namespace, k), val) in zip(keys, all_val_combo):
            getattr(hyperparams, namespace)[k] = val

        all_hyperparams.append(hyperparams)

    return all_hyperparams