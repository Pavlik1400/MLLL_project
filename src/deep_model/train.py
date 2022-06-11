import os
import sys
sys.path.append(os.getcwd())
from src.deep_model.dataset import MNIST_Dataset
from src.deep_model.linear_nn import LinearMnistNN
from src.helpers import birthdays_to_digits
from src.data_preparation import load as load_mnist
from sklearn.metrics import f1_score, accuracy_score
from argparse import ArgumentParser
import yaml
from typing import Callable, Dict, List
from torch.utils.data import DataLoader
import torch
from torch.optim import Adam
from pprint import pprint
import numpy as np
from vizualize import vizualize_history


def train_one_epoch(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    training_loader: DataLoader,
                    loss_fn: Callable,
                    print_each: int,
                    ) -> float:
    running_loss = 0.
    last_loss = 0.
    loss_history = []
    for i, data in enumerate(training_loader):
        inputs, labels = data

        # Zero your gradients for every batch
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % print_each == print_each-1:
            last_loss = running_loss / print_each  # loss per batch
            print(f"  batch {i+1} loss: {last_loss}")
            loss_history.append(float(last_loss))
            running_loss = 0.
    return loss_history


def train_model(model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                config: Dict,
                ) -> Dict[str, List]:
    deep_cnf = config["deep_model"]
    loss_fn = torch.nn.CrossEntropyLoss()

    # load dataset
    dig1, dig2 = birthdays_to_digits(
        config["PB"], config["VB"])
    data = load_mnist(dig1, dig2)

    train_ds = MNIST_Dataset(data, train=True)
    cv_ds = MNIST_Dataset(data, cv=True)

    # prepare data loader
    training_loader = DataLoader(
        train_ds, batch_size=deep_cnf["batch_size"], shuffle=True, num_workers=deep_cnf["loader_num_workers"]
    )

    validation_loader = DataLoader(
        cv_ds, batch_size=deep_cnf["batch_size"], shuffle=False, num_workers=deep_cnf["loader_num_workers"]
    )

    print_each = deep_cnf["print_each"]
    n_epoch = deep_cnf["n_epoch"]

    history = {
        "avg_cv_loss": [],
        "train_loss": [],
        "avg_accuracy": [],
        "avg_f1": [],
    }

    for epoch in range(n_epoch):
        model.train()
        loss_h = train_one_epoch(
            model, optimizer, training_loader, loss_fn, print_each
        )
        model.eval()
        running_vloss = 0.0
        running_accuracy = 0.0
        running_f1 = 0.0
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

            # preds = voutputs.max(dim=1)[1].detach().numpy()
            preds = voutputs.max(dim=1)[1].numpy()
            running_accuracy += accuracy_score(vlabels, preds)
            running_f1 += accuracy_score(vlabels, preds)

        avg_vloss = float(running_vloss / (i + 1))
        avg_vaccuracy = float(running_accuracy / (i + 1))
        avg_vf1 = float(running_f1 / (i + 1))

        history["avg_accuracy"].append(avg_vaccuracy)
        history["avg_f1"].append(avg_vf1)
        history["avg_cv_loss"].append(avg_vloss)
        history["train_loss"] += loss_h
        
        print(f"EPOCH: {epoch} done. loss: {np.mean(loss_h)}, accuracy: {history['avg_accuracy'][-1]}")

    return history


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    with open(args.config, 'r') as cnf_file:
        config = yaml.load(cnf_file, Loader=yaml.FullLoader)
    print("Config:")
    pprint(config)
    model = LinearMnistNN()
    
    optimizer = Adam(list(model.parameters()), lr=0.0005, weight_decay=0.005)
    history = train_model(model, optimizer, config)
    vizualize_history(history)
