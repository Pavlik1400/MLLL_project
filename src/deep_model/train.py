import os
import sys

sys.path.append(os.getcwd())
import copy
from src.deep_model.dataset import MNIST_Dataset
from src.deep_model.linear_nn import LinearMnistNN
from src.helpers import birthdays_to_digits
from src.data_preparation import load as load_mnist
from sklearn.metrics import f1_score, accuracy_score
from argparse import ArgumentParser
import yaml
from typing import Callable, Dict, List, Tuple
from torch.utils.data import DataLoader
import torch
from torch.optim import Adam, SGD
from pprint import pprint
import numpy as np
from eval import eval_model
from torch.nn.functional import relu


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# seed = 41
# torch.manual_seed(seed)
# np.random.seed(seed)


def train_one_epoch(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    training_loader: DataLoader,
                    loss_fn: Callable,
                    print_each: int,
                    verbose=False,
                    ) -> List[float]:
    running_loss = 0.
    # last_loss = 0.
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
        if i % print_each == print_each - 1:
            last_loss = running_loss / print_each  # loss per batch
            verbose and print(f"  batch {i + 1} loss: {last_loss}")
            loss_history.append(float(last_loss))
            running_loss = 0.
    return loss_history


def train_model(model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                config: Dict,
                save_path: str = "models"
                ) -> Tuple[Dict[str, List], torch.nn.Module | None]:
    if not os.path.exists(save_path):
        os.makedirs(save_path)

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

    best_model = None
    best_cv_acc = -1
    best_train_loss = -1

    for epoch in range(n_epoch):
        model.train()
        loss_h = train_one_epoch(
            model, optimizer, training_loader, loss_fn, print_each, verbose=False
        )
        model.eval()
        running_vloss = 0.0
        running_accuracy = 0.0
        running_f1 = 0.0
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss.item()

            # preds = voutputs.max(dim=1)[1].detach().numpy()
            preds = voutputs.max(dim=1)[1].numpy()
            running_accuracy += accuracy_score(vlabels, preds)
            running_f1 += f1_score(vlabels, preds)

        avg_vloss = float(running_vloss / (i + 1))
        avg_vaccuracy = float(running_accuracy / (i + 1))
        avg_vf1 = float(running_f1 / (i + 1))

        history["avg_accuracy"].append(avg_vaccuracy)
        history["avg_f1"].append(avg_vf1)
        history["avg_cv_loss"].append(avg_vloss)
        history["train_loss"] += loss_h

        avg_train_loss = np.mean(loss_h)
        print(f"EPOCH: {epoch} done. loss: {avg_train_loss}, accuracy: {history['avg_accuracy'][-1]}")

        # update best model
        if avg_vaccuracy > best_cv_acc:
            best_model = copy.deepcopy(model)
            best_cv_acc = avg_vaccuracy
            torch.save(model.state_dict(), os.path.join(save_path, "best_val"))
        if avg_train_loss > best_train_loss:
            torch.save(model.state_dict(), os.path.join(save_path, "best_train"))
            best_train_loss = avg_train_loss

    return history, best_model


def grid_searc(
        layers: List[List[int]],
        optimizers: List[Tuple[type, Dict[str, float]]],
        activations: List[Callable],
        config: Dict,
):
    very_best_model = None
    very_best_val_accuracy = -1
    very_best_config = {}
    results = []

    for layer_sizes in layers:
        print(f"Current layers: {layer_sizes}")
        for opt_cnf in optimizers:
            opt_cnf_full = copy.deepcopy(opt_cnf[1])
            opt_cnf_full.update({"name": opt_cnf[0].__class__.__name__})
            print(f"  Current optimizer: {opt_cnf_full}")
            for activation in activations:
                print(f"    Current activation: {activation.__class__.__name__}")
                model = LinearMnistNN(layer_sizes, activation)
                optimizer = opt_cnf[0](list(model.parameters()), **opt_cnf[1])

                configuration_name = f"{layer_sizes}--{opt_cnf_full}--{activation.__class__.__name__}"
                history, best_model = train_model(
                    model, optimizer, config,
                    save_path=f"models/{configuration_name}"
                )

                cur_best_cv_acc = min(history["avg_cv_loss"])
                # cur_best_cv_f1 = min(history["avg_f1"])

                cur_configuration = {
                    "layers": layer_sizes,
                    "activation": activation.__class__.__name__,
                    "optimizer_cnf": opt_cnf_full,
                }

                if cur_best_cv_acc > very_best_val_accuracy:
                    very_best_model = best_model
                    very_best_config = cur_configuration

                results.append(cur_configuration)

    print(f"best model has accuracy {very_best_val_accuracy}, and config:")
    pprint(very_best_config)
    return results, very_best_model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    with open(args.config, 'r') as cnf_file:
        config = yaml.load(cnf_file, Loader=yaml.FullLoader)
    print("Config:")
    pprint(config)
    # model = LinearMnistNN()
    # optimizer = Adam(list(model.parameters()), lr=0.0005, weight_decay=0.005)
    # # optimizer = Adam(list(model.parameters()), lr=0.005, weight_decay=0.001)
    # history, best_model = train_model(model, optimizer, config)
    # vizualize_history(history)
    #
    # eval_model(best_model, config)

    # layers = [
    #     [16, 16],
    #     [64, 32, 16]
    # ]
    #
    # optimizers = [
    #     (Adam, {"lr": 0.005, "weights_decay": 0.005}),
    #     (Adam, {"lr": 0.0005, "weights_decay": 0.001}),
    #     (SGD, {"lr": 0.005, "weights_decay": 0.005}),
    #     (SGD, {"lr": 0.0005, "weights_decay": 0.001}),
    # ]
    #
    # activations = [relu, sigmoid]

    layers = [
        [16, 16],
        # [64, 32, 16]
    ]

    optimizers = [
        (Adam, {"lr": 0.005, "weight_decay": 0.005}),
        # (Adam, {"lr": 0.0005, "weights_decay": 0.001}),
        # (SGD, {"lr": 0.005, "weights_decay": 0.005}),
        (SGD, {"lr": 0.0005, "weight_decay": 0.001}),
    ]

    activations = [torch.sigmoid]

    results, very_best_model = grid_searc(layers, optimizers, activations, config)
    eval_model(very_best_model, config)
    with open("deep_results.yaml", 'w') as res_file:
        yaml.dump(results, res_file)
