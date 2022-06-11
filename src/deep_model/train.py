import torch
from torch.utils.data import DataLoader
from .model import get_model
from .optimizer import get_optimizer
from typing import Dict
from ..data_preparation import load as load_mnist
from ..helpers import birthdays_to_digits
from .dataset import MNIST_Dataset


def train_model(config: Dict):
    # Initialize all parts of algorithm
    model = get_model(config)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(config["optimizer"])

    # load dataset
    dig1, dig2 = birthdays_to_digits(
        config["PB"], config["VB"])
    data = load_mnist(dig1, dig2)

    train_ds = MNIST_Dataset((dig1, dig2), train=True)
    cv_ds = MNIST_Dataset((dig1, dig2), cv=True)

    # prepare data loader
    training_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=config["loader_num_workers"]
    )
    validation_loader = DataLoader(
        cv_ds, batch_size=config["batch_size"], shuffle=False, num_workers=config["loader_num_workers"]
    )
    
    validate_each_epoch = config["validate_each_epoch"]

    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        inputs, labels = data

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
        if i % validate_each_epoch == validate_each_epoch-1:
            last_loss = running_loss / validate_each_epoch  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss
