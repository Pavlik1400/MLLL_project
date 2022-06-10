import numpy as np
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class Dataset:
    train_data: np.ndarray
    train_targets: np.ndarray
    cv_data: np.ndarray
    cv_targets: np.ndarray
    test_data: np.ndarray
    test_targets: np.ndarray


def load(digit1, digit2):
    train_cv_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    train_cv_data, train_cv_targets = transform(train_cv_dataset, digit1, digit2)
    train_data, cv_data, train_targets, cv_targets = train_test_split(train_cv_data, train_cv_targets, test_size=0.2)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
    test_data, test_targets = transform(test_dataset, digit1, digit2)
    return Dataset(train_data, train_targets, cv_data, cv_targets, test_data, test_targets)


def transform(dataset: datasets.MNIST, digit1: int, digit2: int):
    data = dataset.data.numpy().astype(np.float64) / 255
    targets = dataset.targets.numpy()
    mask = np.bitwise_or(targets == digit1, targets == digit2)
    data = data[mask]
    targets = targets[mask]
    return data.reshape(len(data), -1), targets == digit1


if __name__ == '__main__':
    dataset = load(1, 2)
    print(dataset)
