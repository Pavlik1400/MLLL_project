import numpy as np
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import yaml
from argparse import ArgumentParser
from pprint import pprint
from helpers import birthdays_to_digits


@dataclass
class Dataset:
    train_data: np.ndarray
    train_targets: np.ndarray
    cv_data: np.ndarray
    cv_targets: np.ndarray
    test_data: np.ndarray
    test_targets: np.ndarray


def load(digit1: int, digit2: int) -> Dataset:
    """Loads MNITS dataset with 2 numbers

    Args:
        digit1 (int): first number
        digit2 (int): second number

    Returns:
        Dataset: data
    """
    train_cv_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=None)
    train_cv_data, train_cv_targets = transform(
        train_cv_dataset, digit1, digit2)
    train_data, cv_data, train_targets, cv_targets = train_test_split(
        train_cv_data, train_cv_targets, test_size=0.2)
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=None)
    test_data, test_targets = transform(test_dataset, digit1, digit2)
    return Dataset(train_data, train_targets, cv_data, cv_targets, test_data, test_targets)


def transform(dataset: datasets.MNIST, digit1: int, digit2: int) -> np.ndarray:
    """Given MNIST dataset, normalizes images to range 0:255 and leave only 2 digits

    Args:
        dataset (datasets.MNIST): full dataset loaded by pytorch
        digit1 (int):
        digit2 (int):

    Returns:
        np.ndarray:
    """
    data = dataset.data.numpy().astype(np.float64) / 255
    targets = dataset.targets.numpy()
    mask = np.bitwise_or(targets == digit1, targets == digit2)
    data = data[mask]
    targets = targets[mask]
    return data.reshape(len(data), -1), targets == digit1


if __name__ == '__main__':
    parser = ArgumentParser("Loads MNIST dataset with 2 digits")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as cnf_file:
        config = yaml.load(cnf_file, Loader=yaml.FullLoader)
    print("config: ")
    pprint(config)
    b2, b1 = config["Pavlo_birth"], config["Volodymyr_birth"]
    dig1, dig2 = birthdays_to_digits(b1, b2)
    dataset = load(dig1, dig2)

    print(f"Dataset with digits: {dig1}, {dig2}")
    print(f"Size of train: {len(dataset.train_data)}, \
            ratio: {len(dataset.train_targets == dig1) / len(dataset.train_targets == dig2)}")
    print(f"Size of cv: {len(dataset.cv_data)}, \
            ratio: {len(dataset.cv_targets == dig1) / len(dataset.cv_targets == dig2)}")
    print(f"Size of test: {len(dataset.test_data)}, \
            ratio: {len(dataset.test_targets == dig1) / len(dataset.test_targets == dig2)}")

    # print(dataset)
