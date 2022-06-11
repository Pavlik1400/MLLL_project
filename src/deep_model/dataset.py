import os
import sys
sys.path.append(os.getcwd())
from typing import Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data_preparation import load as load_mnist
from src.data_preparation import Dataset as MNIST_RawDs


class MNIST_Dataset(Dataset):
    def __init__(self,
                 dataset: MNIST_RawDs,
                 train=False,
                 cv=False,
                 test=False,
                 length: Optional[int] = None,
                 ) -> None:
        super().__init__()
        if sum((train, cv, test)) != 1:
            raise ValueError("Choose only one - train/cv/test")
        # ugly code
        if train:
            self.data = dataset.train_data[:length]
            self.labels = dataset.train_targets[:length]
        elif cv:
            self.data = dataset.cv_data[:length]
            self.labels = dataset.cv_targets[:length]
        elif test:
            self.data = dataset.test_data[:length]
            self.labels = dataset.test_targets[:length]
        self.data = torch.from_numpy(self.data.astype(np.float32))
        self.labels = torch.from_numpy(self.labels.astype(np.int))
        self.length = length if length is not None else len(self.labels)

    def __getitem__(self, index: int):
        return torch.flatten(self.data[index]), self.labels[index]

    def __len__(self):
        return self.length
