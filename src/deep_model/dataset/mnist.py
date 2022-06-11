from typing import Optional, Tuple
from torch.utils.data import Dataset

from ...data_preparation import load as load_mnist
from ...data_preparation import Dataset as MNIST_RawDs


class MNIST_Dataset(Dataset):
    def __init__(self,
                 digits: Tuple[int, int],
                 train: bool,
                 cv: bool,
                 test: bool,
                 length: Optional[int] = None,
                 ) -> None:
        super().__init__()
        if sum((train, cv, test)) != 1:
            raise ValueError("Choose only one - train/cv/test")
        # ugly code
        dataset: MNIST_RawDs = load_mnist(**digits)
        if train:
            self.data = dataset.train_data[:length]
            self.labels = dataset.train_targets[:length]
        elif cv:
            self.data = dataset.cv_data[:length]
            self.data = dataset.cv_targets[:length]
        elif test:
            self.data = dataset.test_data[:length]
            self.labels = dataset.test_targets[:length]
        self.length = length
        

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.length
