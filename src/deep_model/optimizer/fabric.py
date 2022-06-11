from typing import Callable
import torch
from ...helpers import fabric_check_key

__OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}


def get_optimizer(name: str) -> Callable:
    fabric_check_key(name, __OPTIMIZERS, "optimizer")
    return __OPTIMIZERS[name]
