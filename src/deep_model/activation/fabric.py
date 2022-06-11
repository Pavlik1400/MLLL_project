from this import d
from typing import Callable
import torch.nn.functional as F
from ...helpers import fabric_check_key


__ACTIVATIONS = {
    "relu": F.relu,
    "sigmoid": F.sigmoid,
}


def get_activation(name: str) -> Callable:
    fabric_check_key(name, __ACTIVATIONS, "activation")
    return __ACTIVATIONS[name]
