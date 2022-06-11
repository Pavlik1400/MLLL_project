from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..activation import get_activation


class LinearMnistNN(nn.Module):
    def __init__(self,
                 layers_sizes: Optional[List] = None,
                 activation: str="relu",
                 n_classes=2,
                 input_size=784):
        super(LinearMnistNN, self).__init__()
        if layers_sizes is None:
            layers_sizes = [16, 16]
        # self.layers_sizes = layers_sizes
        self.layers = []
        self.activation = get_activation(activation)
        self.n_classes = n_classes

        prev_size = input_size
        for cur_size in layers_sizes:
            self.layers.append(nn.Linear(prev_size, cur_size))
            prev_size = cur_size
        self.layers.append(nn.Linear(prev_size, n_classes))

    def forward(self, x):
        x = torch.flatten(x, 1)
        for llayer in self.layers:
            x = self.activation(llayer(x))
        x = F.softmax(x)
        return x
