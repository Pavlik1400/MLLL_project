from __future__ import annotations

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearMnistNN(nn.Module):
    def __init__(self,
                 layers_sizes: List | None = None,
                 activation=F.relu,
                 n_classes=2,
                 input_size=784):
        super().__init__()
        if layers_sizes is None:
            layers_sizes = [16, 16]
        # self.layers_sizes = layers_sizes
        self.activation = activation
        self.n_classes = n_classes

        prev_size = input_size
        for i, cur_size in enumerate(layers_sizes):
            self.__setattr__(f"linear_{i}", nn.Linear(prev_size, cur_size))
            prev_size = cur_size
        self.__setattr__(f"linear_{i+1}", nn.Linear(prev_size, n_classes))
        self.n_layers = len(layers_sizes) + 1

    def forward(self, x):
        x = torch.flatten(x, 1)
        for i in range(self.n_layers):
            # print(i, x.shape)
            x = self.activation(self.__getattr__(f"linear_{i}")(x))
        # print(i+1, x.shape)
        x = F.softmax(x, 1)
        # print(i+2, x.shape)
        return x
