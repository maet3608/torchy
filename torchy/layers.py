"""
.. module:: layers
   :synopsis: Various neural network layer definitions.
"""

import torch
import torch.nn as nn


class Conv2d_depthwise_sep(nn.Module):
    def __init__(self, nin, nout):
        super(Conv2d_depthwise_sep, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class Conv2d_spatial_sep(nn.Module):
    def __init__(self, nin, nout):
        super(Conv2d_spatial_sep, self).__init__()
        self.conv1 = nn.Conv2d(nin, 1, kernel_size=(1, 3), groups=1, padding=0)
        self.conv2 = nn.Conv2d(1, nout, kernel_size=(3, 1), groups=1, padding=1)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class RBF(nn.Module):
    def __init__(self):
        super(RBF, self).__init__()
        self.mean = nn.Parameter(torch.Tensor([0.0]))
        self.std = nn.Parameter(torch.Tensor([1.0]))

    def forward(self, x):
        gauss = torch.exp((-(x - self.mean) ** 2) / (2 * self.std ** 2))
        return gauss


class Swish(nn.Sigmoid):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * super(Swish, self).forward(x)
