import math

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision


class AvgChannels(nn.Module):
    def __init__(self):
        super(AvgChannels, self).__init__()
    
    def forward(self, x):
        return torch.mean(x, dim=1, keepdim=True)


def prepare_two_layer_nn_paper(activation='relu', type='linear', number_of_outputs=2, number_of_features=3072,
                               number_of_channels=3):
    if type == 'conv' or type == 'conv_3':
        model = nn.Sequential(nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True),
                              nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True),
                              nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True),
                              nn.Flatten(),
                              nn.LazyLinear(number_of_outputs))
    elif type == 'conv_2':
        model = nn.Sequential(nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True),
                              nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True),
                              nn.Flatten(),
                              nn.LazyLinear(number_of_outputs))
    elif type == 'conv_1':
        model = nn.Sequential(nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True),
                              nn.Flatten(),
                              nn.LazyLinear(number_of_outputs))
    elif type == 'conv_3_relu':
        model = nn.Sequential(nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True),
                              nn.ReLU(),
                              nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True),
                              nn.ReLU(),
                              nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True),
                              nn.ReLU(),
                              nn.Flatten(),
                              nn.LazyLinear(number_of_outputs))
    elif type == 'linear':
        model = nn.Sequential(nn.Flatten(),
                              nn.Linear(number_of_features, number_of_features),
                              nn.Linear(number_of_features, number_of_features),
                              nn.Linear(number_of_features, number_of_features),
                              nn.LazyLinear(number_of_outputs))
    elif type == 'linear_3_relu':
        model = nn.Sequential(nn.Flatten(),
                              nn.Linear(number_of_features, number_of_features),
                              nn.ReLU(),
                              nn.Linear(number_of_features, number_of_features),
                              nn.ReLU(),
                              nn.Linear(number_of_features, number_of_features),
                              nn.ReLU(),
                              nn.LazyLinear(number_of_outputs))
    elif type == 'linear_2':
        model = nn.Sequential(nn.Flatten(),
                              nn.Linear(number_of_features, number_of_features),
                              nn.Linear(number_of_features, number_of_features),
                              nn.LazyLinear(number_of_outputs))
    elif type == 'linear_1':
        model = nn.Sequential(nn.Flatten(),
                              nn.Linear(number_of_features, number_of_features),
                              nn.LazyLinear(number_of_outputs))
    elif type == 'linear_0':
        model = nn.Sequential(nn.Flatten(),
                              nn.LazyLinear(number_of_outputs))
    else:
        assert False
    return model
