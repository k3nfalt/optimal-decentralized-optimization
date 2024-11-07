import torch.nn as nn
import torch.nn.functional as F


def prepare_two_layer_nn(activation='relu'):
    activation = activation if activation is not None else 'relu'
    activation_map = {'relu': nn.ReLU,
                      'elu': nn.ELU,
                      'tanh': nn.Tanh}
    activation = activation_map[activation]
    model = nn.Sequential(nn.Flatten(),
                          nn.Linear(3072, 200, bias=True),
                          activation(),
                          nn.Linear(200, 200, bias=True),
                          activation(),
                          nn.Linear(200, 10, bias=True))
    return model
