import torch
from torch import nn


class SymmetricLogisticLoss(nn.Module):
    def __init__(self, starting_point):
        super().__init__()
        assert len(starting_point) == 2
        self._x = torch.nn.Parameter(torch.FloatTensor([starting_point[0]]))
        self._y = torch.nn.Parameter(torch.FloatTensor([starting_point[1]]))

    def forward(self):
        out = torch.logsumexp(torch.concat((torch.zeros(1), -self._x * self._y)), dim=0)
        out = out + torch.logsumexp(torch.concat((torch.zeros(1), self._x * self._y)), dim=0)
        return out / 2


class SqrtOnePlus(nn.Module):
    def __init__(self, starting_point):
        super().__init__()
        assert len(starting_point) == 2
        self._x = torch.nn.Parameter(torch.FloatTensor([starting_point[0]]))
        self._y = torch.nn.Parameter(torch.FloatTensor([starting_point[1]]))

    def forward(self):
        return torch.sqrt(1 + (self._x * self._y) ** 2)


class SquareOneMinusSquare(nn.Module):
    def __init__(self, starting_point):
        super().__init__()
        assert len(starting_point) == 2
        self._x = torch.nn.Parameter(torch.FloatTensor([starting_point[0]]))
        self._y = torch.nn.Parameter(torch.FloatTensor([starting_point[1]]))

    def forward(self):
        return (1 - (self._x * self._y) ** 2)**2


def prepare_toy_model(model_type, **kwargs):
    if model_type == 'symmetric_logistic_loss':
        return SymmetricLogisticLoss(**kwargs)
    elif model_type == 'sqrt_one_plus':
        return SqrtOnePlus(**kwargs)
    elif model_type == 'square_one_minus_square':
        return SquareOneMinusSquare(**kwargs)
