import pytest
import itertools

import torch
from distributed_optimization_library.models.small_models_eos import Conv2dviaLinear, Circulant
from distributed_optimization_library.models.small_models_eos import circulant

@pytest.mark.parametrize("stride,mask,mask_conv", itertools.product([1, 2, 3], [False, True], [False, True]))
def test_conv2d_via_linear(stride, mask, mask_conv):
    if mask and mask_conv:
        return
    in_channels, out_channels, kernel_size = 5, 7, 3
    conv2d_via_linear = Conv2dviaLinear(in_channels, out_channels, kernel_size, stride, 
                                        mask=mask, mask_conv=mask_conv)
    x = torch.randn(3, in_channels, 11, 15)
    out_conv2d_via_linear = conv2d_via_linear.forward(x)
    conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    conv2d.weight.data = conv2d_via_linear._orig_weight
    conv2d.bias.data = conv2d_via_linear._orig_bias
    out_conv2d = conv2d.forward(x)
    assert (out_conv2d - out_conv2d_via_linear).abs().max() < 1e-3


def test_conv2d_via_linear_backward():
    in_channels, out_channels, kernel_size, stride = 5, 7, 3, 2
    conv2d_via_linear = Conv2dviaLinear(in_channels, out_channels, kernel_size, stride, mask_conv=True)
    x = torch.randn(3, in_channels, 11, 15)
    out_conv2d_via_linear = conv2d_via_linear.forward(x)
    loss = (torch.mean(out_conv2d_via_linear) - 1) ** 2
    loss.backward()
    grad_via_linear = conv2d_via_linear._weight.grad[:, 0, 0, :, :kernel_size, :kernel_size]
    
    conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    conv2d.weight.data = conv2d_via_linear._orig_weight
    conv2d.bias.data = conv2d_via_linear._orig_bias
    out_conv2d = conv2d.forward(x)
    loss = (torch.mean(out_conv2d) - 1) ** 2
    loss.backward()
    grad = conv2d.weight.grad
    assert (grad_via_linear - grad).abs().max() < 1e-3


def test_circulant_function():
    inp = torch.IntTensor([1, 2, 3])
    out = circulant(inp)
    assert torch.all(out == torch.IntTensor([[1, 2, 3], [3, 1, 2], [2, 3, 1]]))


def test_circulant_mini_test():
    in_features = 3
    x = torch.FloatTensor([[1, 0, 0]])
    cirl_layer = Circulant(in_features)
    out = cirl_layer(x)
    assert out.shape[0] == 1
    assert out.shape[1] == 3
    assert (out.flatten()[0] - cirl_layer._weight[0]).abs().max() < 1e-3
    assert (out.flatten()[1] - cirl_layer._weight[2]).abs().max() < 1e-3
    assert (out.flatten()[2] - cirl_layer._weight[1]).abs().max() < 1e-3


def test_circulant_mini_test_mask():
    in_features = 3
    x = torch.FloatTensor([[1, 2, 3]])
    cirl_layer = Circulant(in_features, mask_features=1)
    out = cirl_layer(x)
    assert out.shape[0] == 1
    assert out.shape[1] == 3
    assert (out.flatten()[0] - 1 * cirl_layer._weight[0]).abs().max() < 1e-3
    assert (out.flatten()[1] - 2 * cirl_layer._weight[0]).abs().max() < 1e-3
    assert (out.flatten()[2] - 3 * cirl_layer._weight[0]).abs().max() < 1e-3


def test_circulant_mini_test_mask_mask_features_2():
    in_features = 3
    x = torch.FloatTensor([[1, 2, 3]])
    cirl_layer = Circulant(in_features, mask_features=2)
    out = cirl_layer(x)
    assert out.shape[0] == 1
    assert out.shape[1] == 3
    assert (out.flatten()[0] - (1 * cirl_layer._weight[0] + 2 * cirl_layer._weight[1])).abs().max() < 1e-3
    assert (out.flatten()[1] - (2 * cirl_layer._weight[0] + 3 * cirl_layer._weight[1])).abs().max() < 1e-3
    assert (out.flatten()[2] - (3 * cirl_layer._weight[0] + 1 * cirl_layer._weight[1])).abs().max() < 1e-3
