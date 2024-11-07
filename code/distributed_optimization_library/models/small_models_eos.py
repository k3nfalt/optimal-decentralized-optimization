import math

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        print(x.shape, np.prod(x.shape))
        return x
    

class AvgChannels(nn.Module):
    def __init__(self):
        super(AvgChannels, self).__init__()
    
    def forward(self, x):
        return torch.mean(x, dim=1, keepdim=True)


class Conv2dviaLinear(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, mask=False, mask_conv=False):
        super(Conv2dviaLinear, self).__init__()
        self._weights_are_initialized = False
        self._weight = torch.nn.UninitializedParameter()
        self._bias = torch.nn.UninitializedParameter()
        
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        assert not mask or not mask_conv
        self._use_mask = mask
        self._use_mask_conv = mask_conv

    def _init_weights(self, x):
        in_channels, out_channels, kernel_size, stride = \
            self._in_channels, self._out_channels, self._kernel_size, self._stride
        with torch.no_grad():
            tmp_conv = nn.Conv2d(in_channels, out_channels, kernel_size)
            height, width = x.shape[2], x.shape[3]
            out_height = math.floor((height - kernel_size) / stride + 1)
            out_width = math.floor((width - kernel_size) / stride + 1)
            self._mask = torch.zeros(out_channels, out_height, out_width, in_channels, height, width)
            
            self._height = height
            self._width = width
            self._out_channels = out_channels
            self._out_height = out_height
            self._out_width = out_width
            
            weight = self._repeat_weight(tmp_conv.weight)
            self._mask = self._repeat_weight(torch.ones_like(tmp_conv.weight))
            
            self._orig_weight = tmp_conv.weight.data
            self._orig_bias = tmp_conv.bias.data
            
            bias = torch.clone(tmp_conv.bias)
            bias = self._repeat_bias(bias)
            
            self._weight.materialize(weight.shape)
            self._weight.data = weight.to(self._weight.data)
            self._bias.materialize(bias.shape)
            self._bias.data = bias.to(self._bias.data)
            
            self._mask = self._mask.to(self._weight.data)
            
            self._weights_are_initialized = True
    
    def _repeat_weight(self, weight):
        new_weight = torch.zeros(self._out_channels, self._out_height, self._out_width, 
                                 self._in_channels, self._height, self._width, 
                                 dtype=weight.dtype, device=weight.device)
        for o in range(self._out_channels):
            for h in range(self._out_height):
                for w in range(self._out_width):
                    anchor_h = h * self._stride
                    anchor_w = w * self._stride
                    new_weight[o, h, w, :, anchor_h:anchor_h + self._kernel_size, anchor_w:anchor_w + self._kernel_size] = \
                        weight[o, :, :, :]
        return new_weight
    
    def _repeat_bias(self, bias):
        return bias.view(-1, 1, 1).repeat(1, self._out_height, self._out_width).view(1, -1)

    def forward(self, x):
        if not self._weights_are_initialized:
            self._init_weights(x)
        num_output = self._out_channels * self._out_height * self._out_width
        x_reshape = x.view(x.shape[0], -1)
        weight = self._weight
        if self._use_mask:
            weight = weight * self._mask
        if self._use_mask_conv:
            weight_first = weight[:, 0, 0, :, :self._kernel_size, :self._kernel_size]
            weight = self._repeat_weight(weight_first)
            
        w_reshape = weight.view(num_output, -1)
        out_reshape = x_reshape.matmul(w_reshape.t())
        bias = self._bias
        if self._use_mask or self._use_mask_conv:
            bias = bias.view(self._out_channels, self._out_height, self._out_width)[:, 0, 0]
            bias = self._repeat_bias(bias)
        out_reshape = out_reshape + bias
        out = out_reshape.view(-1, self._out_channels, self._out_height, self._out_width)
        return out


def circulant(tensor, dim=-1):
    len = tensor.shape[dim]
    tmp = torch.cat([tensor.flip((dim,)), torch.narrow(tensor.flip((dim,)), dim=dim, start=0, length=len-1)], dim=dim)
    return tmp.unfold(dim, len, 1).flip((-1,))


class Circulant(nn.Module):
    def __init__(self, in_channels, mask_features=None):
        super(Circulant, self).__init__()
        self._in_channels = in_channels
        self._mask_features = mask_features
        
        tmp_conv = nn.Linear(in_channels, 1)
        init_weight = tmp_conv.weight.data
        assert init_weight.shape[0] == 1
        init_weight = init_weight[0]
        self._weight = torch.nn.Parameter(torch.clone(init_weight))
    
    def forward(self, x):
        weight = self._weight
        if self._mask_features is not None:
            mask = torch.ones_like(self._weight)
            mask[self._mask_features:] = 0
            weight = weight * mask
        weight_circulant = circulant(weight)
        return x.matmul(weight_circulant.t())


class NormalizedCirculantMask2(nn.Module):
    def __init__(self, in_channels):
        super(NormalizedCirculantMask2, self).__init__()
        self._in_channels = in_channels
        tmp_conv = nn.Linear(in_channels, 1)
        init_weight = tmp_conv.weight.data
        assert init_weight.shape[0] == 1
        init_weight = torch.clone(init_weight[0][:1])
        self._weight = torch.nn.Parameter(init_weight)
    
    def forward(self, x):
        weight = torch.concat((self._weight, 1 - self._weight))
        weight = torch.nn.functional.pad(weight, (0, self._in_channels - len(weight)), "constant", 0)
        weight_circulant = circulant(weight)
        return x.matmul(weight_circulant.t())


# class Conv2dviaLinear(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size):
#         super(Conv2dviaLinear, self).__init__()
#         self._kernel_size = kernel_size
#         tmp_conv = nn.Conv2d(in_channels, out_channels, kernel_size)
#         self._weight = torch.nn.Parameter(torch.copy(tmp_conv._weight))
#         self._bias = torch.nn.Parameter(torch.copy(tmp_conv._bias))

#     def forward(self, x):
#         # inp = torch.randn(1, 3, 10, 12)
#         # w = torch.randn(2, 3, 4, 5)
#         x_unf = F.unfold(x, (self._kernel_size, self._kernel_size))
#         w = self._weight
#         out_unf = x_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
#         out_size = [x.shape[2] - self._kernel_size + 1, x.shape[3] - self._kernel_size + 1]
#         out = torch.nn.functional.fold(out_unf, out_size, (1, 1))
#         return out


def prepare_two_layer_nn_eos(activation='relu', type='linear', number_of_outputs=2, number_of_features=3072,
                             number_of_channels=3):
    assert activation == 'elu'
    activation = activation if activation is not None else 'relu'
    activation_map = {'relu': nn.ReLU,
                      'elu': nn.ELU,
                      'tanh': nn.Tanh}
    activation = activation_map[activation]
    if type == 'linear':
        model = nn.Sequential(nn.Flatten(),
                              nn.Linear(number_of_features, 32, bias=True),
                              nn.Linear(32, 32, bias=True),
                              nn.Linear(32, number_of_outputs, bias=True))
    elif type == 'linear_one_no_bias':
        model = nn.Sequential(nn.Flatten(),
                              nn.LazyLinear(number_of_outputs, bias=False))
    elif type == 'linear_5_layers':
        model = nn.Sequential(nn.Flatten(),
                              nn.Linear(number_of_features, 1024, bias=True),
                              nn.Linear(1024, 512, bias=True),
                              nn.Linear(512, 256, bias=True),
                              nn.Linear(256, 128, bias=True),
                              nn.Linear(128, number_of_outputs, bias=True))
    elif type == 'linear_5_layers_no_bias':
        model = nn.Sequential(nn.Flatten(),
                              nn.Linear(number_of_features, 1024, bias=False),
                              nn.Linear(1024, 512, bias=False),
                              nn.Linear(512, 256, bias=False),
                              nn.Linear(256, 128, bias=False),
                              nn.Linear(128, number_of_outputs, bias=False))
    elif type == 'circulant_2_layers_no_bias':
        model = nn.Sequential(nn.Flatten(),
                              Circulant(number_of_features),
                              nn.LazyLinear(number_of_outputs, bias=False))
    elif type == 'circulant_2_layers_mask_3_no_bias':
        model = nn.Sequential(nn.Flatten(),
                              Circulant(number_of_features, mask_features=3),
                              nn.LazyLinear(number_of_outputs, bias=False))
    elif type == 'circulant_2_layers_mask_1_no_bias':
        model = nn.Sequential(nn.Flatten(),
                              Circulant(number_of_features, mask_features=1),
                              nn.LazyLinear(number_of_outputs, bias=False))
    elif type == 'one_feature_circulant_2_layers_mask_1_no_bias':
        model = nn.Sequential(nn.Flatten(),
                              AvgChannels(),
                              Circulant(1, mask_features=1),
                              nn.LazyLinear(number_of_outputs, bias=False))
    elif type == 'circulant_2_layers_mask_2_no_bias':
        model = nn.Sequential(nn.Flatten(),
                              Circulant(number_of_features, mask_features=2),
                              nn.LazyLinear(number_of_outputs, bias=False))
    elif type == 'normalized_circulant_2_layers_mask_2_no_bias':
        model = nn.Sequential(nn.Flatten(),
                              NormalizedCirculantMask2(number_of_features),
                              nn.LazyLinear(number_of_outputs, bias=False))
    elif type == 'circulant_2_layers_mask_10_no_bias':
        model = nn.Sequential(nn.Flatten(),
                              Circulant(number_of_features, mask_features=10),
                              nn.LazyLinear(number_of_outputs, bias=False))
    elif type == 'circulant_2_layers_mask_100_no_bias':
        model = nn.Sequential(nn.Flatten(),
                              Circulant(number_of_features, mask_features=100),
                              nn.LazyLinear(number_of_outputs, bias=False))
    elif type == 'circulant_2_layers_mask_1000_no_bias':
        model = nn.Sequential(nn.Flatten(),
                              Circulant(number_of_features, mask_features=1000),
                              nn.LazyLinear(number_of_outputs, bias=False))
    elif type == 'circulant_2_layers_mask_1500_no_bias':
        model = nn.Sequential(nn.Flatten(),
                              Circulant(number_of_features, mask_features=1500),
                              nn.LazyLinear(number_of_outputs, bias=False))
    elif type == 'circulant_2_layers_mask_2000_no_bias':
        model = nn.Sequential(nn.Flatten(),
                              Circulant(number_of_features, mask_features=2000),
                              nn.LazyLinear(number_of_outputs, bias=False))
    elif type == 'circulant_2_layers_mask_3000_no_bias':
        model = nn.Sequential(nn.Flatten(),
                              Circulant(number_of_features, mask_features=3000),
                              nn.LazyLinear(number_of_outputs, bias=False))
    elif type == 'grayscale_circulant_2_layers_mask_3_no_bias':
        model = nn.Sequential(torchvision.transforms.Grayscale(num_output_channels=1),
                              nn.Flatten(),
                              Circulant(1024, mask_features=3),
                              nn.LazyLinear(number_of_outputs, bias=False))
    elif type == 'conv':
        model = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=True),
                              nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=True),
                              nn.Conv2d(16, number_of_outputs, kernel_size=3, stride=2, padding=1, bias=True),
                              nn.AdaptiveAvgPool2d(1),
                              nn.Flatten())
    elif type == 'linear_via_conv':
        model = nn.Sequential(nn.Conv2d(3, 32, kernel_size=32, stride=1, padding=0, bias=True),
                              nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0, bias=True),
                              nn.Conv2d(32, number_of_outputs, kernel_size=1, stride=1, padding=0, bias=True),
                              nn.AdaptiveAvgPool2d(1),
                              nn.Flatten())
    elif type == 'linear_via_conv_8':
        model = nn.Sequential(nn.Conv2d(3, 8, kernel_size=32, stride=1, padding=0, bias=True),
                              nn.Conv2d(8, 16, kernel_size=1, stride=1, padding=0, bias=True),
                              nn.Conv2d(16, number_of_outputs, kernel_size=1, stride=1, padding=0, bias=True),
                              nn.AdaptiveAvgPool2d(1),
                              nn.Flatten())
    elif type == 'conv_stride_1':
        model = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=0, bias=True),
                              nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0, bias=True),
                              nn.Conv2d(16, number_of_outputs, kernel_size=3, stride=1, padding=0, bias=True),
                              nn.AdaptiveAvgPool2d(1),
                              nn.Flatten())
    elif type == 'conv_stride_2':
        model = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=0, bias=True),
                              nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0, bias=True),
                              nn.Conv2d(16, number_of_outputs, kernel_size=3, stride=2, padding=0, bias=True),
                              nn.AdaptiveAvgPool2d(1),
                              nn.Flatten())
    elif type == 'conv_stride_2_2_layers_large':
        model = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=True),
                              nn.Conv2d(64, number_of_outputs, kernel_size=3, stride=2, padding=1, bias=True),
                              nn.AdaptiveAvgPool2d(1),
                              nn.Flatten())
    elif type == 'conv_stride_2_5_layers':
        model = nn.Sequential(nn.Conv2d(number_of_channels, 8, kernel_size=3, stride=2, padding=1, bias=True),
                              nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=True),
                              nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=True),
                              nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),
                              nn.Conv2d(64, number_of_outputs, kernel_size=3, stride=2, padding=1, bias=True),
                              nn.AdaptiveAvgPool2d(1),
                              nn.Flatten())
    elif type == 'conv_stride_2_5_layers_acitvation':
        model = nn.Sequential(nn.Conv2d(number_of_channels, 8, kernel_size=3, stride=2, padding=1, bias=True),
                              activation(),
                              nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=True),
                              activation(),
                              nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=True),
                              activation(),
                              nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),
                              activation(),
                              nn.Conv2d(64, number_of_outputs, kernel_size=3, stride=2, padding=1, bias=True),
                              nn.AdaptiveAvgPool2d(1),
                              nn.Flatten())
    elif type == 'conv_stride_2_4_layers':
        model = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=True),
                              nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=True),
                              nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=True),
                              nn.Conv2d(32, number_of_outputs, kernel_size=3, stride=2, padding=1, bias=True),
                              nn.AdaptiveAvgPool2d(1),
                              nn.Flatten())
    elif type == 'conv_stride_2_3_layers':
        model = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=True),
                              nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=True),
                              nn.Conv2d(16, number_of_outputs, kernel_size=3, stride=2, padding=1, bias=True),
                              nn.AdaptiveAvgPool2d(1),
                              nn.Flatten())
    elif type == 'conv_stride_2_2_layers':
        model = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=True),
                              nn.Conv2d(8, number_of_outputs, kernel_size=3, stride=2, padding=1, bias=True),
                              nn.AdaptiveAvgPool2d(1),
                              nn.Flatten())
    elif type == 'conv_stride_2_1_layers':
        model = nn.Sequential(nn.Conv2d(3, number_of_outputs, kernel_size=3, stride=2, padding=1, bias=True),
                              nn.AdaptiveAvgPool2d(1),
                              nn.Flatten())
    elif type == 'conv_stride_2_linear':
        model = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=True),
                              nn.Flatten(),
                              nn.LazyLinear(number_of_outputs))
    elif type == 'grayscale_conv_stride_2_linear':
        model = nn.Sequential(torchvision.transforms.Grayscale(num_output_channels=1),
                              nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1, bias=True),
                              nn.Flatten(),
                              nn.LazyLinear(number_of_outputs))
    elif type == 'grayscale_conv_stride_2_out_feat_1_linear':
        model = nn.Sequential(torchvision.transforms.Grayscale(num_output_channels=1),
                              nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=True),
                              nn.Flatten(),
                              nn.LazyLinear(number_of_outputs))
    elif type == 'grayscale_conv_stride_1_kernel_size_1_3_out_feat_1_linear':
        model = nn.Sequential(torchvision.transforms.Grayscale(num_output_channels=1),
                              nn.Conv2d(1, 1, kernel_size=(1, 3), stride=1, padding=1, bias=True),
                              nn.Flatten(),
                              nn.LazyLinear(number_of_outputs))
    elif type == 'conv_kernel_size_1_stride_2_linear':
        model = nn.Sequential(nn.Conv2d(3, 8, kernel_size=1, stride=2, padding=1, bias=True),
                              nn.Flatten(),
                              nn.LazyLinear(number_of_outputs))
    elif type == 'conv_stride_2_5_layers_no_bias':
        model = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=False),
                              nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1, bias=False),
                              nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
                              nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                              nn.Conv2d(64, number_of_outputs, kernel_size=3, stride=2, padding=1, bias=False),
                              nn.AdaptiveAvgPool2d(1),
                              nn.Flatten())
    elif type == 'conv_stride_2_4_layers_no_pad':
        model = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=0, bias=True),
                              nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0, bias=True),
                              nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, bias=True),
                              nn.Conv2d(32, number_of_outputs, kernel_size=3, stride=2, padding=0, bias=True),
                              nn.AdaptiveAvgPool2d(1),
                              nn.Flatten())
    elif type == 'conv_stride_2_5_layers_no_stride':
        model = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=True),
                              nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=True),
                              nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True),
                              nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
                              nn.Conv2d(64, number_of_outputs, kernel_size=3, stride=1, padding=1, bias=True),
                              nn.AdaptiveAvgPool2d(1),
                              nn.Flatten())
    elif type == 'conv_stride_2_4_layers_via_linear':
        model = nn.Sequential(Conv2dviaLinear(3, 8, kernel_size=3, stride=2),
                              Conv2dviaLinear(8, 16, kernel_size=3, stride=2),
                              Conv2dviaLinear(16, 32, kernel_size=3, stride=2),
                              Conv2dviaLinear(32, number_of_outputs, kernel_size=3, stride=2),
                              nn.AdaptiveAvgPool2d(1),
                              nn.Flatten())
    elif type == 'conv_stride_2_4_layers_via_linear_mask':
        model = nn.Sequential(Conv2dviaLinear(3, 8, kernel_size=3, stride=2, mask=True),
                              Conv2dviaLinear(8, 16, kernel_size=3, stride=2, mask=True),
                              Conv2dviaLinear(16, 32, kernel_size=3, stride=2, mask=True),
                              Conv2dviaLinear(32, number_of_outputs, kernel_size=3, stride=2, mask=True),
                              nn.AdaptiveAvgPool2d(1),
                              nn.Flatten())
    elif type == 'conv_stride_2_4_layers_via_linear_mask_conv':
        model = nn.Sequential(Conv2dviaLinear(3, 8, kernel_size=3, stride=2, mask_conv=True),
                              Conv2dviaLinear(8, 16, kernel_size=3, stride=2, mask_conv=True),
                              Conv2dviaLinear(16, 32, kernel_size=3, stride=2, mask_conv=True),
                              Conv2dviaLinear(32, number_of_outputs, kernel_size=3, stride=2, mask_conv=True),
                              nn.AdaptiveAvgPool2d(1),
                              nn.Flatten())
    elif type == 'conv_stride_2_5_layers_via_linear':
        model = nn.Sequential(nn.ZeroPad2d(1), Conv2dviaLinear(3, 8, kernel_size=3, stride=2),
                              nn.ZeroPad2d(1), Conv2dviaLinear(8, 16, kernel_size=3, stride=2),
                              nn.ZeroPad2d(1), Conv2dviaLinear(16, 32, kernel_size=3, stride=2),
                              nn.ZeroPad2d(1), Conv2dviaLinear(32, 64, kernel_size=3, stride=2),
                              nn.ZeroPad2d(1), Conv2dviaLinear(64, number_of_outputs, kernel_size=3, stride=2),
                              nn.AdaptiveAvgPool2d(1),
                              nn.Flatten())
    elif type == 'conv_stride_2_5_layers_via_linear_mask':
        model = nn.Sequential(nn.ZeroPad2d(1), Conv2dviaLinear(3, 8, kernel_size=3, stride=2, mask=True),
                              nn.ZeroPad2d(1), Conv2dviaLinear(8, 16, kernel_size=3, stride=2, mask=True),
                              nn.ZeroPad2d(1), Conv2dviaLinear(16, 32, kernel_size=3, stride=2, mask=True),
                              nn.ZeroPad2d(1), Conv2dviaLinear(32, 64, kernel_size=3, stride=2, mask=True),
                              nn.ZeroPad2d(1), Conv2dviaLinear(64, number_of_outputs, kernel_size=3, stride=2, mask=True),
                              nn.AdaptiveAvgPool2d(1),
                              nn.Flatten())
    elif type == 'conv_stride_2_1_layer':
        model = nn.Sequential(nn.Conv2d(3, number_of_outputs, kernel_size=3, stride=2, padding=1, bias=True),
                              nn.AdaptiveAvgPool2d(1),
                              nn.Flatten())
    elif type == 'conv_1x1':
        model = nn.Sequential(AvgChannels(),
                              nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=True),
                              nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=True),
                              nn.Flatten(),
                              nn.Linear(1024, number_of_outputs, bias=True))
    return model
