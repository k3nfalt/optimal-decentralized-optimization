import torch
import torch.nn as nn
import torch.nn.functional as F


class Interpolate(nn.Module):
    def __init__(self, stride):
        super(Interpolate, self).__init__()
        self.interpolate = F.interpolate
        self.stride = stride
        
    def forward(self, x):
        orig_size = x.size()
        assert len(orig_size) == 4 and orig_size[2] % self.stride == 0 and orig_size[3] % self.stride == 0
        size = (orig_size[2] // self.stride, orig_size[3] // self.stride)
        x = self.interpolate(x, size=size, mode='bilinear')
        return x


class Repeat(nn.Module):
    def __init__(self, planes):
        super(Repeat, self).__init__()
        self.planes = planes
        
    def forward(self, x):
        number_of_channels = x.size(1)
        assert self.planes % number_of_channels == 0
        repeat_times = self.planes // number_of_channels
        return torch.repeat_interleave(x, repeat_times, dim=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, activation, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=True),
            )
            
        self.activation = activation

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class ResNetWithoutBN(nn.Module):
    def __init__(self, num_blocks, activation, 
                 first_conv=True, first_activation=True,
                 avg_pool=True,
                 num_classes=10, num_features=64,
                 number_of_channels=3):
        super(ResNetWithoutBN, self).__init__()
        block = BasicBlock
        self.in_planes = num_features
        self.activation = activation
        self.first_conv = first_conv
        self.first_activation = first_activation
        self.avg_pool = avg_pool
        
        if self.first_conv:
            self.conv1 = nn.Conv2d(number_of_channels, num_features, kernel_size=3,
                                   stride=1, padding=1, bias=True)
        self.layer1 = self._make_layer(block, num_features, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 2 * num_features, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * num_features, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * num_features, num_blocks[3], stride=2)
        self.linear = nn.LazyLinear(num_classes)
        

    def _make_layer(self, block, planes, num_blocks, stride):
        if num_blocks == 0:
            return nn.Identity()
        else:
            strides = [stride] + [1]*(num_blocks-1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, self.activation, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

    def forward(self, x):
        if self.first_conv:
            out = self.conv1(x)
        else:
            out = x
        if self.first_activation:
            out = self.activation(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        assert out.size(2) == out.size(3)
        if self.avg_pool:
            out = F.avg_pool2d(out, out.size(2))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def prepare_resnet_without_bn(activation='relu', blocks=None, **kwargs):
    activation = activation if activation is not None else 'relu'
    blocks = blocks if blocks is not None else [2, 2, 2, 2]
    activation_map = {'relu': F.relu,
                      'elu': F.elu,
                      'tanh': F.tanh}
    activation = activation_map[activation]
    return ResNetWithoutBN(blocks, activation, **kwargs)
