import sys
_module = sys.modules[__name__]
del sys
data = _module
main_binary = _module
main_binary_hinge = _module
main_mnist = _module
models = _module
alexnet = _module
alexnet_binary = _module
binarized_modules = _module
resnet = _module
resnet_binary = _module
vgg_cifar10_binary = _module
preprocess = _module
utils = _module

from _paritybench_helpers import _mock_config
from unittest.mock import mock_open, MagicMock
from torch.autograd import Function
from torch.nn import Module
open = mock_open()
logging = sys = argparse = MagicMock()
ArgumentParser = argparse.ArgumentParser
_global_config = args = argv = cfg = config = params = _mock_config()
argparse.ArgumentParser.return_value.parse_args.return_value = _global_config
sys.argv = _global_config
__version__ = '1.0.0'


import time


import logging


import torch


import torch.nn as nn


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim


import torch.utils.data


from torch.autograd import Variable


import torch.nn.functional as F


import torch.optim as optim


import math


from torch.autograd import Function


import numpy as np


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.infl_ratio = 3
        self.fc1 = BinarizeLinear(784, 2048 * self.infl_ratio)
        self.htanh1 = nn.Hardtanh()
        self.bn1 = nn.BatchNorm1d(2048 * self.infl_ratio)
        self.fc2 = BinarizeLinear(2048 * self.infl_ratio, 2048 * self.
            infl_ratio)
        self.htanh2 = nn.Hardtanh()
        self.bn2 = nn.BatchNorm1d(2048 * self.infl_ratio)
        self.fc3 = BinarizeLinear(2048 * self.infl_ratio, 2048 * self.
            infl_ratio)
        self.htanh3 = nn.Hardtanh()
        self.bn3 = nn.BatchNorm1d(2048 * self.infl_ratio)
        self.fc4 = nn.Linear(2048 * self.infl_ratio, 10)
        self.logsoftmax = nn.LogSoftmax()
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.htanh1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.htanh2(x)
        x = self.fc3(x)
        x = self.drop(x)
        x = self.bn3(x)
        x = self.htanh3(x)
        x = self.fc4(x)
        return self.logsoftmax(x)


class AlexNetOWT_BN(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNetOWT_BN, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11,
            stride=4, padding=2, bias=False), nn.MaxPool2d(kernel_size=3,
            stride=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d
            (64, 192, kernel_size=5, padding=2, bias=False), nn.MaxPool2d(
            kernel_size=3, stride=2), nn.ReLU(inplace=True), nn.BatchNorm2d
            (192), nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False
            ), nn.ReLU(inplace=True), nn.BatchNorm2d(384), nn.Conv2d(384, 
            256, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=
            True), nn.BatchNorm2d(256), nn.Conv2d(256, 256, kernel_size=3,
            padding=1, bias=False), nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True), nn.BatchNorm2d(256))
        self.classifier = nn.Sequential(nn.Linear(256 * 6 * 6, 4096, bias=
            False), nn.BatchNorm1d(4096), nn.ReLU(inplace=True), nn.Dropout
            (0.5), nn.Linear(4096, 4096, bias=False), nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(4096,
            num_classes))
        self.regime = {(0): {'optimizer': 'SGD', 'lr': 0.01, 'weight_decay':
            0.0005, 'momentum': 0.9}, (10): {'lr': 0.005}, (15): {'lr': 
            0.001, 'weight_decay': 0}, (20): {'lr': 0.0005}, (25): {'lr': 
            0.0001}}
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
        self.input_transform = {'train': transforms.Compose([transforms.
            Scale(256), transforms.RandomCrop(224), transforms.
            RandomHorizontalFlip(), transforms.ToTensor(), normalize]),
            'eval': transforms.Compose([transforms.Scale(256), transforms.
            CenterCrop(224), transforms.ToTensor(), normalize])}

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x


class AlexNetOWT_BN(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNetOWT_BN, self).__init__()
        self.ratioInfl = 3
        self.features = nn.Sequential(BinarizeConv2d(3, int(64 * self.
            ratioInfl), kernel_size=11, stride=4, padding=2), nn.MaxPool2d(
            kernel_size=3, stride=2), nn.BatchNorm2d(int(64 * self.
            ratioInfl)), nn.Hardtanh(inplace=True), BinarizeConv2d(int(64 *
            self.ratioInfl), int(192 * self.ratioInfl), kernel_size=5,
            padding=2), nn.MaxPool2d(kernel_size=3, stride=2), nn.
            BatchNorm2d(int(192 * self.ratioInfl)), nn.Hardtanh(inplace=
            True), BinarizeConv2d(int(192 * self.ratioInfl), int(384 * self
            .ratioInfl), kernel_size=3, padding=1), nn.BatchNorm2d(int(384 *
            self.ratioInfl)), nn.Hardtanh(inplace=True), BinarizeConv2d(int
            (384 * self.ratioInfl), int(256 * self.ratioInfl), kernel_size=
            3, padding=1), nn.BatchNorm2d(int(256 * self.ratioInfl)), nn.
            Hardtanh(inplace=True), BinarizeConv2d(int(256 * self.ratioInfl
            ), 256, kernel_size=3, padding=1), nn.MaxPool2d(kernel_size=3,
            stride=2), nn.BatchNorm2d(256), nn.Hardtanh(inplace=True))
        self.classifier = nn.Sequential(BinarizeLinear(256 * 6 * 6, 4096),
            nn.BatchNorm1d(4096), nn.Hardtanh(inplace=True), BinarizeLinear
            (4096, 4096), nn.BatchNorm1d(4096), nn.Hardtanh(inplace=True),
            BinarizeLinear(4096, num_classes), nn.BatchNorm1d(1000), nn.
            LogSoftmax())
        self.regime = {(0): {'optimizer': 'Adam', 'lr': 0.005}, (20): {'lr':
            0.001}, (30): {'lr': 0.0005}, (35): {'lr': 0.0001}, (40): {'lr':
            1e-05}}
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
        self.input_transform = {'train': transforms.Compose([transforms.
            Scale(256), transforms.RandomCrop(224), transforms.
            RandomHorizontalFlip(), transforms.ToTensor(), normalize]),
            'eval': transforms.Compose([transforms.Scale(256), transforms.
            CenterCrop(224), transforms.ToTensor(), normalize])}

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 6 * 6)
        x = self.classifier(x)
        return x


class HingeLoss(nn.Module):

    def __init__(self):
        super(HingeLoss, self).__init__()
        self.margin = 1.0

    def hinge_loss(self, input, target):
        output = self.margin - input.mul(target)
        output[output.le(0)] = 0
        return output.mean()

    def forward(self, input, target):
        return self.hinge_loss(input, target)


def Binarize(tensor, quant_mode='det'):
    if quant_mode == 'det':
        return tensor.sign()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)
            ).clamp_(0, 1).round().mul_(2).add_(-1)


class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if input.size(1) != 784:
            input.data = Binarize(input.data)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = Binarize(self.weight.org)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)
        return out


class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if input.size(1) != 3:
            input.data = Binarize(input.data)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = Binarize(self.weight.org)
        out = nn.functional.conv2d(input, self.weight, None, self.stride,
            self.padding, self.dilation, self.groups)
        if not self.bias is None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def Binaryconv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return BinarizeConv2d(in_planes, out_planes, kernel_size=3, stride=
        stride, padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = BinarizeConv2d(inplanes, planes, kernel_size=1, bias=False
            )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = BinarizeConv2d(planes, planes, kernel_size=3, stride=
            stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = BinarizeConv2d(planes, planes * 4, kernel_size=1, bias
            =False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.tanh = nn.Hardtanh(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = xNone
        pdb.set_trace()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.tanh(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        if self.do_bntan:
            out = self.bn2(out)
            out = self.tanh2(out)
        return out


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

    def _make_layer(self, block, planes, blocks, stride=1, do_bntan=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(BinarizeConv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks - 1):
            layers.append(block(self.inplanes, planes))
        layers.append(block(self.inplanes, planes, do_bntan=do_bntan))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.bn1(x)
        x = self.tanh1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)
        x = self.tanh2(x)
        x = self.fc(x)
        x = self.bn3(x)
        x = self.logsoftmax(x)
        return x


class VGG_Cifar10(nn.Module):

    def __init__(self, num_classes=1000):
        super(VGG_Cifar10, self).__init__()
        self.infl_ratio = 3
        self.features = nn.Sequential(BinarizeConv2d(3, 128 * self.
            infl_ratio, kernel_size=3, stride=1, padding=1, bias=True), nn.
            BatchNorm2d(128 * self.infl_ratio), nn.Hardtanh(inplace=True),
            BinarizeConv2d(128 * self.infl_ratio, 128 * self.infl_ratio,
            kernel_size=3, padding=1, bias=True), nn.MaxPool2d(kernel_size=
            2, stride=2), nn.BatchNorm2d(128 * self.infl_ratio), nn.
            Hardtanh(inplace=True), BinarizeConv2d(128 * self.infl_ratio, 
            256 * self.infl_ratio, kernel_size=3, padding=1, bias=True), nn
            .BatchNorm2d(256 * self.infl_ratio), nn.Hardtanh(inplace=True),
            BinarizeConv2d(256 * self.infl_ratio, 256 * self.infl_ratio,
            kernel_size=3, padding=1, bias=True), nn.MaxPool2d(kernel_size=
            2, stride=2), nn.BatchNorm2d(256 * self.infl_ratio), nn.
            Hardtanh(inplace=True), BinarizeConv2d(256 * self.infl_ratio, 
            512 * self.infl_ratio, kernel_size=3, padding=1, bias=True), nn
            .BatchNorm2d(512 * self.infl_ratio), nn.Hardtanh(inplace=True),
            BinarizeConv2d(512 * self.infl_ratio, 512, kernel_size=3,
            padding=1, bias=True), nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512), nn.Hardtanh(inplace=True))
        self.classifier = nn.Sequential(BinarizeLinear(512 * 4 * 4, 1024,
            bias=True), nn.BatchNorm1d(1024), nn.Hardtanh(inplace=True),
            BinarizeLinear(1024, 1024, bias=True), nn.BatchNorm1d(1024), nn
            .Hardtanh(inplace=True), BinarizeLinear(1024, num_classes, bias
            =True), nn.BatchNorm1d(num_classes, affine=False), nn.LogSoftmax())
        self.regime = {(0): {'optimizer': 'Adam', 'betas': (0.9, 0.999),
            'lr': 0.005}, (40): {'lr': 0.001}, (80): {'lr': 0.0005}, (100):
            {'lr': 0.0001}, (120): {'lr': 5e-05}, (140): {'lr': 1e-05}}

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.classifier(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_itayhubara_BinaryNet_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(BinarizeConv2d(*[], **{'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(BinarizeLinear(*[], **{'in_features': 4, 'out_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(HingeLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(VGG_Cifar10(*[], **{}), [torch.rand([4, 3, 64, 64])], {})
