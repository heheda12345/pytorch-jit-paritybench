import sys
_module = sys.modules[__name__]
del sys
datasets = _module
eval = _module
focal_loss = _module
resnet_features = _module
retinanet = _module
train_coco = _module
utilities = _module
images = _module
layers = _module

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


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch.autograd import Variable


import math


from torch import optim


from torch.optim import lr_scheduler


class FocalLoss(nn.Module):

    def __init__(self, focusing_param=2, balance_param=0.25):
        super(FocalLoss, self).__init__()
        self.focusing_param = focusing_param
        self.balance_param = balance_param

    def forward(self, output, target):
        cross_entropy = F.cross_entropy(output, target)
        cross_entropy_log = torch.log(cross_entropy)
        logpt = -F.cross_entropy(output, target)
        pt = torch.exp(logpt)
        focal_loss = -(1 - pt) ** self.focusing_param * logpt
        balanced_focal_loss = self.balance_param * focal_loss
        return balanced_focal_loss


def init_conv_weights(layer, weights_std=0.01, bias=0):
    """
    RetinaNet's layer initialization

    :layer
    :

    """
    nn.init.normal(layer.weight.data, std=weights_std)
    nn.init.constant(layer.bias.data, val=bias)
    return layer


def conv1x1(in_channels, out_channels, **kwargs):
    """Return a 1x1 convolutional layer with RetinaNet's weight and bias initialization"""
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, **kwargs)
    layer = init_conv_weights(layer)
    return layer


def conv3x3(in_channels, out_channels, **kwargs):
    """Return a 3x3 convolutional layer with RetinaNet's weight and bias initialization"""
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=3, **kwargs)
    layer = init_conv_weights(layer)
    return layer


class FeaturePyramid(nn.Module):

    def __init__(self, resnet):
        super(FeaturePyramid, self).__init__()
        self.resnet = resnet
        self.pyramid_transformation_3 = conv1x1(512, 256)
        self.pyramid_transformation_4 = conv1x1(1024, 256)
        self.pyramid_transformation_5 = conv1x1(2048, 256)
        self.pyramid_transformation_6 = conv3x3(2048, 256, padding=1, stride=2)
        self.pyramid_transformation_7 = conv3x3(256, 256, padding=1, stride=2)
        self.upsample_transform_1 = conv3x3(256, 256, padding=1)
        self.upsample_transform_2 = conv3x3(256, 256, padding=1)

    def _upsample(self, original_feature, scaled_feature, scale_factor=2):
        height, width = scaled_feature.size()[2:]
        return F.upsample(original_feature, scale_factor=scale_factor)[:, :,
            :height, :width]

    def forward(self, x):
        _, resnet_feature_3, resnet_feature_4, resnet_feature_5 = self.resnet(x
            )
        pyramid_feature_6 = self.pyramid_transformation_6(resnet_feature_5)
        pyramid_feature_7 = self.pyramid_transformation_7(F.relu(
            pyramid_feature_6))
        pyramid_feature_5 = self.pyramid_transformation_5(resnet_feature_5)
        pyramid_feature_4 = self.pyramid_transformation_4(resnet_feature_4)
        upsampled_feature_5 = self._upsample(pyramid_feature_5,
            pyramid_feature_4)
        pyramid_feature_4 = self.upsample_transform_1(torch.add(
            upsampled_feature_5, pyramid_feature_4))
        pyramid_feature_3 = self.pyramid_transformation_3(resnet_feature_3)
        upsampled_feature_4 = self._upsample(pyramid_feature_4,
            pyramid_feature_3)
        pyramid_feature_3 = self.upsample_transform_2(torch.add(
            upsampled_feature_4, pyramid_feature_3))
        return (pyramid_feature_3, pyramid_feature_4, pyramid_feature_5,
            pyramid_feature_6, pyramid_feature_7)


class SubNet(nn.Module):

    def __init__(self, mode, anchors=9, classes=80, depth=4,
        base_activation=F.relu, output_activation=F.sigmoid):
        super(SubNet, self).__init__()
        self.anchors = anchors
        self.classes = classes
        self.depth = depth
        self.base_activation = base_activation
        self.output_activation = output_activation
        self.subnet_base = nn.ModuleList([conv3x3(256, 256, padding=1) for
            _ in range(depth)])
        if mode == 'boxes':
            self.subnet_output = conv3x3(256, 4 * self.anchors, padding=1)
        elif mode == 'classes':
            self.subnet_output = conv3x3(256, (1 + self.classes) * self.
                anchors, padding=1)
        self._output_layer_init(self.subnet_output.bias.data)

    def _output_layer_init(self, tensor, pi=0.01):
        fill_constant = -math.log((1 - pi) / pi)
        if isinstance(tensor, Variable):
            self._output_layer_init(tensor.data)
        return tensor.fill_(fill_constant)

    def forward(self, x):
        for layer in self.subnet_base:
            x = self.base_activation(layer(x))
        x = self.subnet_output(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), x.size(2) *
            x.size(3) * self.anchors, -1)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_c0nn3r_RetinaNet(_paritybench_base):
    pass