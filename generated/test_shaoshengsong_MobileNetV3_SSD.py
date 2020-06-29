import sys
_module = sys.modules[__name__]
del sys
convert_to_caffe2_models = _module
draw_eval_results = _module
eval_ssd = _module
extract_tf_weights = _module
open_images_downloader = _module
prune_alexnet = _module
run_ssd_example = _module
run_ssd_live_caffe2 = _module
run_ssd_live_demo = _module
train_ssd = _module
translate_tf_mobilenetv1 = _module
collation = _module
generate_vocdata = _module
open_images = _module
voc_dataset = _module
alexnet = _module
mobilenet = _module
mobilenet_v2 = _module
mobilenet_v3 = _module
multibox_loss = _module
scaled_l2_norm = _module
squeezenet = _module
vgg = _module
prunner = _module
mobilenetv1_ssd_config = _module
mobilenetv3_ssd_config = _module
squeezenet_ssd_config = _module
vgg_ssd_config = _module
data_preprocessing = _module
fpn_mobilenetv1_ssd = _module
fpn_ssd = _module
mobilenet_v2_ssd_lite = _module
mobilenet_v3_ssd_lite = _module
mobilenetv1_ssd = _module
mobilenetv1_ssd_lite = _module
predictor = _module
squeezenet_ssd_lite = _module
ssd = _module
vgg_ssd = _module
test_vgg_ssd = _module
transforms = _module
utils = _module
box_utils = _module
box_utils_numpy = _module
measurements = _module
misc = _module
model_book = _module
visual_tf_models = _module

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


import torch.optim as optim


from torch.optim import lr_scheduler


import logging


import torch.utils.model_zoo as model_zoo


import torch.nn.functional as F


import math


import torch.nn.init as init


from torch.nn import Conv2d


from torch.nn import Sequential


from torch.nn import ModuleList


from torch.nn import ReLU


import numpy as np


from typing import List


from typing import Tuple


from torch.nn import BatchNorm2d


from torch import nn


from collections import namedtuple


from collections import OrderedDict


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=11,
            stride=4, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(
            kernel_size=3, stride=2), nn.Conv2d(64, 192, kernel_size=5,
            padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3,
            stride=2), nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.
            ReLU(inplace=True), nn.Conv2d(384, 256, kernel_size=3, padding=
            1), nn.ReLU(inplace=True), nn.Conv2d(256, 256, kernel_size=3,
            padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3,
            stride=2))
        self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(256 * 6 * 6,
            4096), nn.ReLU(inplace=True), nn.Dropout(), nn.Linear(4096, 
            4096), nn.ReLU(inplace=True), nn.Linear(4096, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class MobileNetV1(nn.Module):

    def __init__(self, num_classes=1024):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=
                False), nn.BatchNorm2d(oup), nn.ReLU(inplace=True))

        def conv_dw(inp, oup, stride):
            return nn.Sequential(nn.Conv2d(inp, inp, 3, stride, 1, groups=
                inp, bias=False), nn.BatchNorm2d(inp), nn.ReLU(inplace=True
                ), nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d
                (oup), nn.ReLU(inplace=True))
        self.model = nn.Sequential(conv_bn(3, 32, 2), conv_dw(32, 64, 1),
            conv_dw(64, 128, 2), conv_dw(128, 128, 1), conv_dw(128, 256, 2),
            conv_dw(256, 256, 1), conv_dw(256, 512, 2), conv_dw(512, 512, 1
            ), conv_dw(512, 512, 1), conv_dw(512, 512, 1), conv_dw(512, 512,
            1), conv_dw(512, 512, 1), conv_dw(512, 1024, 2), conv_dw(1024, 
            1024, 1))
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio, use_batch_norm=True,
        onnx_compatible=False):
        super(InvertedResidual, self).__init__()
        ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        if expand_ratio == 1:
            if use_batch_norm:
                self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim,
                    3, stride, 1, groups=hidden_dim, bias=False), nn.
                    BatchNorm2d(hidden_dim), ReLU(inplace=True), nn.Conv2d(
                    hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup))
            else:
                self.conv = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim,
                    3, stride, 1, groups=hidden_dim, bias=False), ReLU(
                    inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias
                    =False))
        elif use_batch_norm:
            self.conv = nn.Sequential(nn.Conv2d(inp, hidden_dim, 1, 1, 0,
                bias=False), nn.BatchNorm2d(hidden_dim), ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=
                hidden_dim, bias=False), nn.BatchNorm2d(hidden_dim), ReLU(
                inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=
                False), nn.BatchNorm2d(oup))
        else:
            self.conv = nn.Sequential(nn.Conv2d(inp, hidden_dim, 1, 1, 0,
                bias=False), ReLU(inplace=True), nn.Conv2d(hidden_dim,
                hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                ReLU(inplace=True), nn.Conv2d(hidden_dim, oup, 1, 1, 0,
                bias=False))

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_1x1_bn(inp, oup, use_batch_norm=True, onnx_compatible=False):
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    if use_batch_norm:
        return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.
            BatchNorm2d(oup), ReLU(inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), ReLU
            (inplace=True))


def conv_bn(inp, oup, stride, use_batch_norm=True, onnx_compatible=False):
    ReLU = nn.ReLU if onnx_compatible else nn.ReLU6
    if use_batch_norm:
        return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup), ReLU(inplace=True))
    else:
        return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            ReLU(inplace=True))


class MobileNetV2(nn.Module):

    def __init__(self, n_class=1000, input_size=224, width_mult=1.0,
        dropout_ratio=0.2, use_batch_norm=True, onnx_compatible=False):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 
            32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 
            320, 1, 1]]
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult
            ) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2, onnx_compatible=
            onnx_compatible)]
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel,
                        output_channel, s, expand_ratio=t, use_batch_norm=
                        use_batch_norm, onnx_compatible=onnx_compatible))
                else:
                    self.features.append(block(input_channel,
                        output_channel, 1, expand_ratio=t, use_batch_norm=
                        use_batch_norm, onnx_compatible=onnx_compatible))
                input_channel = output_channel
        self.features.append(conv_1x1_bn(input_channel, self.last_channel,
            use_batch_norm=use_batch_norm, onnx_compatible=onnx_compatible))
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(nn.Dropout(dropout_ratio), nn.
            Linear(self.last_channel, n_class))
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class h_sigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class h_swish(nn.Module):

    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3.0, self.inplace) / 6.0
        return out * x


class SqueezeBlock(nn.Module):

    def __init__(self, exp_size, divide=4):
        super(SqueezeBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dense = nn.Sequential(nn.Linear(exp_size, exp_size // divide),
            nn.ReLU(inplace=True), nn.Linear(exp_size // divide, exp_size),
            h_sigmoid())

    def forward(self, x):
        batch, channels, height, width = x.size()
        out = self.avg_pool(x).view(batch, channels)
        out = self.dense(out)
        out = out.view(batch, channels, 1, 1)
        return out * x


class MobileBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernal_size, stride,
        nonLinear, SE, exp_size):
        super(MobileBlock, self).__init__()
        self.out_channels = out_channels
        self.nonLinear = nonLinear
        self.SE = SE
        padding = (kernal_size - 1) // 2
        self.use_connect = stride == 1 and in_channels == out_channels
        if self.nonLinear == 'RE':
            activation = nn.ReLU
        else:
            activation = h_swish
        self.conv = nn.Sequential(nn.Conv2d(in_channels, exp_size,
            kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d
            (exp_size), activation(inplace=True))
        self.depth_conv = nn.Sequential(nn.Conv2d(exp_size, exp_size,
            kernel_size=kernal_size, stride=stride, padding=padding, groups
            =exp_size), nn.BatchNorm2d(exp_size))
        if self.SE:
            self.squeeze_block = SqueezeBlock(exp_size)
        self.point_conv = nn.Sequential(nn.Conv2d(exp_size, out_channels,
            kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(
            out_channels), activation(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        out = self.depth_conv(out)
        if self.SE:
            out = self.squeeze_block(out)
        out = self.point_conv(out)
        if self.use_connect:
            return x + out
        else:
            return out


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


class MobileNetV3(nn.Module):

    def __init__(self, model_mode='SMALL', num_classes=30, multiplier=1.0,
        dropout_rate=0.0):
        super(MobileNetV3, self).__init__()
        self.num_classes = num_classes
        if model_mode == 'LARGE':
            layers = [[16, 16, 3, 1, 'RE', False, 16], [16, 24, 3, 2, 'RE',
                False, 64], [24, 24, 3, 1, 'RE', False, 72], [24, 40, 5, 2,
                'RE', True, 72], [40, 40, 5, 1, 'RE', True, 120], [40, 40, 
                5, 1, 'RE', True, 120], [40, 80, 3, 2, 'HS', False, 240], [
                80, 80, 3, 1, 'HS', False, 200], [80, 80, 3, 1, 'HS', False,
                184], [80, 80, 3, 1, 'HS', False, 184], [80, 112, 3, 1,
                'HS', True, 480], [112, 112, 3, 1, 'HS', True, 672], [112, 
                160, 5, 1, 'HS', True, 672], [160, 160, 5, 2, 'HS', True, 
                672], [160, 160, 5, 1, 'HS', True, 960]]
            init_conv_out = _make_divisible(16 * multiplier)
            self.init_conv = nn.Sequential(nn.Conv2d(in_channels=3,
                out_channels=init_conv_out, kernel_size=3, stride=2,
                padding=1), nn.BatchNorm2d(init_conv_out), h_swish(inplace=
                True))
            self.block = []
            for in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size in layers:
                in_channels = _make_divisible(in_channels * multiplier)
                out_channels = _make_divisible(out_channels * multiplier)
                exp_size = _make_divisible(exp_size * multiplier)
                self.block.append(MobileBlock(in_channels, out_channels,
                    kernal_size, stride, nonlinear, se, exp_size))
            self.block = nn.Sequential(*self.block)
            out_conv1_in = _make_divisible(160 * multiplier)
            out_conv1_out = _make_divisible(960 * multiplier)
            self.out_conv1 = nn.Sequential(nn.Conv2d(out_conv1_in,
                out_conv1_out, kernel_size=1, stride=1), nn.BatchNorm2d(
                out_conv1_out), h_swish(inplace=True))
            out_conv2_in = _make_divisible(960 * multiplier)
            out_conv2_out = _make_divisible(1280 * multiplier)
            self.out_conv2 = nn.Sequential(nn.Conv2d(out_conv2_in,
                out_conv2_out, kernel_size=1, stride=1), h_swish(inplace=
                True), nn.Dropout(dropout_rate), nn.Conv2d(out_conv2_out,
                self.num_classes, kernel_size=1, stride=1))
        elif model_mode == 'SMALL':
            layers = [[16, 16, 3, 2, 'RE', True, 16], [16, 24, 3, 2, 'RE', 
                False, 72], [24, 24, 3, 1, 'RE', False, 88], [24, 40, 5, 2,
                'RE', True, 96], [40, 40, 5, 1, 'RE', True, 240], [40, 40, 
                5, 1, 'RE', True, 240], [40, 48, 5, 1, 'HS', True, 120], [
                48, 48, 5, 1, 'HS', True, 144], [48, 96, 5, 2, 'HS', True, 
                288], [96, 96, 5, 1, 'HS', True, 576], [96, 96, 5, 1, 'HS',
                True, 576]]
            self.features = []
            init_conv_out = _make_divisible(16 * multiplier)
            self.init_conv = nn.Sequential(nn.Conv2d(in_channels=3,
                out_channels=init_conv_out, kernel_size=3, stride=2,
                padding=1), nn.BatchNorm2d(init_conv_out), h_swish(inplace=
                True))
            self.features.append(nn.Conv2d(in_channels=3, out_channels=
                init_conv_out, kernel_size=3, stride=2, padding=1))
            self.features.append(nn.BatchNorm2d(init_conv_out))
            self.features.append(h_swish(inplace=True))
            self.block = []
            for in_channels, out_channels, kernal_size, stride, nonlinear, se, exp_size in layers:
                in_channels = _make_divisible(in_channels * multiplier)
                out_channels = _make_divisible(out_channels * multiplier)
                exp_size = _make_divisible(exp_size * multiplier)
                self.block.append(MobileBlock(in_channels, out_channels,
                    kernal_size, stride, nonlinear, se, exp_size))
                self.features.append(MobileBlock(in_channels, out_channels,
                    kernal_size, stride, nonlinear, se, exp_size))
            self.block = nn.Sequential(*self.block)
            out_conv1_in = _make_divisible(96 * multiplier)
            out_conv1_out = _make_divisible(576 * multiplier)
            self.out_conv1 = nn.Sequential(nn.Conv2d(out_conv1_in,
                out_conv1_out, kernel_size=1, stride=1), SqueezeBlock(
                out_conv1_out), nn.BatchNorm2d(out_conv1_out), h_swish(
                inplace=True))
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.features.append(nn.Conv2d(out_conv1_in, out_conv1_out,
                kernel_size=1, stride=1))
            self.features.append(SqueezeBlock(out_conv1_out))
            self.features.append(nn.BatchNorm2d(out_conv1_out))
            self.features.append(h_swish(inplace=True))
            out_conv2_in = _make_divisible(576 * multiplier)
            out_conv2_out = _make_divisible(1280 * multiplier)
            self.out_conv2 = nn.Sequential(nn.Conv2d(out_conv2_in,
                out_conv2_out, kernel_size=1, stride=1), h_swish(inplace=
                True), nn.Dropout(dropout_rate), nn.Conv2d(out_conv2_out,
                self.num_classes, kernel_size=1, stride=1))
            self.features.append(nn.Conv2d(out_conv2_in, out_conv2_out,
                kernel_size=1, stride=1))
            self.features.append(h_swish(inplace=True))
            self.features = nn.Sequential(*self.features)
        self.apply(_weights_init)

    def forward(self, x):
        out = self.init_conv(x)
        out = self.block(out)
        out = self.out_conv1(out)
        batch, channels, height, width = out.size()
        out = self.avg_pool(out)
        out = self.out_conv2(out).view(batch, -1)
        return out


class MultiboxLoss(nn.Module):

    def __init__(self, priors, iou_threshold, neg_pos_ratio,
        center_variance, size_variance, device):
        """Implement SSD Multibox Loss.

        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiboxLoss, self).__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.priors = priors
        self.priors

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            boxes (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            loss = -F.log_softmax(confidence, dim=2)[:, :, (0)]
            mask = box_utils.hard_negative_mining(loss, labels, self.
                neg_pos_ratio)
        confidence = confidence[(mask), :]
        classification_loss = F.cross_entropy(confidence.reshape(-1,
            num_classes), labels[mask], size_average=False)
        pos_mask = labels > 0
        predicted_locations = predicted_locations[(pos_mask), :].reshape(-1, 4)
        gt_locations = gt_locations[(pos_mask), :].reshape(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations,
            size_average=False)
        num_pos = gt_locations.size(0)
        return smooth_l1_loss / num_pos, classification_loss / num_pos


class ScaledL2Norm(nn.Module):

    def __init__(self, in_channels, initial_scale):
        super(ScaledL2Norm, self).__init__()
        self.in_channels = in_channels
        self.scale = nn.Parameter(torch.Tensor(in_channels))
        self.initial_scale = initial_scale
        self.reset_parameters()

    def forward(self, x):
        return F.normalize(x, p=2, dim=1) * self.scale.unsqueeze(0).unsqueeze(2
            ).unsqueeze(3)

    def reset_parameters(self):
        self.scale.data.fill_(self.initial_scale)


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes, expand1x1_planes,
        expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
            kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
            kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=1000):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError(
                'Unsupported SqueezeNet version {version}:1.0 or 1.1 expected'
                .format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(nn.Conv2d(3, 96, kernel_size=7,
                stride=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=
                3, stride=2, ceil_mode=True), Fire(96, 16, 64, 64), Fire(
                128, 16, 64, 64), Fire(128, 32, 128, 128), nn.MaxPool2d(
                kernel_size=3, stride=2, ceil_mode=True), Fire(256, 32, 128,
                128), Fire(256, 48, 192, 192), Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256), nn.MaxPool2d(kernel_size=3, stride
                =2, ceil_mode=True), Fire(512, 64, 256, 256))
        else:
            self.features = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3,
                stride=2), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=
                3, stride=2), Fire(64, 16, 64, 64), Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2), Fire(128, 32, 128, 
                128), Fire(256, 32, 128, 128), nn.MaxPool2d(kernel_size=3,
                stride=2), Fire(256, 48, 192, 192), Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256), Fire(512, 64, 256, 256))
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(nn.Dropout(p=0.5), final_conv, nn.
            ReLU(inplace=True), nn.AvgPool2d(13, stride=1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)


class FPNSSD(nn.Module):

    def __init__(self, num_classes: int, base_net: nn.ModuleList,
        source_layer_indexes: List[int], extras: nn.ModuleList,
        classification_headers: nn.ModuleList, regression_headers: nn.
        ModuleList, upsample_mode='nearest'):
        """Compose a SSD model using the given components.
        """
        super(FPNSSD, self).__init__()
        self.num_classes = num_classes
        self.base_net = base_net
        self.source_layer_indexes = source_layer_indexes
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.upsample_mode = upsample_mode
        self.source_layer_add_ons = nn.ModuleList([t[1] for t in
            source_layer_indexes if isinstance(t, tuple)])
        self.upsamplers = [nn.Upsample(size=(19, 19), mode='bilinear'), nn.
            Upsample(size=(10, 10), mode='bilinear'), nn.Upsample(size=(5, 
            5), mode='bilinear'), nn.Upsample(size=(3, 3), mode='bilinear'),
            nn.Upsample(size=(2, 2), mode='bilinear')]

    def forward(self, x: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        features = []
        for end_layer_index in self.source_layer_indexes:
            if isinstance(end_layer_index, tuple):
                added_layer = end_layer_index[1]
                end_layer_index = end_layer_index[0]
            else:
                added_layer = None
            for layer in self.base_net[start_layer_index:end_layer_index]:
                x = layer(x)
            start_layer_index = end_layer_index
            if added_layer:
                y = added_layer(x)
            else:
                y = x
            features.append(y)
            header_index += 1
        for layer in self.base_net[end_layer_index:]:
            x = layer(x)
        for layer in self.extras:
            x = layer(x)
            features.append(x)
            header_index += 1
        upstream_feature = None
        for i in range(len(features) - 1, -1, -1):
            feature = features[i]
            if upstream_feature is not None:
                upstream_feature = self.upsamplers[i](upstream_feature)
                upstream_feature += feature
            else:
                upstream_feature = feature
            confidence, location = self.compute_header(i, upstream_feature)
            confidences.append(confidence)
            locations.append(location)
        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        return confidences, locations

    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)
        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)
        return confidence, location

    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda
            storage, loc: storage), strict=False)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage,
            loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


GraphPath = namedtuple('GraphPath', ['s0', 'name'])


class SSD(nn.Module):

    def __init__(self, num_classes: int, base_net: nn.ModuleList,
        source_layer_indexes: List[int], extras: nn.ModuleList,
        classification_headers: nn.ModuleList, regression_headers: nn.
        ModuleList, is_test=False, config=None, device=None):
        """Compose a SSD model using the given components.
        """
        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.base_net = base_net
        self.source_layer_indexes = source_layer_indexes
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.is_test = is_test
        self.config = config
        self.source_layer_add_ons = nn.ModuleList([t[1] for t in
            source_layer_indexes if isinstance(t, tuple) and not isinstance
            (t, GraphPath)])
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available(
                ) else 'cpu')
        if is_test:
            self.config = config
            self.priors = config.priors

    def forward(self, x: torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        for index_0, end_layer_index in enumerate(self.source_layer_indexes):
            if isinstance(end_layer_index, GraphPath):
                path = end_layer_index
                end_layer_index = end_layer_index.s0
                added_layer = None
            else:
                added_layer = None
                path = None
            for index_1, layer in enumerate(self.base_net[start_layer_index
                :end_layer_index]):
                x = layer(x)
            if added_layer:
                y = added_layer(x)
            else:
                y = x
            if path:
                sub = getattr(self.base_net[end_layer_index], path.name)
                for index_2, layer in enumerate(sub):
                    y = layer(y)
            start_layer_index = end_layer_index
            confidence, location = self.compute_header(header_index, y)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)
        for layer in self.base_net[end_layer_index:]:
            x = layer(x)
        for layer in self.extras:
            x = layer(x)
            confidence, location = self.compute_header(header_index, x)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)
        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)
        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(locations, self.
                priors, self.config.center_variance, self.config.size_variance)
            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes
        else:
            return confidences, locations

    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)
        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)
        return confidence, location

    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda
            storage, loc: storage), strict=True)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc:
            storage)
        state_dict = {k: v for k, v in state_dict.items() if not (k.
            startswith('classification_headers') or k.startswith(
            'regression_headers'))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage,
            loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_shaoshengsong_MobileNetV3_SSD(_paritybench_base):
    pass
    def test_000(self):
        self._check(Fire(*[], **{'inplanes': 4, 'squeeze_planes': 4, 'expand1x1_planes': 4, 'expand3x3_planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(InvertedResidual(*[], **{'inp': 4, 'oup': 4, 'stride': 1, 'expand_ratio': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(MobileBlock(*[], **{'in_channels': 4, 'out_channels': 4, 'kernal_size': 4, 'stride': 1, 'nonLinear': 4, 'SE': 4, 'exp_size': 4}), [torch.rand([4, 4, 2, 2])], {})

    @_fails_compile()
    def test_003(self):
        self._check(MobileNetV3(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_004(self):
        self._check(ScaledL2Norm(*[], **{'in_channels': 4, 'initial_scale': 1.0}), [torch.rand([4, 4, 4, 4])], {})

    def test_005(self):
        self._check(SqueezeBlock(*[], **{'exp_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(h_sigmoid(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_007(self):
        self._check(h_swish(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

