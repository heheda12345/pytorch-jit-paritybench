import sys
_module = sys.modules[__name__]
del sys
master = _module
data = _module
base_data_loader = _module
base_dataset = _module
custom_dataset_data_loader = _module
data_loader = _module
dataset = _module
libs = _module
_ext = _module
bn = _module
build = _module
dense = _module
functions = _module
misc = _module
residual = _module
base_model = _module
dunet = _module
dunet_sybn = _module
loss = _module
models = _module
options = _module
base_options = _module
test_options = _module
train_options = _module
test = _module
train = _module
gradcheck = _module
html = _module
util = _module
visualizer = _module

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


from collections import OrderedDict


from collections import Iterable


from itertools import repeat


import torch


import torch.nn as nn


import torch.autograd as autograd


import numpy as np


from torch.nn import functional as F


import functools


import torch.nn.functional as F


from torch.autograd import Variable


class ABN(nn.Sequential):
    """Activated Batch Normalization

    This gathers a `BatchNorm2d` and an activation function in a single module
    """

    def __init__(self, num_features, activation=nn.ReLU(inplace=True), **kwargs
        ):
        """Creates an Activated Batch Normalization module

        Parameters
        ----------
        num_features : int
            Number of feature channels in the input and output.
        activation : nn.Module
            Module used as an activation function.
        kwargs
            All other arguments are forwarded to the `BatchNorm2d` constructor.
        """
        super(ABN, self).__init__(OrderedDict([('bn', nn.BatchNorm2d(
            num_features, **kwargs)), ('act', activation)]))


ACT_LEAKY_RELU = 'leaky_relu'


ACT_ELU = 'elu'


ACT_NONE = 'none'


def _check(fn, *args, **kwargs):
    success = fn(*args, **kwargs)
    if not success:
        raise RuntimeError('CUDA Error encountered in {}'.format(fn))


def _act_backward(ctx, x, dx):
    if ctx.activation == ACT_LEAKY_RELU:
        _check(_ext.leaky_relu_backward_cuda, x, dx, ctx.slope)
        _check(_ext.leaky_relu_cuda, x, 1.0 / ctx.slope)
    elif ctx.activation == ACT_ELU:
        _check(_ext.elu_backward_cuda, x, dx)
        _check(_ext.elu_inv_cuda, x)
    elif ctx.activation == ACT_NONE:
        pass


def _act_forward(ctx, x):
    if ctx.activation == ACT_LEAKY_RELU:
        _check(_ext.leaky_relu_cuda, x, ctx.slope)
    elif ctx.activation == ACT_ELU:
        _check(_ext.elu_cuda, x)
    elif ctx.activation == ACT_NONE:
        pass


def _check_contiguous(*args):
    if not all([(mod is None or mod.is_contiguous()) for mod in args]):
        raise ValueError('Non-contiguous input')


def _count_samples(x):
    count = 1
    for i, s in enumerate(x.size()):
        if i != 1:
            count *= s
    return count


class InPlaceABNWrapper(nn.Module):
    """Wrapper module to make `InPlaceABN` compatible with `ABN`"""

    def __init__(self, *args, **kwargs):
        super(InPlaceABNWrapper, self).__init__()
        self.bn = InPlaceABN(*args, **kwargs)

    def forward(self, input):
        return self.bn(input)


class InPlaceABNSyncWrapper(nn.Module):
    """Wrapper module to make `InPlaceABNSync` compatible with `ABN`"""

    def __init__(self, *args, **kwargs):
        super(InPlaceABNSyncWrapper, self).__init__()
        self.bn = InPlaceABNSync(*args, **kwargs)

    def forward(self, input):
        return self.bn(input)


class DenseModule(nn.Module):

    def __init__(self, in_channels, growth, layers, bottleneck_factor=4,
        norm_act=ABN, dilation=1):
        super(DenseModule, self).__init__()
        self.in_channels = in_channels
        self.growth = growth
        self.layers = layers
        self.convs1 = nn.ModuleList()
        self.convs3 = nn.ModuleList()
        for i in range(self.layers):
            self.convs1.append(nn.Sequential(OrderedDict([('bn', norm_act(
                in_channels)), ('conv', nn.Conv2d(in_channels, self.growth *
                bottleneck_factor, 1, bias=False))])))
            self.convs3.append(nn.Sequential(OrderedDict([('bn', norm_act(
                self.growth * bottleneck_factor)), ('conv', nn.Conv2d(self.
                growth * bottleneck_factor, self.growth, 3, padding=
                dilation, bias=False, dilation=dilation))])))
            in_channels += self.growth

    @property
    def out_channels(self):
        return self.in_channels + self.growth * self.layers

    def forward(self, x):
        inputs = [x]
        for i in range(self.layers):
            x = torch.cat(inputs, dim=1)
            x = self.convs1[i](x)
            x = self.convs3[i](x)
            inputs += [x]
        return torch.cat(inputs, dim=1)


class GlobalAvgPool2d(nn.Module):

    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        return inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)


class IdentityResidualBlock(nn.Module):

    def __init__(self, in_channels, channels, stride=1, dilation=1, groups=
        1, norm_act=ABN, dropout=None):
        """Configurable identity-mapping residual block

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        channels : list of int
            Number of channels in the internal feature maps. Can either have two or three elements: if three construct
            a residual block with two `3 x 3` convolutions, otherwise construct a bottleneck block with `1 x 1`, then
            `3 x 3` then `1 x 1` convolutions.
        stride : int
            Stride of the first `3 x 3` convolution
        dilation : int
            Dilation to apply to the `3 x 3` convolutions.
        groups : int
            Number of convolution groups. This is used to create ResNeXt-style blocks and is only compatible with
            bottleneck blocks.
        norm_act : callable
            Function to create normalization / activation Module.
        dropout: callable
            Function to create Dropout Module.
        """
        super(IdentityResidualBlock, self).__init__()
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError('channels must contain either two or three values'
                )
        if len(channels) == 2 and groups != 1:
            raise ValueError('groups > 1 are only valid if len(channels) == 3')
        is_bottleneck = len(channels) == 3
        need_proj_conv = stride != 1 or in_channels != channels[-1]
        self.bn1 = norm_act(in_channels)
        if not is_bottleneck:
            layers = [('conv1', nn.Conv2d(in_channels, channels[0], 3,
                stride=stride, padding=dilation, bias=False, dilation=
                dilation)), ('bn2', norm_act(channels[0])), ('conv2', nn.
                Conv2d(channels[0], channels[1], 3, stride=1, padding=
                dilation, bias=False, dilation=dilation))]
            if dropout is not None:
                layers = layers[0:2] + [('dropout', dropout())] + layers[2:]
        else:
            layers = [('conv1', nn.Conv2d(in_channels, channels[0], 1,
                stride=stride, padding=0, bias=False)), ('bn2', norm_act(
                channels[0])), ('conv2', nn.Conv2d(channels[0], channels[1],
                3, stride=1, padding=dilation, bias=False, groups=groups,
                dilation=dilation)), ('bn3', norm_act(channels[1])), (
                'conv3', nn.Conv2d(channels[1], channels[2], 1, stride=1,
                padding=0, bias=False))]
            if dropout is not None:
                layers = layers[0:4] + [('dropout', dropout())] + layers[4:]
        self.convs = nn.Sequential(OrderedDict(layers))
        if need_proj_conv:
            self.proj_conv = nn.Conv2d(in_channels, channels[-1], 1, stride
                =stride, padding=0, bias=False)

    def forward(self, x):
        if hasattr(self, 'proj_conv'):
            bn1 = self.bn1(x)
            shortcut = self.proj_conv(bn1)
        else:
            shortcut = x.clone()
            bn1 = self.bn1(x)
        out = self.convs(bn1)
        out.add_(shortcut)
        return out


class DUpsampling(nn.Module):

    def __init__(self, inplanes, scale, num_class=21, pad=0):
        super(DUpsampling, self).__init__()
        self.conv_w = nn.Conv2d(inplanes, num_class * scale * scale,
            kernel_size=1, padding=pad, bias=False)
        self.conv_p = nn.Conv2d(num_class * scale * scale, inplanes,
            kernel_size=1, padding=pad, bias=False)
        self.scale = scale

    def forward(self, x):
        x = self.conv_w(x)
        N, C, H, W = x.size()
        x_permuted = x.permute(0, 3, 2, 1)
        x_permuted = x_permuted.contiguous().view((N, W, H * self.scale,
            int(C / self.scale)))
        x_permuted = x_permuted.permute(0, 2, 1, 3)
        x_permuted = x_permuted.contiguous().view((N, W * self.scale, H *
            self.scale, int(C / (self.scale * self.scale))))
        x = x_permuted.permute(0, 3, 1, 2)
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=
        None, fist_dilation=1, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=dilation * multi_grid, dilation=dilation * multi_grid,
            bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
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
        out = out + residual
        out = self.relu_inplace(out)
        return out


affine_par = True


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 128
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
            ceil_mode=False)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
            dilation=2, multi_grid=(1, 1, 1))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
        multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)
            ] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=
            dilation, downsample=downsample, multi_grid=generate_multi_grid
            (0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                multi_grid=generate_multi_grid(i, multi_grid)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x_13 = x
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_46 = x
        x = self.layer4(x)
        x_13 = F.interpolate(x_13, [x_46.size()[2], x_46.size()[3]], mode=
            'bilinear', align_corners=True)
        x_low = torch.cat((x_13, x_46), dim=1)
        return x, x_low


def load_pretrained_model(net, state_dict, strict=True):
    """Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. If :attr:`strict` is ``True`` then
    the keys of :attr:`state_dict` must exactly match the keys returned
    by this module's :func:`state_dict()` function.

    Arguments:
        state_dict (dict): A dict containing parameters and
            persistent buffers.
        strict (bool): Strictly enforce that the keys in :attr:`state_dict`
            match the keys returned by this module's `:func:`state_dict()`
            function.
    """
    own_state = net.state_dict()
    for name, param in state_dict.items():
        if name in own_state:
            if isinstance(param, torch.nn.Parameter):
                param = param.data
            if strict:
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError(
                        'While copying the parameter named {}, whose dimensions in the model are {} and whose dimensions in the checkpoint are {}.'
                        .format(name, own_state[name].size(), param.size()))
            else:
                try:
                    own_state[name].copy_(param)
                except Exception:
                    print(
                        'Ignoring Error: While copying the parameter named {}, whose dimensions in the model are {} and whose dimensions in the checkpoint are {}.'
                        .format(name, own_state[name].size(), param.size()))
        elif strict:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))
    if strict:
        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))


class Encoder(nn.Module):

    def __init__(self, pretrain=False, model_path=' '):
        super(Encoder, self).__init__()
        self.model = ResNet(Bottleneck, [3, 4, 6, 3])
        if pretrain:
            load_pretrained_model(self.model, torch.load(model_path),
                strict=False)

    def forward(self, x):
        x, x_low = self.model(x)
        return x, x_low


class Decoder(nn.Module):

    def __init__(self, num_class, bn_momentum=0.1):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(1152, 48, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(2096, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout(0.1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=1)
        self.dupsample = DUpsampling(256, 16, num_class=21)
        self._init_weight()
        self.T = torch.nn.Parameter(torch.Tensor([1.0]))

    def forward(self, x, low_level_feature):
        low_level_feature = self.conv1(low_level_feature)
        low_level_feature = self.bn1(low_level_feature)
        low_level_feature = self.relu(low_level_feature)
        x_4_cat = torch.cat((x, low_level_feature), dim=1)
        x_4_cat = self.conv2(x_4_cat)
        x_4_cat = self.bn2(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        x_4_cat = self.dropout2(x_4_cat)
        x_4_cat = self.conv3(x_4_cat)
        x_4_cat = self.bn3(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        x_4_cat = self.dropout3(x_4_cat)
        x_4_cat = self.conv4(x_4_cat)
        out = self.dupsample(x_4_cat)
        out = out / self.T
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class DUNet(nn.Module):

    def __init__(self, encoder_pretrain=False, model_path=' ', num_class=21):
        super(DUNet, self).__init__()
        self.encoder = Encoder(pretrain=encoder_pretrain, model_path=model_path
            )
        self.decoder = Decoder(num_class)

    def forward(self, x):
        x, x_low = self.encoder(x)
        x = self.decoder(x, x_low)
        return x


class DUpsampling(nn.Module):

    def __init__(self, inplanes, scale, num_class=21, pad=0):
        super(DUpsampling, self).__init__()
        self.conv_w = nn.Conv2d(inplanes, num_class * scale * scale,
            kernel_size=1, padding=pad, bias=False)
        self.conv_p = nn.Conv2d(num_class * scale * scale, inplanes,
            kernel_size=1, padding=pad, bias=False)
        self.scale = scale

    def forward(self, x):
        x = self.conv_w(x)
        N, C, H, W = x.size()
        x_permuted = x.permute(0, 3, 2, 1)
        x_permuted = x_permuted.contiguous().view((N, W, H * self.scale,
            int(C / self.scale)))
        x_permuted = x_permuted.permute(0, 2, 1, 3)
        x_permuted = x_permuted.contiguous().view((N, W * self.scale, H *
            self.scale, int(C / (self.scale * self.scale))))
        x = x_permuted.permute(0, 3, 1, 2)
        return x


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
            bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
            dilation=2, multi_grid=(1, 1, 1))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1,
        multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes *
                block.expansion, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par))
        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)
            ] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=
            dilation, downsample=downsample, multi_grid=generate_multi_grid
            (0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                multi_grid=generate_multi_grid(i, multi_grid)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x_13 = x
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_46 = x
        x = self.layer4(x)
        x_13 = F.interpolate(x_13, [x_46.size()[2], x_46.size()[3]], mode=
            'bilinear', align_corners=True)
        x_low = torch.cat((x_13, x_46), dim=1)
        return x, x_low


class Encoder(nn.Module):

    def __init__(self, pretrain=False, model_path=' '):
        super(Encoder, self).__init__()
        self.model = ResNet(Bottleneck, [3, 4, 6, 3])
        if pretrain:
            load_pretrained_model(self.model, torch.load(model_path),
                strict=False)

    def forward(self, x):
        x, x_low = self.model(x)
        return x, x_low


class Decoder(nn.Module):

    def __init__(self, num_class, bn_momentum=0.1):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv2d(1088, 48, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(2096, 256, kernel_size=3, padding=1, bias=False)
        self.bn2 = BatchNorm2d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = BatchNorm2d(256)
        self.dropout3 = nn.Dropout(0.1)
        self.conv4 = nn.Conv2d(256, 64, kernel_size=1)
        self.dupsample = DUpsampling(64, 16, num_class=21)
        self._init_weight()
        self.T = torch.nn.Parameter(torch.Tensor([1.0]))

    def forward(self, x, low_level_feature):
        low_level_feature = self.conv1(low_level_feature)
        low_level_feature = self.bn1(low_level_feature)
        low_level_feature = self.relu(low_level_feature)
        x_4_cat = torch.cat((x, low_level_feature), dim=1)
        x_4_cat = self.conv2(x_4_cat)
        x_4_cat = self.bn2(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        x_4_cat = self.dropout2(x_4_cat)
        x_4_cat = self.conv3(x_4_cat)
        x_4_cat = self.bn3(x_4_cat)
        x_4_cat = self.relu(x_4_cat)
        x_4_cat = self.dropout3(x_4_cat)
        x_4_cat = self.conv4(x_4_cat)
        out = self.dupsample(x_4_cat)
        out = out / self.T
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)


class DUNet(nn.Module):

    def __init__(self, encoder_pretrain=False, model_path=' ', num_class=21):
        super(DUNet, self).__init__()
        self.encoder = Encoder(pretrain=encoder_pretrain, model_path=model_path
            )
        self.decoder = Decoder(num_class)

    def forward(self, x):
        x, x_low = self.encoder(x)
        x = self.decoder(x, x_low)
        return x


class CrossEntropyLoss2d(nn.Module):

    def __init__(self, weight=None, size_average=False, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


class FocalLoss2d(nn.Module):

    def __init__(self, gamma=2.0, weight=None, size_average=True,
        ignore_index=255):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss((1 - F.softmax(inputs)) ** self.gamma * F.
            log_softmax(inputs), targets)


class FocalLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - lpha (1-softmax(x)[class])^gamma \\log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num + 1))
        elif isinstance(alpha, Variable):
            self.alpha = alpha
        else:
            self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        P = F.softmax(inputs)
        b, c, h, w = inputs.size()
        class_mask = Variable(torch.zeros([b, c + 1, h, w]))
        class_mask.scatter_(1, targets.long(), 1.0)
        class_mask = class_mask[:, :-1, :, :]
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha
        alpha = self.alpha[targets.data.view(-1)].view_as(targets)
        probs = (P * class_mask).sum(1)
        log_p = probs.log()
        batch_loss = -alpha * torch.pow(1 - probs, self.gamma) * log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_LinZhuoChen_DUpsampling(_paritybench_base):
    pass
    def test_000(self):
        self._check(ABN(*[], **{'num_features': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(DenseModule(*[], **{'in_channels': 4, 'growth': 4, 'layers': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(GlobalAvgPool2d(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(IdentityResidualBlock(*[], **{'in_channels': 4, 'channels': [4, 4]}), [torch.rand([4, 4, 4, 4])], {})
