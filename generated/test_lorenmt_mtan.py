import sys
_module = sys.modules[__name__]
del sys
create_dataset = _module
aspp = _module
resnet = _module
resnet_dilated = _module
resnet_mtan = _module
model_segnet_cross = _module
model_segnet_dense = _module
model_segnet_mtan = _module
model_segnet_single = _module
model_segnet_split = _module
model_segnet_stan = _module
coco_results = _module
model_wrn_eval = _module
model_wrn_mtan = _module

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


import torch.optim as optim


import torch.utils.data.sampler as sampler


from torch.autograd import Variable


import torch.nn.init as init


import numpy as np


class DeepLabHead(nn.Sequential):

    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(
            256), nn.ReLU(), nn.Conv2d(256, num_classes, 1))


class ASPPConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, dilation):
        modules = [nn.Conv2d(in_channels, out_channels, 3, padding=dilation,
            dilation=dilation, bias=False), nn.BatchNorm2d(out_channels),
            nn.ReLU()]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(nn.AdaptiveAvgPool2d(1), nn.
            Conv2d(in_channels, out_channels, 1, bias=False), nn.
            BatchNorm2d(out_channels), nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False
            )


class ASPP(nn.Module):

    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(nn.Conv2d(in_channels, out_channels, 1,
            bias=False), nn.BatchNorm2d(out_channels), nn.ReLU()))
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))
        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(nn.Conv2d(5 * out_channels,
            out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.
            ReLU(), nn.Dropout(0.5))

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=
        1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                'Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
        bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=
        1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=
        False, groups=1, width_per_group=64, replace_stride_with_dilation=
        None, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                'replace_stride_with_dilation should be None or a 3-element tuple, got {}'
                .format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
            padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
            dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
            dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
            dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes *
                block.expansion, stride), norm_layer(planes * block.expansion))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self
            .groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                base_width=self.base_width, dilation=self.dilation,
                norm_layer=norm_layer))
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
        return x


parser = argparse.ArgumentParser(description=
    'Multi-task: Attention Network on WRN')


opt = parser.parse_args()


class SegNet(nn.Module):

    def __init__(self):
        super(SegNet, self).__init__()
        filter = [64, 128, 256, 512, 512]
        self.class_nb = 13
        self.encoder_block = nn.ModuleList([self.conv_layer([3, filter[0],
            filter[0]], bottle_neck=True)])
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0],
            filter[0], self.class_nb], bottle_neck=True)])
        self.encoder_block_t = nn.ModuleList([nn.ModuleList([self.
            conv_layer([3, filter[0], filter[0]], bottle_neck=True)])])
        self.decoder_block_t = nn.ModuleList([nn.ModuleList([self.
            conv_layer([2 * filter[0], 2 * filter[0], filter[0]],
            bottle_neck=True)])])
        for i in range(4):
            if i == 0:
                self.encoder_block.append(self.conv_layer([filter[i],
                    filter[i + 1], filter[i + 1]], bottle_neck=True))
                self.decoder_block.append(self.conv_layer([filter[i + 1],
                    filter[i], filter[i]], bottle_neck=True))
            else:
                self.encoder_block.append(self.conv_layer([filter[i],
                    filter[i + 1], filter[i + 1]], bottle_neck=False))
                self.decoder_block.append(self.conv_layer([filter[i + 1],
                    filter[i], filter[i]], bottle_neck=False))
        for j in range(3):
            if j < 2:
                self.encoder_block_t.append(nn.ModuleList([self.conv_layer(
                    [3, filter[0], filter[0]], bottle_neck=True)]))
                self.decoder_block_t.append(nn.ModuleList([self.conv_layer(
                    [2 * filter[0], 2 * filter[0], filter[0]], bottle_neck=
                    True)]))
            for i in range(4):
                if i == 0:
                    self.encoder_block_t[j].append(self.conv_layer([2 *
                        filter[i], filter[i + 1], filter[i + 1]],
                        bottle_neck=True))
                    self.decoder_block_t[j].append(self.conv_layer([2 *
                        filter[i + 1], filter[i], filter[i]], bottle_neck=True)
                        )
                else:
                    self.encoder_block_t[j].append(self.conv_layer([2 *
                        filter[i], filter[i + 1], filter[i + 1]],
                        bottle_neck=False))
                    self.decoder_block_t[j].append(self.conv_layer([2 *
                        filter[i + 1], filter[i], filter[i]], bottle_neck=
                        False))
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2,
            return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.pred_task1 = self.conv_layer([filter[0], self.class_nb],
            bottle_neck=True, pred_layer=True)
        self.pred_task2 = self.conv_layer([filter[0], 1], bottle_neck=True,
            pred_layer=True)
        self.pred_task3 = self.conv_layer([filter[0], 3], bottle_neck=True,
            pred_layer=True)
        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5, -0.5]))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def conv_layer(self, channel, bottle_neck, pred_layer=False):
        if bottle_neck:
            if not pred_layer:
                conv_block = nn.Sequential(nn.Conv2d(in_channels=channel[0],
                    out_channels=channel[1], kernel_size=3, padding=1), nn.
                    BatchNorm2d(channel[1]), nn.ReLU(inplace=True), nn.
                    Conv2d(in_channels=channel[1], out_channels=channel[2],
                    kernel_size=3, padding=1), nn.BatchNorm2d(channel[2]),
                    nn.ReLU(inplace=True))
            else:
                conv_block = nn.Sequential(nn.Conv2d(in_channels=channel[0],
                    out_channels=channel[0], kernel_size=3, padding=1), nn.
                    Conv2d(in_channels=channel[0], out_channels=channel[1],
                    kernel_size=1, padding=0))
        else:
            conv_block = nn.Sequential(nn.Conv2d(in_channels=channel[0],
                out_channels=channel[1], kernel_size=3, padding=1), nn.
                BatchNorm2d(channel[1]), nn.ReLU(inplace=True), nn.Conv2d(
                in_channels=channel[1], out_channels=channel[1],
                kernel_size=3, padding=1), nn.BatchNorm2d(channel[1]), nn.
                ReLU(inplace=True), nn.Conv2d(in_channels=channel[1],
                out_channels=channel[2], kernel_size=3, padding=1), nn.
                BatchNorm2d(channel[2]), nn.ReLU(inplace=True))
        return conv_block

    def forward(self, x):
        encoder_conv, decoder_conv, encoder_samp, decoder_samp, indices = (
            [0] * 5 for _ in range(5))
        (encoder_conv_t, decoder_conv_t, encoder_samp_t, decoder_samp_t,
            indices_t) = ([0] * 3 for _ in range(5))
        for i in range(3):
            encoder_conv_t[i], decoder_conv_t[i], encoder_samp_t[i
                ], decoder_samp_t[i], indices_t[i] = ([0] * 5 for _ in range(5)
                )
        for i in range(5):
            if i == 0:
                encoder_conv[i] = self.encoder_block[i](x)
                encoder_samp[i], indices[i] = self.down_sampling(encoder_conv
                    [i])
            else:
                encoder_conv[i] = self.encoder_block[i](encoder_samp[i - 1])
                encoder_samp[i], indices[i] = self.down_sampling(encoder_conv
                    [i])
        for i in range(5):
            if i == 0:
                decoder_samp[i] = self.up_sampling(encoder_samp[-1],
                    indices[-1])
                decoder_conv[i] = self.decoder_block[-i - 1](decoder_samp[i])
            else:
                decoder_samp[i] = self.up_sampling(decoder_conv[i - 1],
                    indices[-i - 1])
                decoder_conv[i] = self.decoder_block[-i - 1](decoder_samp[i])
        for j in range(3):
            for i in range(5):
                if i == 0:
                    encoder_conv_t[j][i] = self.encoder_block_t[j][i](x)
                    encoder_samp_t[j][i], indices_t[j][i] = self.down_sampling(
                        encoder_conv_t[j][i])
                else:
                    encoder_conv_t[j][i] = self.encoder_block_t[j][i](torch
                        .cat((encoder_samp_t[j][i - 1], encoder_samp[i - 1]
                        ), dim=1))
                    encoder_samp_t[j][i], indices_t[j][i] = self.down_sampling(
                        encoder_conv_t[j][i])
            for i in range(5):
                if i == 0:
                    decoder_samp_t[j][i] = self.up_sampling(encoder_samp_t[
                        j][-1], indices_t[j][-1])
                    decoder_conv_t[j][i] = self.decoder_block_t[j][-i - 1](
                        torch.cat((decoder_samp_t[j][i], decoder_samp[i]),
                        dim=1))
                else:
                    decoder_samp_t[j][i] = self.up_sampling(decoder_conv_t[
                        j][i - 1], indices_t[j][-i - 1])
                    decoder_conv_t[j][i] = self.decoder_block_t[j][-i - 1](
                        torch.cat((decoder_samp_t[j][i], decoder_samp[i]),
                        dim=1))
        t1_pred = F.log_softmax(self.pred_task1(decoder_conv_t[0][-1]), dim=1)
        t2_pred = self.pred_task2(decoder_conv_t[1][-1])
        t3_pred = self.pred_task3(decoder_conv_t[2][-1])
        t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)
        return [t1_pred, t2_pred, t3_pred], self.logsigma

    def model_fit(self, x_pred1, x_output1, x_pred2, x_output2, x_pred3,
        x_output3):
        binary_mask = (torch.sum(x_output2, dim=1) != 0).type(torch.FloatTensor
            ).unsqueeze(1).to(device)
        loss1 = F.nll_loss(x_pred1, x_output1, ignore_index=-1)
        loss2 = torch.sum(torch.abs(x_pred2 - x_output2) * binary_mask
            ) / torch.nonzero(binary_mask).size(0)
        loss3 = 1 - torch.sum(x_pred3 * x_output3 * binary_mask
            ) / torch.nonzero(binary_mask).size(0)
        return [loss1, loss2, loss3]

    def compute_miou(self, x_pred, x_output):
        _, x_pred_label = torch.max(x_pred, dim=1)
        x_output_label = x_output
        batch_size = x_pred.size(0)
        for i in range(batch_size):
            true_class = 0
            first_switch = True
            for j in range(self.class_nb):
                pred_mask = torch.eq(x_pred_label[i], j * torch.ones(
                    x_pred_label[i].shape).type(torch.LongTensor).to(device))
                true_mask = torch.eq(x_output_label[i], j * torch.ones(
                    x_output_label[i].shape).type(torch.LongTensor).to(device))
                mask_comb = pred_mask.type(torch.FloatTensor) + true_mask.type(
                    torch.FloatTensor)
                union = torch.sum((mask_comb > 0).type(torch.FloatTensor))
                intsec = torch.sum((mask_comb > 1).type(torch.FloatTensor))
                if union == 0:
                    continue
                if first_switch:
                    class_prob = intsec / union
                    first_switch = False
                else:
                    class_prob = intsec / union + class_prob
                true_class += 1
            if i == 0:
                batch_avg = class_prob / true_class
            else:
                batch_avg = class_prob / true_class + batch_avg
        return batch_avg / batch_size

    def compute_iou(self, x_pred, x_output):
        _, x_pred_label = torch.max(x_pred, dim=1)
        x_output_label = x_output
        batch_size = x_pred.size(0)
        for i in range(batch_size):
            if i == 0:
                pixel_acc = torch.div(torch.sum(torch.eq(x_pred_label[i],
                    x_output_label[i]).type(torch.FloatTensor)), torch.sum(
                    (x_output_label[i] >= 0).type(torch.FloatTensor)))
            else:
                pixel_acc = pixel_acc + torch.div(torch.sum(torch.eq(
                    x_pred_label[i], x_output_label[i]).type(torch.
                    FloatTensor)), torch.sum((x_output_label[i] >= 0).type(
                    torch.FloatTensor)))
        return pixel_acc / batch_size

    def depth_error(self, x_pred, x_output):
        binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
        x_pred_true = x_pred.masked_select(binary_mask)
        x_output_true = x_output.masked_select(binary_mask)
        abs_err = torch.abs(x_pred_true - x_output_true)
        rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
        return torch.sum(abs_err) / torch.nonzero(binary_mask).size(0
            ), torch.sum(rel_err) / torch.nonzero(binary_mask).size(0)

    def normal_error(self, x_pred, x_output):
        binary_mask = torch.sum(x_output, dim=1) != 0
        error = torch.acos(torch.clamp(torch.sum(x_pred * x_output, 1).
            masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
        error = np.degrees(error)
        return np.mean(error), np.median(error), np.mean(error < 11.25
            ), np.mean(error < 22.5), np.mean(error < 30)


class SegNet(nn.Module):

    def __init__(self):
        super(SegNet, self).__init__()
        filter = [64, 128, 256, 512, 512]
        self.class_nb = 13
        self.encoder_block = nn.ModuleList([self.conv_layer([3, filter[0]])])
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0],
            filter[0]])])
        for i in range(4):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i +
                1]]))
            self.decoder_block.append(self.conv_layer([filter[i + 1],
                filter[i]]))
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0],
            filter[0]])])
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filter[0],
            filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(self.conv_layer([filter[i + 1],
                    filter[i + 1]]))
                self.conv_block_dec.append(self.conv_layer([filter[i],
                    filter[i]]))
            else:
                self.conv_block_enc.append(nn.Sequential(self.conv_layer([
                    filter[i + 1], filter[i + 1]]), self.conv_layer([filter
                    [i + 1], filter[i + 1]])))
                self.conv_block_dec.append(nn.Sequential(self.conv_layer([
                    filter[i], filter[i]]), self.conv_layer([filter[i],
                    filter[i]])))
        self.encoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([
            filter[0], filter[0], filter[0]])])])
        self.decoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([2 *
            filter[0], filter[0], filter[0]])])])
        self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0],
            filter[1]])])
        self.decoder_block_att = nn.ModuleList([self.conv_layer([filter[0],
            filter[0]])])
        for j in range(3):
            if j < 2:
                self.encoder_att.append(nn.ModuleList([self.att_layer([
                    filter[0], filter[0], filter[0]])]))
                self.decoder_att.append(nn.ModuleList([self.att_layer([2 *
                    filter[0], filter[0], filter[0]])]))
            for i in range(4):
                self.encoder_att[j].append(self.att_layer([2 * filter[i + 1
                    ], filter[i + 1], filter[i + 1]]))
                self.decoder_att[j].append(self.att_layer([filter[i + 1] +
                    filter[i], filter[i], filter[i]]))
        for i in range(4):
            if i < 3:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1
                    ], filter[i + 2]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1
                    ], filter[i]]))
            else:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1
                    ], filter[i + 1]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1
                    ], filter[i + 1]]))
        self.pred_task1 = self.conv_layer([filter[0], self.class_nb], pred=True
            )
        self.pred_task2 = self.conv_layer([filter[0], 1], pred=True)
        self.pred_task3 = self.conv_layer([filter[0], 3], pred=True)
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2,
            return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5, -0.5]))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def conv_layer(self, channel, pred=False):
        if not pred:
            conv_block = nn.Sequential(nn.Conv2d(in_channels=channel[0],
                out_channels=channel[1], kernel_size=3, padding=1), nn.
                BatchNorm2d(num_features=channel[1]), nn.ReLU(inplace=True))
        else:
            conv_block = nn.Sequential(nn.Conv2d(in_channels=channel[0],
                out_channels=channel[0], kernel_size=3, padding=1), nn.
                Conv2d(in_channels=channel[0], out_channels=channel[1],
                kernel_size=1, padding=0))
        return conv_block

    def att_layer(self, channel):
        att_block = nn.Sequential(nn.Conv2d(in_channels=channel[0],
            out_channels=channel[1], kernel_size=1, padding=0), nn.
            BatchNorm2d(channel[1]), nn.ReLU(inplace=True), nn.Conv2d(
            in_channels=channel[1], out_channels=channel[2], kernel_size=1,
            padding=0), nn.BatchNorm2d(channel[2]), nn.Sigmoid())
        return att_block

    def forward(self, x):
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = ([0] * 5 for
            _ in range(5))
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))
        atten_encoder, atten_decoder = ([0] * 3 for _ in range(2))
        for i in range(3):
            atten_encoder[i], atten_decoder[i] = ([0] * 5 for _ in range(2))
        for i in range(3):
            for j in range(5):
                atten_encoder[i][j], atten_decoder[i][j] = ([0] * 3 for _ in
                    range(2))
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
        for i in range(5):
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1],
                    indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
        for i in range(3):
            for j in range(5):
                if j == 0:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](g_encoder
                        [j][0])
                    atten_encoder[i][j][1] = atten_encoder[i][j][0
                        ] * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](
                        atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][
                        j][2], kernel_size=2, stride=2)
                else:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](torch.
                        cat((g_encoder[j][0], atten_encoder[i][j - 1][2]),
                        dim=1))
                    atten_encoder[i][j][1] = atten_encoder[i][j][0
                        ] * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](
                        atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][
                        j][2], kernel_size=2, stride=2)
            for j in range(5):
                if j == 0:
                    atten_decoder[i][j][0] = F.interpolate(atten_encoder[i]
                        [-1][-1], scale_factor=2, mode='bilinear',
                        align_corners=True)
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](
                        atten_decoder[i][j][0])
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](torch
                        .cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = atten_decoder[i][j][1
                        ] * g_decoder[j][-1]
                else:
                    atten_decoder[i][j][0] = F.interpolate(atten_decoder[i]
                        [j - 1][2], scale_factor=2, mode='bilinear',
                        align_corners=True)
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](
                        atten_decoder[i][j][0])
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](torch
                        .cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = atten_decoder[i][j][1
                        ] * g_decoder[j][-1]
        t1_pred = F.log_softmax(self.pred_task1(atten_decoder[0][-1][-1]),
            dim=1)
        t2_pred = self.pred_task2(atten_decoder[1][-1][-1])
        t3_pred = self.pred_task3(atten_decoder[2][-1][-1])
        t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)
        return [t1_pred, t2_pred, t3_pred], self.logsigma

    def model_fit(self, x_pred1, x_output1, x_pred2, x_output2, x_pred3,
        x_output3):
        binary_mask = (torch.sum(x_output2, dim=1) != 0).type(torch.FloatTensor
            ).unsqueeze(1).to(device)
        loss1 = F.nll_loss(x_pred1, x_output1, ignore_index=-1)
        loss2 = torch.sum(torch.abs(x_pred2 - x_output2) * binary_mask
            ) / torch.nonzero(binary_mask).size(0)
        loss3 = 1 - torch.sum(x_pred3 * x_output3 * binary_mask
            ) / torch.nonzero(binary_mask).size(0)
        return [loss1, loss2, loss3]

    def compute_miou(self, x_pred, x_output):
        _, x_pred_label = torch.max(x_pred, dim=1)
        x_output_label = x_output
        batch_size = x_pred.size(0)
        for i in range(batch_size):
            true_class = 0
            first_switch = True
            for j in range(self.class_nb):
                pred_mask = torch.eq(x_pred_label[i], j * torch.ones(
                    x_pred_label[i].shape).type(torch.LongTensor).to(device))
                true_mask = torch.eq(x_output_label[i], j * torch.ones(
                    x_output_label[i].shape).type(torch.LongTensor).to(device))
                mask_comb = pred_mask.type(torch.FloatTensor) + true_mask.type(
                    torch.FloatTensor)
                union = torch.sum((mask_comb > 0).type(torch.FloatTensor))
                intsec = torch.sum((mask_comb > 1).type(torch.FloatTensor))
                if union == 0:
                    continue
                if first_switch:
                    class_prob = intsec / union
                    first_switch = False
                else:
                    class_prob = intsec / union + class_prob
                true_class += 1
            if i == 0:
                batch_avg = class_prob / true_class
            else:
                batch_avg = class_prob / true_class + batch_avg
        return batch_avg / batch_size

    def compute_iou(self, x_pred, x_output):
        _, x_pred_label = torch.max(x_pred, dim=1)
        x_output_label = x_output
        batch_size = x_pred.size(0)
        for i in range(batch_size):
            if i == 0:
                pixel_acc = torch.div(torch.sum(torch.eq(x_pred_label[i],
                    x_output_label[i]).type(torch.FloatTensor)), torch.sum(
                    (x_output_label[i] >= 0).type(torch.FloatTensor)))
            else:
                pixel_acc = pixel_acc + torch.div(torch.sum(torch.eq(
                    x_pred_label[i], x_output_label[i]).type(torch.
                    FloatTensor)), torch.sum((x_output_label[i] >= 0).type(
                    torch.FloatTensor)))
        return pixel_acc / batch_size

    def depth_error(self, x_pred, x_output):
        binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
        x_pred_true = x_pred.masked_select(binary_mask)
        x_output_true = x_output.masked_select(binary_mask)
        abs_err = torch.abs(x_pred_true - x_output_true)
        rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
        return torch.sum(abs_err) / torch.nonzero(binary_mask).size(0
            ), torch.sum(rel_err) / torch.nonzero(binary_mask).size(0)

    def normal_error(self, x_pred, x_output):
        binary_mask = torch.sum(x_output, dim=1) != 0
        error = torch.acos(torch.clamp(torch.sum(x_pred * x_output, 1).
            masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
        error = np.degrees(error)
        return np.mean(error), np.median(error), np.mean(error < 11.25
            ), np.mean(error < 22.5), np.mean(error < 30)


class SegNet(nn.Module):

    def __init__(self):
        super(SegNet, self).__init__()
        filter = [64, 128, 256, 512, 512]
        self.class_nb = 13
        self.encoder_block = nn.ModuleList([self.conv_layer([3, filter[0]])])
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0],
            filter[0]])])
        for i in range(4):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i +
                1]]))
            self.decoder_block.append(self.conv_layer([filter[i + 1],
                filter[i]]))
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0],
            filter[0]])])
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filter[0],
            filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(self.conv_layer([filter[i + 1],
                    filter[i + 1]]))
                self.conv_block_dec.append(self.conv_layer([filter[i],
                    filter[i]]))
            else:
                self.conv_block_enc.append(nn.Sequential(self.conv_layer([
                    filter[i + 1], filter[i + 1]]), self.conv_layer([filter
                    [i + 1], filter[i + 1]])))
                self.conv_block_dec.append(nn.Sequential(self.conv_layer([
                    filter[i], filter[i]]), self.conv_layer([filter[i],
                    filter[i]])))
        if opt.task == 'semantic':
            self.pred_task = self.conv_layer([filter[0], self.class_nb],
                pred=True)
        if opt.task == 'depth':
            self.pred_task = self.conv_layer([filter[0], 1], pred=True)
        if opt.task == 'normal':
            self.pred_task = self.conv_layer([filter[0], 3], pred=True)
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2,
            return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def conv_layer(self, channel, pred=False):
        if not pred:
            conv_block = nn.Sequential(nn.Conv2d(in_channels=channel[0],
                out_channels=channel[1], kernel_size=3, padding=1), nn.
                BatchNorm2d(num_features=channel[1]), nn.ReLU(inplace=True))
        else:
            conv_block = nn.Sequential(nn.Conv2d(in_channels=channel[0],
                out_channels=channel[0], kernel_size=3, padding=1), nn.
                Conv2d(in_channels=channel[0], out_channels=channel[1],
                kernel_size=1, padding=0))
        return conv_block

    def forward(self, x):
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = ([0] * 5 for
            _ in range(5))
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
        for i in range(5):
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1],
                    indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
        if opt.task == 'semantic':
            pred = F.log_softmax(self.pred_task(g_decoder[-1][-1]), dim=1)
        if opt.task == 'depth':
            pred = self.pred_task(g_decoder[-1][-1])
        if opt.task == 'normal':
            pred = self.pred_task(g_decoder[-1][-1])
            pred = pred / torch.norm(pred, p=2, dim=1, keepdim=True)
        return pred

    def model_fit(self, x_pred, x_output):
        if opt.task == 'semantic':
            loss = F.nll_loss(x_pred, x_output, ignore_index=-1)
        if opt.task == 'depth':
            binary_mask = (torch.sum(x_output, dim=1) != 0).type(torch.
                FloatTensor).unsqueeze(1).to(device)
            loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask
                ) / torch.nonzero(binary_mask).size(0)
        if opt.task == 'normal':
            binary_mask = (torch.sum(x_output, dim=1) != 0).type(torch.
                FloatTensor).unsqueeze(1).to(device)
            loss = 1 - torch.sum(x_pred * x_output * binary_mask
                ) / torch.nonzero(binary_mask).size(0)
        return loss

    def compute_miou(self, x_pred, x_output):
        _, x_pred_label = torch.max(x_pred, dim=1)
        x_output_label = x_output
        batch_size = x_pred.size(0)
        for i in range(batch_size):
            true_class = 0
            first_switch = True
            for j in range(self.class_nb):
                pred_mask = torch.eq(x_pred_label[i], j * torch.ones(
                    x_pred_label[i].shape).type(torch.LongTensor).to(device))
                true_mask = torch.eq(x_output_label[i], j * torch.ones(
                    x_output_label[i].shape).type(torch.LongTensor).to(device))
                mask_comb = pred_mask.type(torch.FloatTensor) + true_mask.type(
                    torch.FloatTensor)
                union = torch.sum((mask_comb > 0).type(torch.FloatTensor))
                intsec = torch.sum((mask_comb > 1).type(torch.FloatTensor))
                if union == 0:
                    continue
                if first_switch:
                    class_prob = intsec / union
                    first_switch = False
                else:
                    class_prob = intsec / union + class_prob
                true_class += 1
            if i == 0:
                batch_avg = class_prob / true_class
            else:
                batch_avg = class_prob / true_class + batch_avg
        return batch_avg / batch_size

    def compute_iou(self, x_pred, x_output):
        _, x_pred_label = torch.max(x_pred, dim=1)
        x_output_label = x_output
        batch_size = x_pred.size(0)
        for i in range(batch_size):
            if i == 0:
                pixel_acc = torch.div(torch.sum(torch.eq(x_pred_label[i],
                    x_output_label[i]).type(torch.FloatTensor)), torch.sum(
                    (x_output_label[i] >= 0).type(torch.FloatTensor)))
            else:
                pixel_acc = pixel_acc + torch.div(torch.sum(torch.eq(
                    x_pred_label[i], x_output_label[i]).type(torch.
                    FloatTensor)), torch.sum((x_output_label[i] >= 0).type(
                    torch.FloatTensor)))
        return pixel_acc / batch_size

    def depth_error(self, x_pred, x_output):
        binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
        x_pred_true = x_pred.masked_select(binary_mask)
        x_output_true = x_output.masked_select(binary_mask)
        abs_err = torch.abs(x_pred_true - x_output_true)
        rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
        return torch.sum(abs_err) / torch.nonzero(binary_mask).size(0
            ), torch.sum(rel_err) / torch.nonzero(binary_mask).size(0)

    def normal_error(self, x_pred, x_output):
        binary_mask = torch.sum(x_output, dim=1) != 0
        error = torch.acos(torch.clamp(torch.sum(x_pred * x_output, 1).
            masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
        error = np.degrees(error)
        return np.mean(error), np.median(error), np.mean(error < 11.25
            ), np.mean(error < 22.5), np.mean(error < 30)


class SegNet(nn.Module):

    def __init__(self):
        super(SegNet, self).__init__()
        if opt.type == 'wide':
            filter = [64, 128, 256, 512, 1024]
        else:
            filter = [64, 128, 256, 512, 512]
        self.class_nb = 13
        self.encoder_block = nn.ModuleList([self.conv_layer([3, filter[0]])])
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0],
            filter[0]])])
        for i in range(4):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i +
                1]]))
            self.decoder_block.append(self.conv_layer([filter[i + 1],
                filter[i]]))
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0],
            filter[0]])])
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filter[0],
            filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(self.conv_layer([filter[i + 1],
                    filter[i + 1]]))
                self.conv_block_dec.append(self.conv_layer([filter[i],
                    filter[i]]))
            else:
                self.conv_block_enc.append(nn.Sequential(self.conv_layer([
                    filter[i + 1], filter[i + 1]]), self.conv_layer([filter
                    [i + 1], filter[i + 1]])))
                self.conv_block_dec.append(nn.Sequential(self.conv_layer([
                    filter[i], filter[i]]), self.conv_layer([filter[i],
                    filter[i]])))
        self.pred_task1 = nn.Sequential(nn.Conv2d(in_channels=filter[0],
            out_channels=filter[0], kernel_size=3, padding=1), nn.Conv2d(
            in_channels=filter[0], out_channels=self.class_nb, kernel_size=
            1, padding=0))
        self.pred_task2 = nn.Sequential(nn.Conv2d(in_channels=filter[0],
            out_channels=filter[0], kernel_size=3, padding=1), nn.Conv2d(
            in_channels=filter[0], out_channels=1, kernel_size=1, padding=0))
        self.pred_task3 = nn.Sequential(nn.Conv2d(in_channels=filter[0],
            out_channels=filter[0], kernel_size=3, padding=1), nn.Conv2d(
            in_channels=filter[0], out_channels=3, kernel_size=1, padding=0))
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2,
            return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.logsigma = nn.Parameter(torch.FloatTensor([-0.5, -0.5, -0.5]))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def conv_layer(self, channel):
        if opt.type == 'deep':
            conv_block = nn.Sequential(nn.Conv2d(in_channels=channel[0],
                out_channels=channel[1], kernel_size=3, padding=1), nn.
                BatchNorm2d(num_features=channel[1]), nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=channel[1], out_channels=channel[1],
                kernel_size=3, padding=1), nn.BatchNorm2d(num_features=
                channel[1]), nn.ReLU(inplace=True))
        else:
            conv_block = nn.Sequential(nn.Conv2d(in_channels=channel[0],
                out_channels=channel[1], kernel_size=3, padding=1), nn.
                BatchNorm2d(num_features=channel[1]), nn.ReLU(inplace=True))
        return conv_block

    def forward(self, x):
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = ([0] * 5 for
            _ in range(5))
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
        for i in range(5):
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1],
                    indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
        t1_pred = F.log_softmax(self.pred_task1(g_decoder[i][1]), dim=1)
        t2_pred = self.pred_task2(g_decoder[i][1])
        t3_pred = self.pred_task3(g_decoder[i][1])
        t3_pred = t3_pred / torch.norm(t3_pred, p=2, dim=1, keepdim=True)
        return [t1_pred, t2_pred, t3_pred], self.logsigma

    def model_fit(self, x_pred1, x_output1, x_pred2, x_output2, x_pred3,
        x_output3):
        binary_mask = (torch.sum(x_output2, dim=1) != 0).type(torch.FloatTensor
            ).unsqueeze(1).to(device)
        loss1 = F.nll_loss(x_pred1, x_output1, ignore_index=-1)
        loss2 = torch.sum(torch.abs(x_pred2 - x_output2) * binary_mask
            ) / torch.nonzero(binary_mask).size(0)
        loss3 = 1 - torch.sum(x_pred3 * x_output3 * binary_mask
            ) / torch.nonzero(binary_mask).size(0)
        return [loss1, loss2, loss3]

    def compute_miou(self, x_pred, x_output):
        _, x_pred_label = torch.max(x_pred, dim=1)
        x_output_label = x_output
        batch_size = x_pred.size(0)
        for i in range(batch_size):
            true_class = 0
            first_switch = True
            for j in range(self.class_nb):
                pred_mask = torch.eq(x_pred_label[i], j * torch.ones(
                    x_pred_label[i].shape).type(torch.LongTensor).to(device))
                true_mask = torch.eq(x_output_label[i], j * torch.ones(
                    x_output_label[i].shape).type(torch.LongTensor).to(device))
                mask_comb = pred_mask.type(torch.FloatTensor) + true_mask.type(
                    torch.FloatTensor)
                union = torch.sum((mask_comb > 0).type(torch.FloatTensor))
                intsec = torch.sum((mask_comb > 1).type(torch.FloatTensor))
                if union == 0:
                    continue
                if first_switch:
                    class_prob = intsec / union
                    first_switch = False
                else:
                    class_prob = intsec / union + class_prob
                true_class += 1
            if i == 0:
                batch_avg = class_prob / true_class
            else:
                batch_avg = class_prob / true_class + batch_avg
        return batch_avg / batch_size

    def compute_iou(self, x_pred, x_output):
        _, x_pred_label = torch.max(x_pred, dim=1)
        x_output_label = x_output
        batch_size = x_pred.size(0)
        for i in range(batch_size):
            if i == 0:
                pixel_acc = torch.div(torch.sum(torch.eq(x_pred_label[i],
                    x_output_label[i]).type(torch.FloatTensor)), torch.sum(
                    (x_output_label[i] >= 0).type(torch.FloatTensor)))
            else:
                pixel_acc = pixel_acc + torch.div(torch.sum(torch.eq(
                    x_pred_label[i], x_output_label[i]).type(torch.
                    FloatTensor)), torch.sum((x_output_label[i] >= 0).type(
                    torch.FloatTensor)))
        return pixel_acc / batch_size

    def depth_error(self, x_pred, x_output):
        binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
        x_pred_true = x_pred.masked_select(binary_mask)
        x_output_true = x_output.masked_select(binary_mask)
        abs_err = torch.abs(x_pred_true - x_output_true)
        rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
        return torch.sum(abs_err) / torch.nonzero(binary_mask).size(0
            ), torch.sum(rel_err) / torch.nonzero(binary_mask).size(0)

    def normal_error(self, x_pred, x_output):
        binary_mask = torch.sum(x_output, dim=1) != 0
        error = torch.acos(torch.clamp(torch.sum(x_pred * x_output, 1).
            masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
        error = np.degrees(error)
        return np.mean(error), np.median(error), np.mean(error < 11.25
            ), np.mean(error < 22.5), np.mean(error < 30)


class SegNet(nn.Module):

    def __init__(self):
        super(SegNet, self).__init__()
        filter = [64, 128, 256, 512, 512]
        self.class_nb = 13
        self.encoder_block = nn.ModuleList([self.conv_layer([3, filter[0]])])
        self.decoder_block = nn.ModuleList([self.conv_layer([filter[0],
            filter[0]])])
        for i in range(4):
            self.encoder_block.append(self.conv_layer([filter[i], filter[i +
                1]]))
            self.decoder_block.append(self.conv_layer([filter[i + 1],
                filter[i]]))
        self.conv_block_enc = nn.ModuleList([self.conv_layer([filter[0],
            filter[0]])])
        self.conv_block_dec = nn.ModuleList([self.conv_layer([filter[0],
            filter[0]])])
        for i in range(4):
            if i == 0:
                self.conv_block_enc.append(self.conv_layer([filter[i + 1],
                    filter[i + 1]]))
                self.conv_block_dec.append(self.conv_layer([filter[i],
                    filter[i]]))
            else:
                self.conv_block_enc.append(nn.Sequential(self.conv_layer([
                    filter[i + 1], filter[i + 1]]), self.conv_layer([filter
                    [i + 1], filter[i + 1]])))
                self.conv_block_dec.append(nn.Sequential(self.conv_layer([
                    filter[i], filter[i]]), self.conv_layer([filter[i],
                    filter[i]])))
        self.encoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([
            filter[0], filter[0], filter[0]])])])
        self.decoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([2 *
            filter[0], filter[0], filter[0]])])])
        self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0],
            filter[1]])])
        self.decoder_block_att = nn.ModuleList([self.conv_layer([filter[0],
            filter[0]])])
        for j in range(1):
            for i in range(4):
                self.encoder_att[j].append(self.att_layer([2 * filter[i + 1
                    ], filter[i + 1], filter[i + 1]]))
                self.decoder_att[j].append(self.att_layer([filter[i + 1] +
                    filter[i], filter[i], filter[i]]))
        for i in range(4):
            if i < 3:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1
                    ], filter[i + 2]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1
                    ], filter[i]]))
            else:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1
                    ], filter[i + 1]]))
                self.decoder_block_att.append(self.conv_layer([filter[i + 1
                    ], filter[i + 1]]))
        if opt.task == 'semantic':
            self.pred_task = self.conv_layer([filter[0], self.class_nb],
                pred=True)
        if opt.task == 'depth':
            self.pred_task = self.conv_layer([filter[0], 1], pred=True)
        if opt.task == 'normal':
            self.pred_task = self.conv_layer([filter[0], 3], pred=True)
        self.down_sampling = nn.MaxPool2d(kernel_size=2, stride=2,
            return_indices=True)
        self.up_sampling = nn.MaxUnpool2d(kernel_size=2, stride=2)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def conv_layer(self, channel, pred=False):
        if not pred:
            conv_block = nn.Sequential(nn.Conv2d(in_channels=channel[0],
                out_channels=channel[1], kernel_size=3, padding=1), nn.
                BatchNorm2d(num_features=channel[1]), nn.ReLU(inplace=True))
        else:
            conv_block = nn.Sequential(nn.Conv2d(in_channels=channel[0],
                out_channels=channel[0], kernel_size=3, padding=1), nn.
                Conv2d(in_channels=channel[0], out_channels=channel[1],
                kernel_size=1, padding=0))
        return conv_block

    def att_layer(self, channel):
        att_block = nn.Sequential(nn.Conv2d(in_channels=channel[0],
            out_channels=channel[1], kernel_size=1, padding=0), nn.
            BatchNorm2d(channel[1]), nn.ReLU(inplace=True), nn.Conv2d(
            in_channels=channel[1], out_channels=channel[2], kernel_size=1,
            padding=0), nn.BatchNorm2d(channel[2]), nn.Sigmoid())
        return att_block

    def forward(self, x):
        g_encoder, g_decoder, g_maxpool, g_upsampl, indices = ([0] * 5 for
            _ in range(5))
        for i in range(5):
            g_encoder[i], g_decoder[-i - 1] = ([0] * 2 for _ in range(2))
        atten_encoder, atten_decoder = ([0] * 3 for _ in range(2))
        for i in range(3):
            atten_encoder[i], atten_decoder[i] = ([0] * 5 for _ in range(2))
        for i in range(3):
            for j in range(5):
                atten_encoder[i][j], atten_decoder[i][j] = ([0] * 3 for _ in
                    range(2))
        for i in range(5):
            if i == 0:
                g_encoder[i][0] = self.encoder_block[i](x)
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.conv_block_enc[i](g_encoder[i][0])
                g_maxpool[i], indices[i] = self.down_sampling(g_encoder[i][1])
        for i in range(5):
            if i == 0:
                g_upsampl[i] = self.up_sampling(g_maxpool[-1], indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
            else:
                g_upsampl[i] = self.up_sampling(g_decoder[i - 1][-1],
                    indices[-i - 1])
                g_decoder[i][0] = self.decoder_block[-i - 1](g_upsampl[i])
                g_decoder[i][1] = self.conv_block_dec[-i - 1](g_decoder[i][0])
        for i in range(1):
            for j in range(5):
                if j == 0:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](g_encoder
                        [j][0])
                    atten_encoder[i][j][1] = atten_encoder[i][j][0
                        ] * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](
                        atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][
                        j][2], kernel_size=2, stride=2)
                else:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](torch.
                        cat((g_encoder[j][0], atten_encoder[i][j - 1][2]),
                        dim=1))
                    atten_encoder[i][j][1] = atten_encoder[i][j][0
                        ] * g_encoder[j][1]
                    atten_encoder[i][j][2] = self.encoder_block_att[j](
                        atten_encoder[i][j][1])
                    atten_encoder[i][j][2] = F.max_pool2d(atten_encoder[i][
                        j][2], kernel_size=2, stride=2)
            for j in range(5):
                if j == 0:
                    atten_decoder[i][j][0] = F.interpolate(atten_encoder[i]
                        [-1][-1], scale_factor=2, mode='bilinear',
                        align_corners=True)
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](
                        atten_decoder[i][j][0])
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](torch
                        .cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = atten_decoder[i][j][1
                        ] * g_decoder[j][-1]
                else:
                    atten_decoder[i][j][0] = F.interpolate(atten_decoder[i]
                        [j - 1][2], scale_factor=2, mode='bilinear',
                        align_corners=True)
                    atten_decoder[i][j][0] = self.decoder_block_att[-j - 1](
                        atten_decoder[i][j][0])
                    atten_decoder[i][j][1] = self.decoder_att[i][-j - 1](torch
                        .cat((g_upsampl[j], atten_decoder[i][j][0]), dim=1))
                    atten_decoder[i][j][2] = atten_decoder[i][j][1
                        ] * g_decoder[j][-1]
        if opt.task == 'semantic':
            pred = F.log_softmax(self.pred_task(atten_decoder[0][-1][-1]),
                dim=1)
        if opt.task == 'depth':
            pred = self.pred_task(atten_decoder[0][-1][-1])
        if opt.task == 'normal':
            pred = self.pred_task(atten_decoder[0][-1][-1])
            pred = pred / torch.norm(pred, p=2, dim=1, keepdim=True)
        return pred

    def model_fit(self, x_pred, x_output):
        if opt.task == 'semantic':
            loss = F.nll_loss(x_pred, x_output, ignore_index=-1)
        if opt.task == 'depth':
            binary_mask = (torch.sum(x_output, dim=1) != 0).type(torch.
                FloatTensor).unsqueeze(1).to(device)
            loss = torch.sum(torch.abs(x_pred - x_output) * binary_mask
                ) / torch.nonzero(binary_mask).size(0)
        if opt.task == 'normal':
            binary_mask = (torch.sum(x_output, dim=1) != 0).type(torch.
                FloatTensor).unsqueeze(1).to(device)
            loss = 1 - torch.sum(x_pred * x_output * binary_mask
                ) / torch.nonzero(binary_mask).size(0)
        return loss

    def compute_miou(self, x_pred, x_output):
        _, x_pred_label = torch.max(x_pred, dim=1)
        x_output_label = x_output
        batch_size = x_pred.size(0)
        for i in range(batch_size):
            true_class = 0
            first_switch = True
            for j in range(self.class_nb):
                pred_mask = torch.eq(x_pred_label[i], j * torch.ones(
                    x_pred_label[i].shape).type(torch.LongTensor).to(device))
                true_mask = torch.eq(x_output_label[i], j * torch.ones(
                    x_output_label[i].shape).type(torch.LongTensor).to(device))
                mask_comb = pred_mask.type(torch.FloatTensor) + true_mask.type(
                    torch.FloatTensor)
                union = torch.sum((mask_comb > 0).type(torch.FloatTensor))
                intsec = torch.sum((mask_comb > 1).type(torch.FloatTensor))
                if union == 0:
                    continue
                if first_switch:
                    class_prob = intsec / union
                    first_switch = False
                else:
                    class_prob = intsec / union + class_prob
                true_class += 1
            if i == 0:
                batch_avg = class_prob / true_class
            else:
                batch_avg = class_prob / true_class + batch_avg
        return batch_avg / batch_size

    def compute_iou(self, x_pred, x_output):
        _, x_pred_label = torch.max(x_pred, dim=1)
        x_output_label = x_output
        batch_size = x_pred.size(0)
        for i in range(batch_size):
            if i == 0:
                pixel_acc = torch.div(torch.sum(torch.eq(x_pred_label[i],
                    x_output_label[i]).type(torch.FloatTensor)), torch.sum(
                    (x_output_label[i] >= 0).type(torch.FloatTensor)))
            else:
                pixel_acc = pixel_acc + torch.div(torch.sum(torch.eq(
                    x_pred_label[i], x_output_label[i]).type(torch.
                    FloatTensor)), torch.sum((x_output_label[i] >= 0).type(
                    torch.FloatTensor)))
        return pixel_acc / batch_size

    def depth_error(self, x_pred, x_output):
        binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1).to(device)
        x_pred_true = x_pred.masked_select(binary_mask)
        x_output_true = x_output.masked_select(binary_mask)
        abs_err = torch.abs(x_pred_true - x_output_true)
        rel_err = torch.abs(x_pred_true - x_output_true) / x_output_true
        return torch.sum(abs_err) / torch.nonzero(binary_mask).size(0
            ), torch.sum(rel_err) / torch.nonzero(binary_mask).size(0)

    def normal_error(self, x_pred, x_output):
        binary_mask = torch.sum(x_output, dim=1) != 0
        error = torch.acos(torch.clamp(torch.sum(x_pred * x_output, 1).
            masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
        error = np.degrees(error)
        return np.mean(error), np.median(error), np.mean(error < 11.25
            ), np.mean(error < 22.5), np.mean(error < 30)


class wide_basic(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1,
            bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes,
                kernel_size=1, stride=stride, bias=True))

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):

    def __init__(self, depth, widen_factor, num_classes):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        n = int((depth - 4) / 6)
        k = widen_factor
        filter = [16, 16 * k, 32 * k, 64 * k]
        self.conv1 = conv3x3(3, filter[0], stride=1)
        self.layer1 = self._wide_layer(wide_basic, filter[1], n, stride=2)
        self.layer2 = self._wide_layer(wide_basic, filter[2], n, stride=2)
        self.layer3 = self._wide_layer(wide_basic, filter[3], n, stride=2)
        self.bn1 = nn.BatchNorm2d(filter[3], momentum=0.9)
        self.linear = nn.ModuleList([nn.Sequential(nn.Linear(filter[3],
            num_classes[0]), nn.Softmax(dim=1))])
        self.encoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([
            filter[0], filter[0], filter[0]])])])
        self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0],
            filter[1]])])
        for j in range(10):
            if j < 9:
                self.encoder_att.append(nn.ModuleList([self.att_layer([
                    filter[0], filter[0], filter[0]])]))
                self.linear.append(nn.Sequential(nn.Linear(filter[3],
                    num_classes[j + 1]), nn.Softmax(dim=1)))
            for i in range(3):
                self.encoder_att[j].append(self.att_layer([2 * filter[i + 1
                    ], filter[i + 1], filter[i + 1]]))
        for i in range(3):
            if i < 2:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1
                    ], filter[i + 2]]))
            else:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1
                    ], filter[i + 1]]))

    def conv_layer(self, channel):
        conv_block = nn.Sequential(nn.Conv2d(in_channels=channel[0],
            out_channels=channel[1], kernel_size=3, padding=1), nn.
            BatchNorm2d(num_features=channel[1]), nn.ReLU(inplace=True))
        return conv_block

    def att_layer(self, channel):
        att_block = nn.Sequential(nn.Conv2d(in_channels=channel[0],
            out_channels=channel[1], kernel_size=1, padding=0), nn.
            BatchNorm2d(channel[1]), nn.ReLU(inplace=True), nn.Conv2d(
            in_channels=channel[1], out_channels=channel[2], kernel_size=1,
            padding=0), nn.BatchNorm2d(channel[2]), nn.Sigmoid())
        return att_block

    def _wide_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, k):
        g_encoder = [0] * 4
        atten_encoder = [0] * 10
        for i in range(10):
            atten_encoder[i] = [0] * 4
        for i in range(10):
            for j in range(4):
                atten_encoder[i][j] = [0] * 3
        g_encoder[0] = self.conv1(x)
        g_encoder[1] = self.layer1(g_encoder[0])
        g_encoder[2] = self.layer2(g_encoder[1])
        g_encoder[3] = F.relu(self.bn1(self.layer3(g_encoder[2])))
        for j in range(4):
            if j == 0:
                atten_encoder[k][j][0] = self.encoder_att[k][j](g_encoder[0])
                atten_encoder[k][j][1] = atten_encoder[k][j][0] * g_encoder[0]
                atten_encoder[k][j][2] = self.encoder_block_att[j](
                    atten_encoder[k][j][1])
                atten_encoder[k][j][2] = F.max_pool2d(atten_encoder[k][j][2
                    ], kernel_size=2, stride=2)
            else:
                atten_encoder[k][j][0] = self.encoder_att[k][j](torch.cat((
                    g_encoder[j], atten_encoder[k][j - 1][2]), dim=1))
                atten_encoder[k][j][1] = atten_encoder[k][j][0] * g_encoder[j]
                atten_encoder[k][j][2] = self.encoder_block_att[j](
                    atten_encoder[k][j][1])
                if j < 3:
                    atten_encoder[k][j][2] = F.max_pool2d(atten_encoder[k][
                        j][2], kernel_size=2, stride=2)
        pred = F.avg_pool2d(atten_encoder[k][-1][-1], 8)
        pred = pred.view(pred.size(0), -1)
        out = self.linear[k](pred)
        return out

    def model_fit(self, x_pred, x_output, num_output):
        x_output_onehot = torch.zeros((len(x_output), num_output)).to(device)
        x_output_onehot.scatter_(1, x_output.unsqueeze(1), 1)
        loss = x_output_onehot * torch.log(x_pred + 1e-20)
        return torch.sum(-loss, dim=1)


class wide_basic(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1,
            bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes,
                kernel_size=1, stride=stride, bias=True))

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):

    def __init__(self, depth, widen_factor, num_classes):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        n = int((depth - 4) / 6)
        k = widen_factor
        filter = [16, 16 * k, 32 * k, 64 * k]
        self.conv1 = conv3x3(3, filter[0], stride=1)
        self.layer1 = self._wide_layer(wide_basic, filter[1], n, stride=2)
        self.layer2 = self._wide_layer(wide_basic, filter[2], n, stride=2)
        self.layer3 = self._wide_layer(wide_basic, filter[3], n, stride=2)
        self.bn1 = nn.BatchNorm2d(filter[3], momentum=0.9)
        self.linear = nn.ModuleList([nn.Sequential(nn.Linear(filter[3],
            num_classes[0]), nn.Softmax(dim=1))])
        self.encoder_att = nn.ModuleList([nn.ModuleList([self.att_layer([
            filter[0], filter[0], filter[0]])])])
        self.encoder_block_att = nn.ModuleList([self.conv_layer([filter[0],
            filter[1]])])
        for j in range(10):
            if j < 9:
                self.encoder_att.append(nn.ModuleList([self.att_layer([
                    filter[0], filter[0], filter[0]])]))
                self.linear.append(nn.Sequential(nn.Linear(filter[3],
                    num_classes[j + 1]), nn.Softmax(dim=1)))
            for i in range(3):
                self.encoder_att[j].append(self.att_layer([2 * filter[i + 1
                    ], filter[i + 1], filter[i + 1]]))
        for i in range(3):
            if i < 2:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1
                    ], filter[i + 2]]))
            else:
                self.encoder_block_att.append(self.conv_layer([filter[i + 1
                    ], filter[i + 1]]))

    def conv_layer(self, channel):
        conv_block = nn.Sequential(nn.Conv2d(in_channels=channel[0],
            out_channels=channel[1], kernel_size=3, padding=1), nn.
            BatchNorm2d(num_features=channel[1]), nn.ReLU(inplace=True))
        return conv_block

    def att_layer(self, channel):
        att_block = nn.Sequential(nn.Conv2d(in_channels=channel[0],
            out_channels=channel[1], kernel_size=1, padding=0), nn.
            BatchNorm2d(channel[1]), nn.ReLU(inplace=True), nn.Conv2d(
            in_channels=channel[1], out_channels=channel[2], kernel_size=1,
            padding=0), nn.BatchNorm2d(channel[2]), nn.Sigmoid())
        return att_block

    def _wide_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, k):
        g_encoder = [0] * 4
        atten_encoder = [0] * 10
        for i in range(10):
            atten_encoder[i] = [0] * 4
        for i in range(10):
            for j in range(4):
                atten_encoder[i][j] = [0] * 3
        g_encoder[0] = self.conv1(x)
        g_encoder[1] = self.layer1(g_encoder[0])
        g_encoder[2] = self.layer2(g_encoder[1])
        g_encoder[3] = F.relu(self.bn1(self.layer3(g_encoder[2])))
        for j in range(4):
            if j == 0:
                atten_encoder[k][j][0] = self.encoder_att[k][j](g_encoder[0])
                atten_encoder[k][j][1] = atten_encoder[k][j][0] * g_encoder[0]
                atten_encoder[k][j][2] = self.encoder_block_att[j](
                    atten_encoder[k][j][1])
                atten_encoder[k][j][2] = F.max_pool2d(atten_encoder[k][j][2
                    ], kernel_size=2, stride=2)
            else:
                atten_encoder[k][j][0] = self.encoder_att[k][j](torch.cat((
                    g_encoder[j], atten_encoder[k][j - 1][2]), dim=1))
                atten_encoder[k][j][1] = atten_encoder[k][j][0] * g_encoder[j]
                atten_encoder[k][j][2] = self.encoder_block_att[j](
                    atten_encoder[k][j][1])
                if j < 3:
                    atten_encoder[k][j][2] = F.max_pool2d(atten_encoder[k][
                        j][2], kernel_size=2, stride=2)
        pred = F.avg_pool2d(atten_encoder[k][-1][-1], 8)
        pred = pred.view(pred.size(0), -1)
        out = self.linear[k](pred)
        return out

    def model_fit(self, x_pred, x_output, num_output):
        x_output_onehot = torch.zeros((len(x_output), num_output)).to(device)
        x_output_onehot.scatter_(1, x_output.unsqueeze(1), 1)
        loss = x_output_onehot * torch.log(x_pred + 1e-20)
        return torch.sum(-loss, dim=1)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_lorenmt_mtan(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(ASPP(*[], **{'in_channels': 4, 'atrous_rates': [4, 4, 4]}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(ASPPConv(*[], **{'in_channels': 4, 'out_channels': 4, 'dilation': 1}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(ASPPPooling(*[], **{'in_channels': 4, 'out_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(BasicBlock(*[], **{'inplanes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(DeepLabHead(*[], **{'in_channels': 4, 'num_classes': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(WideResNet(*[], **{'depth': 1, 'widen_factor': 4, 'num_classes': [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]}), [torch.rand([4, 3, 64, 64]), 0], {})

    def test_006(self):
        self._check(wide_basic(*[], **{'in_planes': 4, 'planes': 4}), [torch.rand([4, 4, 4, 4])], {})
