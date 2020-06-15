import sys
_module = sys.modules[__name__]
del sys
exp0 = _module
data_utils = _module
debug = _module
losses = _module
lr_scheduler = _module
metrics = _module
mlsnet = _module
models = _module
utils = _module
test_utils = _module

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


import numpy as np


import torch


import torch.nn as nn


import torch.optim as optim


from torch.utils.data import DataLoader


import torch.nn.functional as F


from itertools import filterfalse


from torch import nn


from torch.autograd import Variable


import logging


import time


from collections import OrderedDict


from collections import deque


class MixLoss(nn.Module):

    def __init__(self, bce_w=1.0, dice_w=0.0, focal_w=0.0, lovasz_w=0.0,
        bce_kwargs={}, dice_kwargs={}, focal_kwargs={}, lovasz_kwargs={}):
        super(MixLoss, self).__init__()
        self.bce_w = bce_w
        self.dice_w = dice_w
        self.focal_w = focal_w
        self.lovasz_w = lovasz_w
        self.bce_loss = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dice_loss = DiceLoss(**dice_kwargs)
        self.focal_loss = FocalLoss(**focal_kwargs)
        self.lovasz_loss = LovaszHinge(**lovasz_kwargs)

    def forward(self, output, target):
        loss = 0.0
        if self.bce_w:
            loss += self.bce_w * self.bce_loss(output, target)
        if self.dice_w:
            loss += self.dice_w * self.dice_loss(output, target)
        if self.focal_w:
            loss += self.focal_w * self.focal_loss(output, target)
        if self.lovasz_w:
            loss += self.lovasz_w * self.lovasz_loss(output, target)
        return loss


class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0, eps=1e-07):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps

    def forward(self, output, target):
        output = torch.sigmoid(output)
        if torch.sum(target) == 0:
            output = 1.0 - output
            target = 1.0 - target
        return 1.0 - (2 * torch.sum(output * target) + self.smooth) / (
            torch.sum(output) + torch.sum(target) + self.smooth + self.eps)


class SoftIoULoss(nn.Module):

    def __init__(self, n_classes=19):
        super(SoftIoULoss, self).__init__()
        self.n_classes = n_classes

    @staticmethod
    def to_one_hot(tensor, n_classes):
        n, h, w = tensor.size()
        one_hot = torch.zeros(n, n_classes, h, w).scatter_(1, tensor.view(n,
            1, h, w), 1)
        return one_hot

    def forward(self, logit, target):
        N = len(logit)
        pred = F.softmax(logit, dim=1)
        target_onehot = self.to_one_hot(target, self.n_classes)
        inter = pred * target_onehot
        inter = inter.view(N, self.n_classes, -1).sum(2)
        union = pred + target_onehot - pred * target_onehot
        union = union.view(N, self.n_classes, -1).sum(2)
        loss = inter / (union + 1e-16)
        return -loss.mean()


class FocalLoss(nn.Module):

    def __init__(self, gamma=2, eps=1e-07):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, logit, target):
        prob = torch.sigmoid(logit)
        prob = prob.clamp(self.eps, 1.0 - self.eps)
        loss = -1 * target * torch.log(prob)
        loss = loss * (1 - logit) ** self.gamma
        return loss.sum()


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def isnan(x):
    return x != x


def mean(l, ignore_nan=True, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = filterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


class LovaszHinge(nn.Module):

    def __init__(self, activation=lambda x: F.elu(x, inplace=True) + 1.0,
        per_image=True, ignore=None):
        super(LovaszHinge, self).__init__()
        self.activation = activation
        self.per_image = per_image
        self.ignore = ignore

    def lovasz_hinge_flat(self, logits, labels):
        """
        Binary Lovasz hinge loss
          logits: [P] Variable, logits at each prediction (between -\\infty and +\\infty)
          labels: [P] Tensor, binary ground truth labels (0 or 1)
          ignore: label to ignore
        """
        if len(labels) == 0:
            return logits.sum() * 0.0
        signs = 2.0 * labels.float() - 1.0
        errors = 1.0 - logits * signs
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = lovasz_grad(gt_sorted)
        loss = torch.dot(self.activation(errors_sorted), grad)
        return loss

    def forward(self, logits, labels):
        if self.per_image:
            loss = mean(self.lovasz_hinge_flat(*flatten_binary_scores(log.
                unsqueeze(0), lab.unsqueeze(0), self.ignore)) for log, lab in
                zip(logits, labels))
        else:
            loss = self.lovasz_hinge_flat(*flatten_binary_scores(logits,
                labels, self.ignore))
        return loss


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def lovasz_softmax_flat(probas, labels, only_present=False):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
    """
    if len(probas) == 0:
        return np.nan
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float()
        if only_present and fg.sum() == 0:
            continue
        errors = (fg - probas[:, (c)]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return mean(losses)


class LovaszSoftmax(nn.Module):
    """
    Multi-class Lovasz-Softmax loss
      logits: [B, C, H, W] class logits at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """

    def __init__(self, only_present=False, per_image=True, ignore=None):
        super(LovaszSoftmax, self).__init__()
        self.only_present = only_present
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, logits, labels):
        probas = F.softmax(logits, dim=1)
        if self.per_image:
            loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(
                0), lab.unsqueeze(0), self.ignore), only_present=self.
                only_present) for prob, lab in zip(probas, labels))
        else:
            loss = lovasz_softmax_flat(*flatten_probas(probas, labels, self
                .ignore), only_present=self.only_present)
        return loss


class OhemCrossEntropy2d(nn.Module):

    def __init__(self, ignore_label=255, thresh=0.6, min_kept=0, use_weight
        =True):
        super(OhemCrossEntropy2d, self).__init__()
        self.ignore_label = ignore_label
        self.thresh = float(thresh)
        self.min_kept = int(min_kept)
        if use_weight:
            None
            weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 
                1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 
                0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507]
                )
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight,
                ignore_index=ignore_label)
        else:
            None
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=
                ignore_label)

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), '{0} vs {1} '.format(predict
            .size(0), target.size(0))
        assert predict.size(2) == target.size(1), '{0} vs {1} '.format(predict
            .size(2), target.size(1))
        assert predict.size(3) == target.size(2), '{0} vs {1} '.format(predict
            .size(3), target.size(3))
        n, c, h, w = predict.size()
        input_label = target.data.cpu().numpy().ravel().astype(np.int32)
        x = np.rollaxis(predict.data.cpu().numpy(), 1).reshape((c, -1))
        input_prob = np.exp(x - x.max(axis=0).reshape((1, -1)))
        input_prob /= input_prob.sum(axis=0).reshape((1, -1))
        valid_flag = input_label != self.ignore_label
        valid_inds = np.where(valid_flag)[0]
        label = input_label[valid_flag]
        num_valid = valid_flag.sum()
        if self.min_kept >= num_valid:
            None
        elif num_valid > 0:
            prob = input_prob[:, (valid_flag)]
            pred = prob[label, np.arange(len(label), dtype=np.int32)]
            threshold = self.thresh
            if self.min_kept > 0:
                index = pred.argsort()
                threshold_index = index[min(len(index), self.min_kept) - 1]
                if pred[threshold_index] > self.thresh:
                    threshold = pred[threshold_index]
            kept_flag = pred <= threshold
            valid_inds = valid_inds[kept_flag]
            None
        label = input_label[valid_inds].copy()
        input_label.fill(self.ignore_label)
        input_label[valid_inds] = label
        None
        target = torch.from_numpy(input_label.reshape(target.size())).long()
        return self.criterion(predict, target)


class CriterionCrossEntropy(nn.Module):

    def __init__(self, ignore_index=255, weight='lightnet'):
        super(CriterionCrossEntropy, self).__init__()
        self.ignore_index = ignore_index
        if weight == 'lightnet':
            self.weight = torch.FloatTensor([0.05570516, 0.32337477, 
                0.08998544, 1.03602707, 1.03413147, 1.68195437, 5.58540548,
                3.56563995, 0.12704978, 1.0, 0.46783719, 1.34551528, 
                5.29974114, 0.28342531, 0.9396095, 0.81551811, 0.42679146, 
                3.6399074, 2.78376194])
        else:
            self.weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 
                1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 
                0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507]
                )
        self.criterion = torch.nn.CrossEntropyLoss(weight=self.weight,
            ignore_index=ignore_index)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        scale_pred = F.interpolate(input=preds, size=(h, w), mode=
            'bilinear', align_corners=True)
        loss = self.criterion(scale_pred, target)
        return loss


class CriterionDSN(nn.Module):

    def __init__(self, ignore_index=255, use_weight=True,
        loss_balance_coefs=(0.4, 1.0)):
        super(CriterionDSN, self).__init__()
        self.ignore_index = ignore_index
        self.loss_balance_coefs = loss_balance_coefs
        weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 
            0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116,
            0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        if use_weight:
            self.criterion = torch.nn.CrossEntropyLoss(weight=weight,
                ignore_index=ignore_index)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=
                ignore_index)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        assert len(preds) == len(self.loss_balance_coefs)
        losses = []
        for pred, coef in zip(preds, self.loss_balance_coefs):
            scale_pred = F.interpolate(input=pred, size=(h, w), mode=
                'bilinear', align_corners=True)
            losses.append(self.criterion(scale_pred, target) * coef)
        return sum(losses)


class CriterionOhemDSN(nn.Module):
    """
    DSN + OHEM : We need to consider two supervision for the model.
    """

    def __init__(self, ignore_index=255, thres=0.7, min_kept=100000,
        dsn_weight=0.4, use_weight=True):
        super(CriterionOhemDSN, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        self.criterion = OhemCrossEntropy2d(ignore_index, thres, min_kept,
            use_weight=use_weight)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        scale_pred = F.interpolate(input=preds[0], size=(h, w), mode=
            'bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)
        scale_pred = F.interpolate(input=preds[1], size=(h, w), mode=
            'bilinear', align_corners=True)
        loss2 = self.criterion(scale_pred, target)
        return self.dsn_weight * loss1 + loss2


class CriterionOhemDSN_single(nn.Module):
    """
    DSN + OHEM : we find that use hard-mining for both supervision harms the performance.
                Thus we choose the original loss for the shallow supervision
                and the hard-mining loss for the deeper supervision
    """

    def __init__(self, ignore_index=255, thres=0.7, min_kept=100000,
        dsn_weight=0.4):
        super(CriterionOhemDSN_single, self).__init__()
        self.ignore_index = ignore_index
        self.dsn_weight = dsn_weight
        weight = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 1.0166, 
            0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116,
            0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507])
        self.criterion = torch.nn.CrossEntropyLoss(weight=weight,
            ignore_index=ignore_index)
        self.criterion_ohem = OhemCrossEntropy2d(ignore_index, thres,
            min_kept, use_weight=True)

    def forward(self, preds, target):
        h, w = target.size(1), target.size(2)
        scale_pred = F.interpolate(input=preds[0], size=(h, w), mode=
            'bilinear', align_corners=True)
        loss1 = self.criterion(scale_pred, target)
        scale_pred = F.interpolate(input=preds[1], size=(h, w), mode=
            'bilinear', align_corners=True)
        loss2 = self.criterion_ohem(scale_pred, target)
        return self.dsn_weight * loss1 + loss2


class NoOperation(nn.Module):

    def __init__(self, *args, **kwargs):
        super(NoOperation, self).__init__()

    def forward(self, x):
        return x


class EncoderBase(nn.Module):

    def __init__(self, encoder, channels_list):
        super(EncoderBase, self).__init__()
        self.encoder = encoder
        self.channels_list = channels_list

    def forward(self, x):
        bridges = []
        for down in self.encoder:
            x = down(x)
            bridges.append(x)
        return bridges


class SelfAttentionBlock(nn.Module):
    """
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    """

    def __init__(self, in_channels, key_channels, value_channels,
        out_channels=None, scale=1, bn_module=nn.BatchNorm2d):
        super(SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels is None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,
            out_channels=self.key_channels, kernel_size=1, stride=1,
            padding=0), bn_module(self.key_channels))
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels
            =self.value_channels, kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=
            self.out_channels, kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)
        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)
        query = key.permute(0, 2, 1)
        sim_map = torch.matmul(query, key)
        sim_map = self.key_channels ** -0.5 * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, *x.size()[2:])
        context = self.W(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode=
                'bilinear', align_corners=True)
        return context


class BaseOC_Context_Module(nn.Module):
    """
    Output only the context features.
    Parameters:
        in_features / out_features: the channels of the input / output feature maps.
        size: we find that directly learn the attention weights on even 1/8 feature maps is hard.
    Return:
        features after "concat" or "add"
    """

    def __init__(self, in_channels, out_channels, key_channels,
        value_channels, sizes=[1], bn_module=nn.BatchNorm2d):
        super(BaseOC_Context_Module, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_channels,
            out_channels, key_channels, value_channels, size, bn_module) for
            size in sizes])

    @staticmethod
    def _make_stage(in_channels, output_channels, key_channels,
        value_channels, size, bn_module):
        return SelfAttentionBlock(in_channels, key_channels, value_channels,
            output_channels, size, bn_module)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        return context


class ASP_OC_Module(nn.Module):
    """
    OC-Module (bit modified version)
    ref: https://github.com/PkuRainBow/OCNet/blob/master/LICENSE
    """

    def __init__(self, in_features=2048, out_features=2048, dilations=(2, 5,
        9), bn_module=nn.BatchNorm2d, size=1):
        super(ASP_OC_Module, self).__init__()
        internal_features = in_features // 4
        self.context = nn.Sequential(nn.Conv2d(in_features,
            internal_features, kernel_size=3, padding=1, dilation=1, bias=
            True), bn_module(internal_features), BaseOC_Context_Module(
            in_channels=internal_features, out_channels=internal_features,
            key_channels=internal_features // 2, value_channels=
            internal_features, sizes=[size], bn_module=bn_module))
        self.conv2 = nn.Sequential(nn.Conv2d(in_features, internal_features,
            kernel_size=1, padding=0, dilation=1, bias=False), bn_module(
            internal_features))
        self.conv3 = nn.Sequential(nn.Conv2d(in_features, internal_features,
            kernel_size=3, padding=dilations[0], dilation=dilations[0],
            bias=False), bn_module(internal_features))
        self.conv4 = nn.Sequential(nn.Conv2d(in_features, internal_features,
            kernel_size=3, padding=dilations[1], dilation=dilations[1],
            bias=False), bn_module(internal_features))
        self.conv5 = nn.Sequential(nn.Conv2d(in_features, internal_features,
            kernel_size=3, padding=dilations[2], dilation=dilations[2],
            bias=False), bn_module(internal_features))
        self.conv_bn_dropout = nn.Sequential(nn.Conv2d(internal_features * 
            5, out_features, kernel_size=1, padding=0, dilation=1, bias=
            False), bn_module(out_features), nn.Dropout2d(0.1))

    @staticmethod
    def _cat_each(feat1, feat2, feat3, feat4, feat5):
        assert len(feat1) == len(feat2)
        z = []
        for i in range(len(feat1)):
            z.append(torch.cat((feat1[i], feat2[i], feat3[i], feat4[i],
                feat5[i]), 1))
        return z

    def forward(self, x):
        if isinstance(x, Variable):
            _, _, h, w = x.size()
        elif isinstance(x, tuple) or isinstance(x, list):
            _, _, h, w = x[0].size()
        else:
            raise RuntimeError('unknown input type')
        feat1 = self.context(x)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        if isinstance(x, Variable):
            out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        elif isinstance(x, tuple) or isinstance(x, list):
            out = self._cat_each(feat1, feat2, feat3, feat4, feat5)
        else:
            raise RuntimeError('unknown input type')
        output = self.conv_bn_dropout(out)
        return output


class UNet(nn.Module):

    def __init__(self, in_channels=1, n_classes=2, depth=5, ch_first=6,
        padding=False, batch_norm=False, up_mode='upconv'):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Using the default arguments will yield the exact version used
        in the original paper
        Args:
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            ch_first (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            up_mode (str): one of 'deconv' or 'upconv'.
                           'deconv' will use transposed convolutions for
                           learned upsampling.
                           'upconv' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('deconv', 'upconv')
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2 ** (
                ch_first + i), padding, batch_norm))
            prev_channels = 2 ** (ch_first + i)
        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2 ** (ch_first +
                i), up_mode, padding, batch_norm))
            prev_channels = 2 ** (ch_first + i)
        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.avg_pool2d(x, 2)
        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])
        return self.last(x)


class UNetConvBlock(nn.Module):

    def __init__(self, in_size, out_size, padding, batch_norm):
        super(UNetConvBlock, self).__init__()
        block = []
        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=
            int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=
            int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):

    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        if up_mode == 'deconv':
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2,
                stride=2)
        elif up_mode == 'upconv':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear',
                scale_factor=2), nn.Conv2d(in_size, out_size, kernel_size=1))
        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:diff_y + target_size[0], diff_x:diff_x +
            target_size[1]]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)
        return out


class MultiModalNN(nn.Module):

    def __init__(self, emb_dims, n_numeric_feats, n_channels_list=(64, 128),
        n_classes=1, emb_dropout=0.2, dropout_list=(0.5, 0.5)):
        """
        Parameters
        ----------

        emb_dims: List of two element tuples
          This list will contain a two element tuple for each
          categorical feature. The first element of a tuple will
          denote the number of unique values of the categorical
          feature. The second element will denote the embedding
          dimension to be used for that feature.

        n_numeric_feats: Integer
          The number of continuous features in the data.

        n_channels_list: List of integers.
          The size of each linear layer. The length will be equal
          to the total number
          of linear layers in the network.

        n_classes: Integer
          The size of the final output.

        emb_dropout: Float
          The dropout to be used after the embedding layers.

        dropout_list: List of floats
          The dropouts to be used after each linear layer.

        Examples
        --------
        >>> cat_dims = [int(data[col].nunique()) for col in categorical_features]
        >>> cat_dims
        [15, 5, 2, 4, 112]
        >>> emb_dims = [(x, min(32, (x + 1) // 2)) for x in cat_dims]
        >>> emb_dims
        [(15, 8), (5, 3), (2, 1), (4, 2), (112, 32)]
        >>> model = MultiModalNN(emb_dims, n_numeric_feats=4, lin_layer_sizes=[50, 100],
        >>>                      output_size=1, emb_dropout=0.04,
        >>>                      lin_layer_dropouts=[0.001,0.01]).to(device)
        """
        super(MultiModalNN, self).__init__()
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in
            emb_dims])
        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.n_numeric_feats = n_numeric_feats
        first_lin_layer = nn.Linear(self.no_of_embs + self.n_numeric_feats,
            n_channels_list[0])
        self.lin_layers = nn.ModuleList([first_lin_layer] + [nn.Linear(
            n_channels_list[i], n_channels_list[i + 1]) for i in range(len(
            n_channels_list) - 1)])
        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)
        self.output_layer = nn.Linear(n_channels_list[-1], n_classes)
        nn.init.kaiming_normal_(self.output_layer.weight.data)
        self.first_bn_layer = nn.BatchNorm1d(self.n_numeric_feats)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size) for size in
            n_channels_list])
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.droput_layers = nn.ModuleList([nn.Dropout(size) for size in
            dropout_list])

    def forward(self, numeric_feats, categorical_feats):
        if self.no_of_embs != 0:
            x = [emb_layer(categorical_feats[:, (i)]) for i, emb_layer in
                enumerate(self.emb_layers)]
            x = torch.cat(x, 1)
            x = self.emb_dropout_layer(x)
        if self.n_numeric_feats != 0:
            normalized_numeric_feats = self.first_bn_layer(numeric_feats)
            if self.no_of_embs != 0:
                x = torch.cat([x, normalized_numeric_feats], 1)
            else:
                x = normalized_numeric_feats
        for lin_layer, dropout_layer, bn_layer in zip(self.lin_layers, self
            .droput_layers, self.bn_layers):
            x = F.relu(lin_layer(x))
            x = bn_layer(x)
            x = dropout_layer(x)
        x = self.output_layer(x)
        return x


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_lyakaap_pytorch_template(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(ASP_OC_Module(*[], **{}), [torch.rand([4, 2048, 64, 64])], {})

    @_fails_compile()
    def test_001(self):
        self._check(BaseOC_Context_Module(*[], **{'in_channels': 4, 'out_channels': 4, 'key_channels': 4, 'value_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    def test_002(self):
        self._check(DiceLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_003(self):
        self._check(FocalLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_004(self):
        self._check(LovaszHinge(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_005(self):
        self._check(MixLoss(*[], **{}), [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])], {})

    def test_006(self):
        self._check(NoOperation(*[], **{}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_007(self):
        self._check(SelfAttentionBlock(*[], **{'in_channels': 4, 'key_channels': 4, 'value_channels': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_008(self):
        self._check(SoftIoULoss(*[], **{}), [torch.rand([4, 19, 4, 4]), torch.zeros([4, 4, 4], dtype=torch.int64)], {})

    def test_009(self):
        self._check(UNetConvBlock(*[], **{'in_size': 4, 'out_size': 4, 'padding': 4, 'batch_norm': 4}), [torch.rand([4, 4, 4, 4])], {})
