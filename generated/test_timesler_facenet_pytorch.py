import sys
_module = sys.modules[__name__]
del sys
master = _module
inception_resnet_v1 = _module
mtcnn = _module
detect_face = _module
tensorflow2pytorch = _module
training = _module
setup = _module
perf_test = _module
travis_test = _module

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


from torch import nn


from torch.nn import functional as F


import numpy as np


from torch.nn.functional import interpolate


import time


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1,
            affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(256, 32, kernel_size=1,
            stride=1), BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1))
        self.branch2 = nn.Sequential(BasicConv2d(256, 32, kernel_size=1,
            stride=1), BasicConv2d(32, 32, kernel_size=3, stride=1, padding
            =1), BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1))
        self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale
        self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(896, 128, kernel_size=1,
            stride=1), BasicConv2d(128, 128, kernel_size=(1, 7), stride=1,
            padding=(0, 3)), BasicConv2d(128, 128, kernel_size=(7, 1),
            stride=1, padding=(3, 0)))
        self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super().__init__()
        self.scale = scale
        self.noReLU = noReLU
        self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)
        self.branch1 = nn.Sequential(BasicConv2d(1792, 192, kernel_size=1,
            stride=1), BasicConv2d(192, 192, kernel_size=(1, 3), stride=1,
            padding=(0, 1)), BasicConv2d(192, 192, kernel_size=(3, 1),
            stride=1, padding=(1, 0)))
        self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super().__init__()
        self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)
        self.branch1 = nn.Sequential(BasicConv2d(256, 192, kernel_size=1,
            stride=1), BasicConv2d(192, 192, kernel_size=3, stride=1,
            padding=1), BasicConv2d(192, 256, kernel_size=3, stride=2))
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super().__init__()
        self.branch0 = nn.Sequential(BasicConv2d(896, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 384, kernel_size=3, stride=2))
        self.branch1 = nn.Sequential(BasicConv2d(896, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 256, kernel_size=3, stride=2))
        self.branch2 = nn.Sequential(BasicConv2d(896, 256, kernel_size=1,
            stride=1), BasicConv2d(256, 256, kernel_size=3, stride=1,
            padding=1), BasicConv2d(256, 256, kernel_size=3, stride=2))
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


def get_torch_home():
    torch_home = os.path.expanduser(os.getenv('TORCH_HOME', os.path.join(os
        .getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))
    return torch_home


def load_weights(mdl, name):
    """Download pretrained state_dict and load into model.

    Arguments:
        mdl {torch.nn.Module} -- Pytorch model.
        name {str} -- Name of dataset that was used to generate pretrained state_dict.

    Raises:
        ValueError: If 'pretrained' not equal to 'vggface2' or 'casia-webface'.
    """
    if name == 'vggface2':
        features_path = (
            'https://drive.google.com/uc?export=download&id=1cWLH_hPns8kSfMz9kKl9PsG5aNV2VSMn'
            )
        logits_path = (
            'https://drive.google.com/uc?export=download&id=1mAie3nzZeno9UIzFXvmVZrDG3kwML46X'
            )
    elif name == 'casia-webface':
        features_path = (
            'https://drive.google.com/uc?export=download&id=1LSHHee_IQj5W3vjBcRyVaALv4py1XaGy'
            )
        logits_path = (
            'https://drive.google.com/uc?export=download&id=1QrhPgn1bGlDxAil2uc07ctunCQoDnCzT'
            )
    else:
        raise ValueError(
            'Pretrained models only exist for "vggface2" and "casia-webface"')
    model_dir = os.path.join(get_torch_home(), 'checkpoints')
    os.makedirs(model_dir, exist_ok=True)
    state_dict = {}
    for i, path in enumerate([features_path, logits_path]):
        cached_file = os.path.join(model_dir, '{}_{}.pt'.format(name, path[
            -10:]))
        if not os.path.exists(cached_file):
            print('Downloading parameters ({}/2)'.format(i + 1))
            s = requests.Session()
            s.mount('https://', HTTPAdapter(max_retries=10))
            r = s.get(path, allow_redirects=True)
            with open(cached_file, 'wb') as f:
                f.write(r.content)
        state_dict.update(torch.load(cached_file))
    mdl.load_state_dict(state_dict)


class InceptionResnetV1(nn.Module):
    """Inception Resnet V1 model with optional loading of pretrained weights.

    Model parameters can be loaded based on pretraining on the VGGFace2 or CASIA-Webface
    datasets. Pretrained state_dicts are automatically downloaded on model instantiation if
    requested and cached in the torch cache. Subsequent instantiations use the cache rather than
    redownloading.

    Keyword Arguments:
        pretrained {str} -- Optional pretraining dataset. Either 'vggface2' or 'casia-webface'.
            (default: {None})
        classify {bool} -- Whether the model should output classification probabilities or feature
            embeddings. (default: {False})
        num_classes {int} -- Number of output classes. If 'pretrained' is set and num_classes not
            equal to that used for the pretrained model, the final linear layer will be randomly
            initialized. (default: {None})
        dropout_prob {float} -- Dropout probability. (default: {0.6})
    """

    def __init__(self, pretrained=None, classify=False, num_classes=None,
        dropout_prob=0.6, device=None):
        super().__init__()
        self.pretrained = pretrained
        self.classify = classify
        self.num_classes = num_classes
        if pretrained == 'vggface2':
            tmp_classes = 8631
        elif pretrained == 'casia-webface':
            tmp_classes = 10575
        elif pretrained is None and self.num_classes is None:
            raise Exception(
                'At least one of "pretrained" or "num_classes" must be specified'
                )
        else:
            tmp_classes = self.num_classes
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1
            )
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)
        self.repeat_1 = nn.Sequential(Block35(scale=0.17), Block35(scale=
            0.17), Block35(scale=0.17), Block35(scale=0.17), Block35(scale=
            0.17))
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(Block17(scale=0.1), Block17(scale=0.1
            ), Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1), Block17(scale=0.1),
            Block17(scale=0.1), Block17(scale=0.1))
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(Block8(scale=0.2), Block8(scale=0.2),
            Block8(scale=0.2), Block8(scale=0.2), Block8(scale=0.2))
        self.block8 = Block8(noReLU=True)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.last_linear = nn.Linear(1792, 512, bias=False)
        self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True
            )
        self.logits = nn.Linear(512, tmp_classes)
        if pretrained is not None:
            load_weights(self, pretrained)
        if self.num_classes is not None:
            self.logits = nn.Linear(512, self.num_classes)
        self.device = torch.device('cpu')
        if device is not None:
            self.device = device
            self.to(device)

    def forward(self, x):
        """Calculate embeddings or logits given a batch of input image tensors.

        Arguments:
            x {torch.tensor} -- Batch of image tensors representing faces.

        Returns:
            torch.tensor -- Batch of embedding vectors or multinomial logits.
        """
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        x = self.last_bn(x)
        if self.classify:
            x = self.logits(x)
        else:
            x = F.normalize(x, p=2, dim=1)
        return x


class PNet(nn.Module):
    """MTCNN PNet.
    
    Keyword Arguments:
        pretrained {bool} -- Whether or not to load saved pretrained weights (default: {True})
    """

    def __init__(self, pretrained=True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3)
        self.prelu2 = nn.PReLU(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.prelu3 = nn.PReLU(32)
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1)
        self.softmax4_1 = nn.Softmax(dim=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1)
        self.training = False
        if pretrained:
            state_dict_path = os.path.join(os.path.dirname(__file__),
                '../data/pnet.pt')
            state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        a = self.conv4_1(x)
        a = self.softmax4_1(a)
        b = self.conv4_2(x)
        return b, a


class RNet(nn.Module):
    """MTCNN RNet.
    
    Keyword Arguments:
        pretrained {bool} -- Whether or not to load saved pretrained weights (default: {True})
    """

    def __init__(self, pretrained=True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 28, kernel_size=3)
        self.prelu1 = nn.PReLU(28)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(28, 48, kernel_size=3)
        self.prelu2 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=2)
        self.prelu3 = nn.PReLU(64)
        self.dense4 = nn.Linear(576, 128)
        self.prelu4 = nn.PReLU(128)
        self.dense5_1 = nn.Linear(128, 2)
        self.softmax5_1 = nn.Softmax(dim=1)
        self.dense5_2 = nn.Linear(128, 4)
        self.training = False
        if pretrained:
            state_dict_path = os.path.join(os.path.dirname(__file__),
                '../data/rnet.pt')
            state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense4(x.view(x.shape[0], -1))
        x = self.prelu4(x)
        a = self.dense5_1(x)
        a = self.softmax5_1(a)
        b = self.dense5_2(x)
        return b, a


class ONet(nn.Module):
    """MTCNN ONet.
    
    Keyword Arguments:
        pretrained {bool} -- Whether or not to load saved pretrained weights (default: {True})
    """

    def __init__(self, pretrained=True):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.prelu1 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.prelu2 = nn.PReLU(64)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.prelu3 = nn.PReLU(64)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)
        self.prelu4 = nn.PReLU(128)
        self.dense5 = nn.Linear(1152, 256)
        self.prelu5 = nn.PReLU(256)
        self.dense6_1 = nn.Linear(256, 2)
        self.softmax6_1 = nn.Softmax(dim=1)
        self.dense6_2 = nn.Linear(256, 4)
        self.dense6_3 = nn.Linear(256, 10)
        self.training = False
        if pretrained:
            state_dict_path = os.path.join(os.path.dirname(__file__),
                '../data/onet.pt')
            state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense5(x.view(x.shape[0], -1))
        x = self.prelu5(x)
        a = self.dense6_1(x)
        a = self.softmax6_1(a)
        b = self.dense6_2(x)
        c = self.dense6_3(x)
        return b, c, a


def nms_numpy(boxes, scores, threshold, method):
    if boxes.size == 0:
        return np.empty((0, 3))
    x1 = boxes[:, (0)].copy()
    y1 = boxes[:, (1)].copy()
    x2 = boxes[:, (2)].copy()
    y2 = boxes[:, (3)].copy()
    s = scores
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]
        xx1 = np.maximum(x1[i], x1[idx]).copy()
        yy1 = np.maximum(y1[i], y1[idx]).copy()
        xx2 = np.minimum(x2[i], x2[idx]).copy()
        yy2 = np.minimum(y2[i], y2[idx]).copy()
        w = np.maximum(0.0, xx2 - xx1 + 1).copy()
        h = np.maximum(0.0, yy2 - yy1 + 1).copy()
        inter = w * h
        if method is 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o <= threshold)]
    pick = pick[:counter].copy()
    return pick


def batched_nms_numpy(boxes, scores, idxs, threshold, method):
    device = boxes.device
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=device)
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, (None)]
    boxes_for_nms = boxes_for_nms.cpu().numpy()
    scores = scores.cpu().numpy()
    keep = nms_numpy(boxes_for_nms, scores, threshold, method)
    return torch.as_tensor(keep, dtype=torch.long, device=device)


def bbreg(boundingbox, reg):
    if reg.shape[1] == 1:
        reg = torch.reshape(reg, (reg.shape[2], reg.shape[3]))
    w = boundingbox[:, (2)] - boundingbox[:, (0)] + 1
    h = boundingbox[:, (3)] - boundingbox[:, (1)] + 1
    b1 = boundingbox[:, (0)] + reg[:, (0)] * w
    b2 = boundingbox[:, (1)] + reg[:, (1)] * h
    b3 = boundingbox[:, (2)] + reg[:, (2)] * w
    b4 = boundingbox[:, (3)] + reg[:, (3)] * h
    boundingbox[:, :4] = torch.stack([b1, b2, b3, b4]).permute(1, 0)
    return boundingbox


def generateBoundingBox(reg, probs, scale, thresh):
    stride = 2
    cellsize = 12
    reg = reg.permute(1, 0, 2, 3)
    mask = probs >= thresh
    mask_inds = mask.nonzero()
    image_inds = mask_inds[:, (0)]
    score = probs[mask]
    reg = reg[:, (mask)].permute(1, 0)
    bb = mask_inds[:, 1:].type(reg.dtype).flip(1)
    q1 = ((stride * bb + 1) / scale).floor()
    q2 = ((stride * bb + cellsize - 1 + 1) / scale).floor()
    boundingbox = torch.cat([q1, q2, score.unsqueeze(1), reg], dim=1)
    return boundingbox, image_inds


def imresample(img, sz):
    im_data = interpolate(img, size=sz, mode='area')
    return im_data


def pad(boxes, w, h):
    boxes = boxes.trunc().int().cpu().numpy()
    x = boxes[:, (0)]
    y = boxes[:, (1)]
    ex = boxes[:, (2)]
    ey = boxes[:, (3)]
    x[x < 1] = 1
    y[y < 1] = 1
    ex[ex > w] = w
    ey[ey > h] = h
    return y, ey, x, ex


def rerec(bboxA):
    h = bboxA[:, (3)] - bboxA[:, (1)]
    w = bboxA[:, (2)] - bboxA[:, (0)]
    l = torch.max(w, h)
    bboxA[:, (0)] = bboxA[:, (0)] + w * 0.5 - l * 0.5
    bboxA[:, (1)] = bboxA[:, (1)] + h * 0.5 - l * 0.5
    bboxA[:, 2:4] = bboxA[:, :2] + l.repeat(2, 1).permute(1, 0)
    return bboxA


def detect_face(imgs, minsize, pnet, rnet, onet, threshold, factor, device):
    if isinstance(imgs, (np.ndarray, torch.Tensor)):
        imgs = torch.as_tensor(imgs, device=device)
        if len(imgs.shape) == 3:
            imgs = imgs.unsqueeze(0)
    else:
        if not isinstance(imgs, (list, tuple)):
            imgs = [imgs]
        if any(img.size != imgs[0].size for img in imgs):
            raise Exception(
                'MTCNN batch processing only compatible with equal-dimension images.'
                )
        imgs = np.stack([np.uint8(img) for img in imgs])
    imgs = torch.as_tensor(imgs, device=device)
    model_dtype = next(pnet.parameters()).dtype
    imgs = imgs.permute(0, 3, 1, 2).type(model_dtype)
    batch_size = len(imgs)
    h, w = imgs.shape[2:4]
    m = 12.0 / minsize
    minl = min(h, w)
    minl = minl * m
    scale_i = m
    scales = []
    while minl >= 12:
        scales.append(scale_i)
        scale_i = scale_i * factor
        minl = minl * factor
    boxes = []
    image_inds = []
    all_inds = []
    all_i = 0
    for scale in scales:
        im_data = imresample(imgs, (int(h * scale + 1), int(w * scale + 1)))
        im_data = (im_data - 127.5) * 0.0078125
        reg, probs = pnet(im_data)
        boxes_scale, image_inds_scale = generateBoundingBox(reg, probs[:, (
            1)], scale, threshold[0])
        boxes.append(boxes_scale)
        image_inds.append(image_inds_scale)
        all_inds.append(all_i + image_inds_scale)
        all_i += batch_size
    boxes = torch.cat(boxes, dim=0)
    image_inds = torch.cat(image_inds, dim=0).cpu()
    all_inds = torch.cat(all_inds, dim=0)
    pick = batched_nms(boxes[:, :4], boxes[:, (4)], all_inds, 0.5)
    boxes, image_inds = boxes[pick], image_inds[pick]
    pick = batched_nms(boxes[:, :4], boxes[:, (4)], image_inds, 0.7)
    boxes, image_inds = boxes[pick], image_inds[pick]
    regw = boxes[:, (2)] - boxes[:, (0)]
    regh = boxes[:, (3)] - boxes[:, (1)]
    qq1 = boxes[:, (0)] + boxes[:, (5)] * regw
    qq2 = boxes[:, (1)] + boxes[:, (6)] * regh
    qq3 = boxes[:, (2)] + boxes[:, (7)] * regw
    qq4 = boxes[:, (3)] + boxes[:, (8)] * regh
    boxes = torch.stack([qq1, qq2, qq3, qq4, boxes[:, (4)]]).permute(1, 0)
    boxes = rerec(boxes)
    y, ey, x, ex = pad(boxes, w, h)
    if len(boxes) > 0:
        im_data = []
        for k in range(len(y)):
            if ey[k] > y[k] - 1 and ex[k] > x[k] - 1:
                img_k = imgs[(image_inds[k]), :, y[k] - 1:ey[k], x[k] - 1:ex[k]
                    ].unsqueeze(0)
                im_data.append(imresample(img_k, (24, 24)))
        im_data = torch.cat(im_data, dim=0)
        im_data = (im_data - 127.5) * 0.0078125
        out = rnet(im_data)
        out0 = out[0].permute(1, 0)
        out1 = out[1].permute(1, 0)
        score = out1[(1), :]
        ipass = score > threshold[1]
        boxes = torch.cat((boxes[(ipass), :4], score[ipass].unsqueeze(1)),
            dim=1)
        image_inds = image_inds[ipass]
        mv = out0[:, (ipass)].permute(1, 0)
        pick = batched_nms(boxes[:, :4], boxes[:, (4)], image_inds, 0.7)
        boxes, image_inds, mv = boxes[pick], image_inds[pick], mv[pick]
        boxes = bbreg(boxes, mv)
        boxes = rerec(boxes)
    points = torch.zeros(0, 5, 2, device=device)
    if len(boxes) > 0:
        y, ey, x, ex = pad(boxes, w, h)
        im_data = []
        for k in range(len(y)):
            if ey[k] > y[k] - 1 and ex[k] > x[k] - 1:
                img_k = imgs[(image_inds[k]), :, y[k] - 1:ey[k], x[k] - 1:ex[k]
                    ].unsqueeze(0)
                im_data.append(imresample(img_k, (48, 48)))
        im_data = torch.cat(im_data, dim=0)
        im_data = (im_data - 127.5) * 0.0078125
        out = onet(im_data)
        out0 = out[0].permute(1, 0)
        out1 = out[1].permute(1, 0)
        out2 = out[2].permute(1, 0)
        score = out2[(1), :]
        points = out1
        ipass = score > threshold[2]
        points = points[:, (ipass)]
        boxes = torch.cat((boxes[(ipass), :4], score[ipass].unsqueeze(1)),
            dim=1)
        image_inds = image_inds[ipass]
        mv = out0[:, (ipass)].permute(1, 0)
        w_i = boxes[:, (2)] - boxes[:, (0)] + 1
        h_i = boxes[:, (3)] - boxes[:, (1)] + 1
        points_x = w_i.repeat(5, 1) * points[:5, :] + boxes[:, (0)].repeat(5, 1
            ) - 1
        points_y = h_i.repeat(5, 1) * points[5:10, :] + boxes[:, (1)].repeat(
            5, 1) - 1
        points = torch.stack((points_x, points_y)).permute(2, 1, 0)
        boxes = bbreg(boxes, mv)
        pick = batched_nms_numpy(boxes[:, :4], boxes[:, (4)], image_inds, 
            0.7, 'Min')
        boxes, image_inds, points = boxes[pick], image_inds[pick], points[pick]
    boxes = boxes.cpu().numpy()
    points = points.cpu().numpy()
    batch_boxes = []
    batch_points = []
    for b_i in range(batch_size):
        b_i_inds = np.where(image_inds == b_i)
        batch_boxes.append(boxes[b_i_inds].copy())
        batch_points.append(points[b_i_inds].copy())
    batch_boxes, batch_points = np.array(batch_boxes), np.array(batch_points)
    return batch_boxes, batch_points


def crop_resize(img, box, image_size):
    if isinstance(img, np.ndarray):
        out = cv2.resize(img[box[1]:box[3], box[0]:box[2]], (image_size,
            image_size), interpolation=cv2.INTER_AREA).copy()
    else:
        out = img.crop(box).copy().resize((image_size, image_size), Image.
            BILINEAR)
    return out


def get_size(img):
    if isinstance(img, np.ndarray):
        return img.shape[1::-1]
    else:
        return img.size


def save_img(img, path):
    if isinstance(img, np.ndarray):
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        img.save(path)


def extract_face(img, box, image_size=160, margin=0, save_path=None):
    """Extract face + margin from PIL Image given bounding box.
    
    Arguments:
        img {PIL.Image} -- A PIL Image.
        box {numpy.ndarray} -- Four-element bounding box.
        image_size {int} -- Output image size in pixels. The image will be square.
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image. 
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size.
        save_path {str} -- Save path for extracted face image. (default: {None})
    
    Returns:
        torch.tensor -- tensor representing the extracted face.
    """
    margin = [margin * (box[2] - box[0]) / (image_size - margin), margin *
        (box[3] - box[1]) / (image_size - margin)]
    raw_image_size = get_size(img)
    box = [int(max(box[0] - margin[0] / 2, 0)), int(max(box[1] - margin[1] /
        2, 0)), int(min(box[2] + margin[0] / 2, raw_image_size[0])), int(
        min(box[3] + margin[1] / 2, raw_image_size[1]))]
    face = crop_resize(img, box, image_size)
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) + '/', exist_ok=True)
        save_img(face, save_path)
    face = F.to_tensor(np.float32(face))
    return face


def fixed_image_standardization(image_tensor):
    processed_tensor = (image_tensor - 127.5) / 128.0
    return processed_tensor


class MTCNN(nn.Module):
    """MTCNN face detection module.

    This class loads pretrained P-, R-, and O-nets and returns images cropped to include the face
    only, given raw input images of one of the following types:
        - PIL image or list of PIL images
        - numpy.ndarray (uint8) representing either a single image (3D) or a batch of images (4D).
    Cropped faces can optionally be saved to file
    also.
    
    Keyword Arguments:
        image_size {int} -- Output image size in pixels. The image will be square. (default: {160})
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image. 
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size (this is a bug in davidsandberg/facenet).
            (default: {0})
        min_face_size {int} -- Minimum face size to search for. (default: {20})
        thresholds {list} -- MTCNN face detection thresholds (default: {[0.6, 0.7, 0.7]})
        factor {float} -- Factor used to create a scaling pyramid of face sizes. (default: {0.709})
        post_process {bool} -- Whether or not to post process images tensors before returning.
            (default: {True})
        select_largest {bool} -- If True, if multiple faces are detected, the largest is returned.
            If False, the face with the highest detection probability is returned.
            (default: {True})
        keep_all {bool} -- If True, all detected faces are returned, in the order dictated by the
            select_largest parameter. If a save_path is specified, the first face is saved to that
            path and the remaining faces are saved to <save_path>1, <save_path>2 etc.
        device {torch.device} -- The device on which to run neural net passes. Image tensors and
            models are copied to this device before running forward passes. (default: {None})
    """

    def __init__(self, image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        select_largest=True, keep_all=False, device=None):
        super().__init__()
        self.image_size = image_size
        self.margin = margin
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.factor = factor
        self.post_process = post_process
        self.select_largest = select_largest
        self.keep_all = keep_all
        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()
        self.device = torch.device('cpu')
        if device is not None:
            self.device = device
            self.to(device)

    def forward(self, img, save_path=None, return_prob=False):
        """Run MTCNN face detection on a PIL image or numpy array. This method performs both
        detection and extraction of faces, returning tensors representing detected faces rather
        than the bounding boxes. To access bounding boxes, see the MTCNN.detect() method below.
        
        Arguments:
            img {PIL.Image, np.ndarray, or list} -- A PIL image, np.ndarray, or list.
        
        Keyword Arguments:
            save_path {str} -- An optional save path for the cropped image. Note that when
                self.post_process=True, although the returned tensor is post processed, the saved
                face image is not, so it is a true representation of the face in the input image.
                If `img` is a list of images, `save_path` should be a list of equal length.
                (default: {None})
            return_prob {bool} -- Whether or not to return the detection probability.
                (default: {False})
        
        Returns:
            Union[torch.Tensor, tuple(torch.tensor, float)] -- If detected, cropped image of a face
                with dimensions 3 x image_size x image_size. Optionally, the probability that a
                face was detected. If self.keep_all is True, n detected faces are returned in an
                n x 3 x image_size x image_size tensor with an optional list of detection
                probabilities. If `img` is a list of images, the item(s) returned have an extra 
                dimension (batch) as the first dimension.

        Example:
        >>> from facenet_pytorch import MTCNN
        >>> mtcnn = MTCNN()
        >>> face_tensor, prob = mtcnn(img, save_path='face.png', return_prob=True)
        """
        with torch.no_grad():
            batch_boxes, batch_probs = self.detect(img)
        batch_mode = True
        if not isinstance(img, (list, tuple)) and not (isinstance(img, np.
            ndarray) and len(img.shape) == 4):
            img = [img]
            batch_boxes = [batch_boxes]
            batch_probs = [batch_probs]
            batch_mode = False
        if save_path is not None:
            if isinstance(save_path, str):
                save_path = [save_path]
        else:
            save_path = [None for _ in range(len(img))]
        faces, probs = [], []
        for im, box_im, prob_im, path_im in zip(img, batch_boxes,
            batch_probs, save_path):
            if box_im is None:
                faces.append(None)
                probs.append([None] if self.keep_all else None)
                continue
            if not self.keep_all:
                box_im = box_im[[0]]
            faces_im = []
            for i, box in enumerate(box_im):
                face_path = path_im
                if path_im is not None and i > 0:
                    save_name, ext = os.path.splitext(path_im)
                    face_path = save_name + '_' + str(i + 1) + ext
                face = extract_face(im, box, self.image_size, self.margin,
                    face_path)
                if self.post_process:
                    face = fixed_image_standardization(face)
                faces_im.append(face)
            if self.keep_all:
                faces_im = torch.stack(faces_im)
            else:
                faces_im = faces_im[0]
                prob_im = prob_im[0]
            faces.append(faces_im)
            probs.append(prob_im)
        if not batch_mode:
            faces = faces[0]
            probs = probs[0]
        if return_prob:
            return faces, probs
        else:
            return faces

    def detect(self, img, landmarks=False):
        """Detect all faces in PIL image and return bounding boxes and optional facial landmarks.

        This method is used by the forward method and is also useful for face detection tasks
        that require lower-level handling of bounding boxes and facial landmarks (e.g., face
        tracking). The functionality of the forward function can be emulated by using this method
        followed by the extract_face() function.
        
        Arguments:
            img {PIL.Image, np.ndarray, or list} -- A PIL image or a list of PIL images.

        Keyword Arguments:
            landmarks {bool} -- Whether to return facial landmarks in addition to bounding boxes.
                (default: {False})
        
        Returns:
            tuple(numpy.ndarray, list) -- For N detected faces, a tuple containing an
                Nx4 array of bounding boxes and a length N list of detection probabilities.
                Returned boxes will be sorted in descending order by detection probability if
                self.select_largest=False, otherwise the largest face will be returned first.
                If `img` is a list of images, the items returned have an extra dimension
                (batch) as the first dimension. Optionally, a third item, the facial landmarks,
                are returned if `landmarks=True`.

        Example:
        >>> from PIL import Image, ImageDraw
        >>> from facenet_pytorch import MTCNN, extract_face
        >>> mtcnn = MTCNN(keep_all=True)
        >>> boxes, probs, points = mtcnn.detect(img, landmarks=True)
        >>> # Draw boxes and save faces
        >>> img_draw = img.copy()
        >>> draw = ImageDraw.Draw(img_draw)
        >>> for i, (box, point) in enumerate(zip(boxes, points)):
        ...     draw.rectangle(box.tolist(), width=5)
        ...     for p in point:
        ...         draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
        ...     extract_face(img, box, save_path='detected_face_{}.png'.format(i))
        >>> img_draw.save('annotated_faces.png')
        """
        with torch.no_grad():
            batch_boxes, batch_points = detect_face(img, self.min_face_size,
                self.pnet, self.rnet, self.onet, self.thresholds, self.
                factor, self.device)
        boxes, probs, points = [], [], []
        for box, point in zip(batch_boxes, batch_points):
            box = np.array(box)
            point = np.array(point)
            if len(box) == 0:
                boxes.append(None)
                probs.append([None])
                points.append(None)
            elif self.select_largest:
                box_order = np.argsort((box[:, (2)] - box[:, (0)]) * (box[:,
                    (3)] - box[:, (1)]))[::-1]
                box = box[box_order]
                point = point[box_order]
                boxes.append(box[:, :4])
                probs.append(box[:, (4)])
                points.append(point)
            else:
                boxes.append(box[:, :4])
                probs.append(box[:, (4)])
                points.append(point)
        boxes = np.array(boxes)
        probs = np.array(probs)
        points = np.array(points)
        if not isinstance(img, (list, tuple)) and not (isinstance(img, np.
            ndarray) and len(img.shape) == 4):
            boxes = boxes[0]
            probs = probs[0]
            points = points[0]
        if landmarks:
            return boxes, probs, points
        return boxes, probs


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_timesler_facenet_pytorch(_paritybench_base):
    pass
    def test_000(self):
        self._check(BasicConv2d(*[], **{'in_planes': 4, 'out_planes': 4, 'kernel_size': 4, 'stride': 1}), [torch.rand([4, 4, 4, 4])], {})

    def test_001(self):
        self._check(Block17(*[], **{}), [torch.rand([4, 896, 64, 64])], {})

    def test_002(self):
        self._check(Block35(*[], **{}), [torch.rand([4, 256, 64, 64])], {})

    def test_003(self):
        self._check(Block8(*[], **{}), [torch.rand([4, 1792, 64, 64])], {})

    def test_004(self):
        self._check(Mixed_6a(*[], **{}), [torch.rand([4, 256, 64, 64])], {})

    def test_005(self):
        self._check(Mixed_7a(*[], **{}), [torch.rand([4, 896, 64, 64])], {})
