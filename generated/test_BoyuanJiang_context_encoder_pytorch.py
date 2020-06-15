import sys
_module = sys.modules[__name__]
del sys
model = _module
old_way = _module
old_way_p = _module
psnr = _module
test = _module
test_one = _module
train = _module
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


import torch


import torch.nn as nn


import random


import torch.nn.parallel


import torch.backends.cudnn as cudnn


import torch.optim as optim


import torch.utils.data


from torch.autograd import Variable


import numpy as np


class _netG(nn.Module):

    def __init__(self, opt):
        super(_netG, self).__init__()
        self.ngpu = opt.ngpu
        self.main = nn.Sequential(nn.Conv2d(opt.nc, opt.nef, 4, 2, 1, bias=
            False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(opt.nef, opt
            .nef, 4, 2, 1, bias=False), nn.BatchNorm2d(opt.nef), nn.
            LeakyReLU(0.2, inplace=True), nn.Conv2d(opt.nef, opt.nef * 2, 4,
            2, 1, bias=False), nn.BatchNorm2d(opt.nef * 2), nn.LeakyReLU(
            0.2, inplace=True), nn.Conv2d(opt.nef * 2, opt.nef * 4, 4, 2, 1,
            bias=False), nn.BatchNorm2d(opt.nef * 4), nn.LeakyReLU(0.2,
            inplace=True), nn.Conv2d(opt.nef * 4, opt.nef * 8, 4, 2, 1,
            bias=False), nn.BatchNorm2d(opt.nef * 8), nn.LeakyReLU(0.2,
            inplace=True), nn.Conv2d(opt.nef * 8, opt.nBottleneck, 4, bias=
            False), nn.BatchNorm2d(opt.nBottleneck), nn.LeakyReLU(0.2,
            inplace=True), nn.ConvTranspose2d(opt.nBottleneck, opt.ngf * 8,
            4, 1, 0, bias=False), nn.BatchNorm2d(opt.ngf * 8), nn.ReLU(True
            ), nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=
            False), nn.BatchNorm2d(opt.ngf * 4), nn.ReLU(True), nn.
            ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 2), nn.ReLU(True), nn.ConvTranspose2d(
            opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(opt.
            ngf), nn.ReLU(True), nn.ConvTranspose2d(opt.ngf, opt.nc, 4, 2, 
            1, bias=False), nn.Tanh())

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self
                .ngpu))
        else:
            output = self.main(input)
        return output


class _netlocalD(nn.Module):

    def __init__(self, opt):
        super(_netlocalD, self).__init__()
        self.ngpu = opt.ngpu
        self.main = nn.Sequential(nn.Conv2d(opt.nc, opt.ndf, 4, 2, 1, bias=
            False), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(opt.ndf, opt
            .ndf * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(opt.ndf * 2), nn
            .LeakyReLU(0.2, inplace=True), nn.Conv2d(opt.ndf * 2, opt.ndf *
            4, 4, 2, 1, bias=False), nn.BatchNorm2d(opt.ndf * 4), nn.
            LeakyReLU(0.2, inplace=True), nn.Conv2d(opt.ndf * 4, opt.ndf * 
            8, 4, 2, 1, bias=False), nn.BatchNorm2d(opt.ndf * 8), nn.
            LeakyReLU(0.2, inplace=True), nn.Conv2d(opt.ndf * 8, 1, 4, 1, 0,
            bias=False), nn.Sigmoid())

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self
                .ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_BoyuanJiang_context_encoder_pytorch(_paritybench_base):
    pass