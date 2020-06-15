import sys
_module = sys.modules[__name__]
del sys
dataset = _module
main = _module
model = _module
ops = _module
solver = _module
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


import torch.nn as nn


import torch.nn.init as init


import torch


import torch.nn.functional as F


import torch.optim as optim


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.normal_(m.weight, 0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class Discriminator(nn.Module):

    def __init__(self, z_dim):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(nn.Linear(z_dim, 1000), nn.LeakyReLU(0.2, 
            True), nn.Linear(1000, 1000), nn.LeakyReLU(0.2, True), nn.
            Linear(1000, 1000), nn.LeakyReLU(0.2, True), nn.Linear(1000, 
            1000), nn.LeakyReLU(0.2, True), nn.Linear(1000, 1000), nn.
            LeakyReLU(0.2, True), nn.Linear(1000, 2))
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init
        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def forward(self, z):
        return self.net(z).squeeze()


class FactorVAE1(nn.Module):
    """Encoder and Decoder architecture for 2D Shapes data."""

    def __init__(self, z_dim=10):
        super(FactorVAE1, self).__init__()
        self.z_dim = z_dim
        self.encode = nn.Sequential(nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(True
            ), nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(True), nn.Conv2d(32, 64,
            4, 2, 1), nn.ReLU(True), nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(
            True), nn.Conv2d(64, 128, 4, 1), nn.ReLU(True), nn.Conv2d(128, 
            2 * z_dim, 1))
        self.decode = nn.Sequential(nn.Conv2d(z_dim, 128, 1), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4), nn.ReLU(True), nn.
            ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(True), nn.
            ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True), nn.
            ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(True), nn.
            ConvTranspose2d(32, 1, 4, 2, 1))
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init
        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, no_dec=False):
        stats = self.encode(x)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)
        if no_dec:
            return z.squeeze()
        else:
            x_recon = self.decode(z).view(x.size())
            return x_recon, mu, logvar, z.squeeze()


class FactorVAE2(nn.Module):
    """Encoder and Decoder architecture for 3D Shapes, Celeba, Chairs data."""

    def __init__(self, z_dim=10):
        super(FactorVAE2, self).__init__()
        self.z_dim = z_dim
        self.encode = nn.Sequential(nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(True
            ), nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(True), nn.Conv2d(32, 64,
            4, 2, 1), nn.ReLU(True), nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(
            True), nn.Conv2d(64, 256, 4, 1), nn.ReLU(True), nn.Conv2d(256, 
            2 * z_dim, 1))
        self.decode = nn.Sequential(nn.Conv2d(z_dim, 256, 1), nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4), nn.ReLU(True), nn.
            ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(True), nn.
            ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True), nn.
            ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(True), nn.
            ConvTranspose2d(32, 3, 4, 2, 1))
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init
        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, no_dec=False):
        stats = self.encode(x)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)
        if no_dec:
            return z.squeeze()
        else:
            x_recon = self.decode(z)
            return x_recon, mu, logvar, z.squeeze()


class FactorVAE3(nn.Module):
    """Encoder and Decoder architecture for 3D Faces data."""

    def __init__(self, z_dim=10):
        super(FactorVAE3, self).__init__()
        self.z_dim = z_dim
        self.encode = nn.Sequential(nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(True
            ), nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(True), nn.Conv2d(32, 64,
            4, 2, 1), nn.ReLU(True), nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(
            True), nn.Conv2d(64, 256, 4, 1), nn.ReLU(True), nn.Conv2d(256, 
            2 * z_dim, 1))
        self.decode = nn.Sequential(nn.Conv2d(z_dim, 256, 1), nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4), nn.ReLU(True), nn.
            ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(True), nn.
            ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True), nn.
            ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(True), nn.
            ConvTranspose2d(32, 1, 4, 2, 1))
        self.weight_init()

    def weight_init(self, mode='normal'):
        if mode == 'kaiming':
            initializer = kaiming_init
        elif mode == 'normal':
            initializer = normal_init
        for block in self._modules:
            for m in self._modules[block]:
                initializer(m)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = std.data.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x, no_dec=False):
        stats = self.encode(x)
        mu = stats[:, :self.z_dim]
        logvar = stats[:, self.z_dim:]
        z = self.reparametrize(mu, logvar)
        if no_dec:
            return z.squeeze()
        else:
            x_recon = self.decode(z)
            return x_recon, mu, logvar, z.squeeze()


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_1Konny_FactorVAE(_paritybench_base):
    pass
    def test_000(self):
        self._check(Discriminator(*[], **{'z_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(FactorVAE1(*[], **{}), [torch.rand([4, 1, 64, 64])], {})

    @_fails_compile()
    def test_002(self):
        self._check(FactorVAE2(*[], **{}), [torch.rand([4, 3, 64, 64])], {})

    @_fails_compile()
    def test_003(self):
        self._check(FactorVAE3(*[], **{}), [torch.rand([4, 1, 64, 64])], {})
