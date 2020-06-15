import sys
_module = sys.modules[__name__]
del sys
adda = _module
config = _module
data = _module
models = _module
revgrad = _module
test_model = _module
train_source = _module
utils = _module
wdgrl = _module

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


from torch import nn


import torch


import torch.nn.functional as F


from torch.utils.data import DataLoader


import numpy as np


from torch.utils.data.sampler import SubsetRandomSampler


from torch.autograd import Function


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(nn.Conv2d(3, 10, kernel_size
            =5), nn.MaxPool2d(2), nn.ReLU(), nn.Conv2d(10, 20, kernel_size=
            5), nn.MaxPool2d(2), nn.Dropout2d())
        self.classifier = nn.Sequential(nn.Linear(320, 50), nn.ReLU(), nn.
            Dropout(), nn.Linear(50, 10))

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(x.shape[0], -1)
        logits = self.classifier(features)
        return logits


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)

    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):

    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_jvanvugt_pytorch_domain_adaptation(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(GradientReversal(*[], **{}), [torch.rand([4, 4, 4, 4])], {})
