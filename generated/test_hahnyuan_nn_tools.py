import sys
_module = sys.modules[__name__]
del sys
Caffe = _module
caffe_lmdb = _module
caffe_net = _module
caffe_pb2 = _module
layer_param = _module
net = _module
Datasets = _module
adience = _module
base_datasets = _module
cifar10 = _module
imagenet = _module
imdb_wiki = _module
lmdb_data_pb2 = _module
lmdb_datasets = _module
mnist = _module
wider = _module
Pytorch = _module
augmentations = _module
eval = _module
train = _module
utils = _module
Tensorflow = _module
constructor = _module
graph = _module
master = _module
CaffeA = _module
MxnetA = _module
PytorchA = _module
analysis = _module
blob = _module
layers = _module
roi = _module
caffe_analyser = _module
alexnet_pytorch_to_caffe = _module
densenet_pytorch_to_caffe = _module
inceptionv3_pytorch_to_caffe = _module
resnet_pytorch_analysis_example = _module
resnet_pytorch_to_caffe = _module
testify_pytorch_to_caffe_example = _module
vgg19_pytorch_to_caffe = _module
funcs = _module
keras_to_caffe = _module
mxnet_analyser = _module
pytorch_analyser = _module
pytorch_to_caffe = _module

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


import torch.optim as optim


from torch.autograd import Variable


import torch.nn as nn


import torch


import time


import numpy as np


from collections import OrderedDict


import torch.nn.functional as F


from torch.nn.modules.utils import _pair


import inspect


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_hahnyuan_nn_tools(_paritybench_base):
    pass