import sys
_module = sys.modules[__name__]
del sys
amla = _module
common = _module
comm = _module
schedule = _module
store = _module
task = _module
generate = _module
scheduler = _module
tf = _module
cell = _module
cell_classification = _module
cell_dag = _module
cell_init = _module
cell_main = _module
cifar10_input = _module
imagenet = _module
dataset = _module
image_processing = _module
imagenet_data = _module
imagenet_input = _module
evaluate = _module
generate_network = _module
net = _module
train = _module

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
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_CiscoAI_amla(_paritybench_base):
    pass