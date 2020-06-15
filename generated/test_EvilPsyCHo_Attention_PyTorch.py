import sys
_module = sys.modules[__name__]
del sys
models = _module
fasttext_attention = _module
layer = _module
utils = _module
base = _module
constant = _module
label = _module
text = _module
vocabulary = _module

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


from torch import nn


class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_model, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temper = np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(1)

    def forward(self, q, k, v, attn_mask=None):
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper
        if attn_mask is not None:
            assert attn_mask.size() == attn.size(
                ), 'Attention mask shape {} mismatch with Attention logit tensor shape {}.'.format(
                attn_mask.size(), attn.size())
            attn.data.masked_fill_(attn_mask, -float('inf'))
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_EvilPsyCHo_Attention_PyTorch(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(ScaledDotProductAttention(*[], **{'d_model': 4}), [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])], {})
