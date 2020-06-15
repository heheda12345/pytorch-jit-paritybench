import sys
_module = sys.modules[__name__]
del sys
setup = _module
test_functional = _module
test_layers = _module
torchaudio_contrib = _module
beta_hpss = _module
functional = _module
layers = _module

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


import torch.nn.functional as F


import math


def hpss(mag_specgrams, kernel_size=31, power=2.0, hard=False, mask_only=False
    ):
    """
    A function that performs harmonic-percussive source separation.
    Original method is by Derry Fitzgerald
    (https://www.researchgate.net/publication/254583990_HarmonicPercussive_Separation_using_Median_Filtering).

    Args:
        mag_specgrams (Tensor): any magnitude spectrograms in batch, (not in a decibel scale!)
            in a shape of (batch, ch, freq, time)

        kernel_size (int or (int, int)): odd-numbered
            if tuple,
                1st: width of percussive-enhancing filter (one along freq axis)
                2nd: width of harmonic-enhancing filter (one along time axis)
            if int,
                it's applied for both perc/harm filters

        power (float): to which the enhanced spectrograms are used in computing soft masks.

        hard (bool): whether the mask will be binarized (True) or not

        mask_only (bool): if true, returns the masks only.

    Returns:
        ret (Tuple): A tuple of four

            ret[0]: magnitude spectrograms - harmonic parts (Tensor, in same size with `mag_specgrams`)
            ret[1]: magnitude spectrograms - percussive parts (Tensor, in same size with `mag_specgrams`)
            ret[2]: harmonic mask (Tensor, in same size with `mag_specgrams`)
            ret[3]: percussive mask (Tensor, in same size with `mag_specgrams`)
    """

    def _enhance_either_hpss(mag_specgrams_padded, out, kernel_size, power,
        which, offset):
        """
        A helper function for HPSS

        Args:
            mag_specgrams_padded (Tensor): one that median filtering can be directly applied

            out (Tensor): The tensor to store the result

            kernel_size (int): The kernel size of median filter

            power (float): to which the enhanced spectrograms are used in computing soft masks.

            which (str): either 'harm' or 'perc'

            offset (int): the padded length

        """
        if which == 'harm':
            for t in range(out.shape[3]):
                out[:, :, :, (t)] = torch.median(mag_specgrams_padded[:, :,
                    offset:-offset, t:t + kernel_size], dim=3)[0]
        elif which == 'perc':
            for f in range(out.shape[2]):
                out[:, :, (f), :] = torch.median(mag_specgrams_padded[:, :,
                    f:f + kernel_size, offset:-offset], dim=2)[0]
        else:
            raise NotImplementedError(
                'it should be either but you passed which={}'.format(which))
        if power != 1.0:
            out.pow_(power)
    eps = 1e-06
    if not (isinstance(kernel_size, tuple) or isinstance(kernel_size, int)):
        raise TypeError(
            'kernel_size is expected to be either tuple of input, but it is: %s'
             % type(kernel_size))
    if isinstance(kernel_size, int):
        kernel_size = kernel_size, kernel_size
    pad = kernel_size[0] // 2, kernel_size[0] // 2, kernel_size[1
        ] // 2, kernel_size[1] // 2
    harm, perc, ret = torch.empty_like(mag_specgrams), torch.empty_like(
        mag_specgrams), torch.empty_like(mag_specgrams)
    mag_specgrams_padded = F.pad(mag_specgrams, pad=pad, mode='reflect')
    _enhance_either_hpss(mag_specgrams_padded, out=perc, kernel_size=
        kernel_size[0], power=power, which='perc', offset=kernel_size[1] // 2)
    _enhance_either_hpss(mag_specgrams_padded, out=harm, kernel_size=
        kernel_size[1], power=power, which='harm', offset=kernel_size[0] // 2)
    if hard:
        mask_harm = harm > perc
        mask_perc = harm < perc
    else:
        mask_harm = (harm + eps) / (harm + perc + eps)
        mask_perc = (perc + eps) / (harm + perc + eps)
    if mask_only:
        return None, None, mask_harm, mask_perc
    return (mag_specgrams * mask_harm, mag_specgrams * mask_perc, mask_harm,
        mask_perc)


class HPSS(nn.Module):
    """
    Wrap hpss.

    Args and Returns --> see `hpss`.
    """

    def __init__(self, kernel_size=31, power=2.0, hard=False, mask_only=False):
        super(HPSS, self).__init__()
        self.kernel_size = kernel_size
        self.power = power
        self.hard = hard
        self.mask_only = mask_only

    def forward(self, mag_specgrams):
        return hpss(mag_specgrams, self.kernel_size, self.power, self.hard,
            self.mask_only)

    def __repr__(self):
        return (self.__class__.__name__ +
            '(kernel_size={}, power={}, hard={}, mask_only={})'.format(self
            .kernel_size, self.power, self.hard, self.mask_only))


class _ModuleNoStateBuffers(nn.Module):
    """
    Extension of nn.Module that removes buffers
    from state_dict.
    """

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        ret = super(_ModuleNoStateBuffers, self).state_dict(destination,
            prefix, keep_vars)
        for k in self._buffers:
            del ret[prefix + k]
        return ret

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        buffers = self._buffers
        self._buffers = {}
        result = super(_ModuleNoStateBuffers, self)._load_from_state_dict(
            state_dict, prefix, *args, **kwargs)
        self._buffers = buffers
        return result


def complex_norm(complex_tensor, power=1.0):
    """Compute the norm of complex tensor input

    Args:
        complex_tensor (Tensor): Tensor shape of `(*, complex=2)`
        power (float): Power of the norm. Defaults to `1.0`.

    Returns:
        Tensor: power of the normed input tensor, shape of `(*, )`
    """
    if power == 1.0:
        return torch.norm(complex_tensor, 2, -1)
    return torch.norm(complex_tensor, 2, -1).pow(power)


class ComplexNorm(nn.Module):
    """Compute the norm of complex tensor input

    Args:
        power (float): Power of the norm. Defaults to `1.0`.

    """

    def __init__(self, power=1.0):
        super(ComplexNorm, self).__init__()
        self.power = power

    def forward(self, complex_tensor):
        """
        Args:
            complex_tensor (Tensor): Tensor shape of `(*, complex=2)`

        Returns:
            Tensor: norm of the input tensor, shape of `(*, )`
        """
        return complex_norm(complex_tensor, self.power)

    def __repr__(self):
        return self.__class__.__name__ + '(power={})'.format(self.power)


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_keunwoochoi_torchaudio_contrib(_paritybench_base):
    pass
    @_fails_compile()
    def test_000(self):
        self._check(ComplexNorm(*[], **{}), [torch.rand([4, 4, 4, 4])], {})
