import sys
_module = sys.modules[__name__]
del sys
pytorch_translate = _module
attention = _module
attention_utils = _module
base_attention = _module
dot_attention = _module
mlp_attention = _module
multihead_attention = _module
no_attention = _module
pooling_attention = _module
average_attention = _module
beam_decode = _module
beam_search_and_decode_v2 = _module
benchmark = _module
bleu_significance = _module
char_aware_hybrid = _module
char_encoder = _module
char_source_hybrid = _module
char_source_model = _module
char_source_transformer_model = _module
checkpoint = _module
common_layers = _module
constants = _module
data = _module
char_data = _module
dictionary = _module
iterators = _module
language_pair_upsampling_dataset = _module
masked_lm_dictionary = _module
utils = _module
weighted_data = _module
dual_learning = _module
dual_learning_criterion = _module
dual_learning_models = _module
dual_learning_task = _module
ensemble_export = _module
evals = _module
file_io = _module
generate = _module
hybrid_transformer_rnn = _module
model_constants = _module
models = _module
transformer_from_pretrained_xlm = _module
multi_model = _module
multilingual = _module
multilingual_model = _module
multilingual_utils = _module
ngram = _module
options = _module
preprocess = _module
model_scorers = _module
rescorer = _module
test_model_scorers = _module
test_rescorer = _module
weights_search = _module
research = _module
multihead_attention = _module
beam_search = _module
competing_completed = _module
deliberation_networks = _module
knowledge_distillation = _module
collect_top_k_probs = _module
dual_decoder_kd_loss = _module
dual_decoder_kd_model = _module
hybrid_dual_decoder_kd_model = _module
knowledge_distillation_loss = _module
teacher_score_data = _module
lexical_choice = _module
lexical_translation = _module
multisource = _module
multisource_data = _module
multisource_decode = _module
rescore = _module
cloze_transformer_model = _module
rescoring_criterion = _module
test = _module
test_knowledge_distillation = _module
test_teacher_score_dataset = _module
test_unsupervised_morphology = _module
tune_model_weights = _module
tune_model_weights_with_ax = _module
rnn = _module
rnn_cell = _module
semi_supervised = _module
sequence_criterions = _module
tasks = _module
cross_lingual_lm = _module
denoising_autoencoder_task = _module
knowledge_distillation_task = _module
multilingual_task = _module
pytorch_translate_multi_task = _module
pytorch_translate_task = _module
semi_supervised_task = _module
translation_from_pretrained_xlm = _module
translation_lev_task = _module
gpu = _module
test_integration_gpu = _module
test_DecoderBatchedStepEnsemble = _module
test_attention = _module
test_beam_decode = _module
test_beam_search_and_decode = _module
test_bleu_significance = _module
test_char_aware_hybrid = _module
test_checkpoint = _module
test_data = _module
test_dictionary = _module
test_export = _module
test_export_beam_decode = _module
test_integration = _module
test_multilingual_utils = _module
test_options = _module
test_preprocess = _module
test_semi_supervised_task = _module
test_train = _module
test_utils = _module
test_vocab_reduction = _module
torchscript_export = _module
train = _module
transformer = _module
transformer_aan = _module
utils = _module
vocab_constants = _module
vocab_reduction = _module
weighted_criterions = _module
word_prediction = _module
word_prediction_criterion = _module
word_prediction_model = _module
word_predictor = _module
setup = _module

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


from typing import Dict


from typing import Optional


import numpy as np


import torch


import torch.nn.functional as F


from torch import Tensor


import torch.nn as nn


from torch import nn


import math


from typing import List


from torch.nn.utils.rnn import pack_padded_sequence


import logging


import abc


import copy


from typing import Tuple


import torch.jit


import torch.jit.quantized


import torch.onnx.operators


import time


from torch.serialization import default_restore_location


from collections import OrderedDict


from typing import Any


from torch.nn.utils.rnn import PackedSequence


from torch.nn.utils.rnn import pad_packed_sequence


import numpy.testing as npt


import random


from typing import Union


class BaseAttention(nn.Module):

    def __init__(self, decoder_hidden_state_dim, context_dim):
        super().__init__()
        self.decoder_hidden_state_dim = decoder_hidden_state_dim
        self.context_dim = context_dim

    def forward(self, decoder_state, source_hids, src_lengths):
        """
        Input
            decoder_state: bsz x decoder_hidden_state_dim
            source_hids: srclen x bsz x context_dim
            src_lengths: bsz x 1, actual sequence lengths
        Output
            output: bsz x context_dim
            attn_scores: max_src_len x bsz
        """
        raise NotImplementedError


class HighwayLayer(nn.Module):

    def __init__(self, input_dim, transform_activation=F.relu,
        gate_activation=F.softmax, gate_bias=-2):
        super().__init__()
        self.highway_transform_activation = transform_activation
        self.highway_gate_activation = gate_activation
        self.highway_transform = nn.Linear(input_dim, input_dim)
        self.highway_gate = nn.Linear(input_dim, input_dim)
        self.highway_gate.bias.data.fill_(gate_bias)

    def forward(self, x):
        transform_output = self.highway_transform_activation(self.
            highway_transform(x))
        gate_output = self.highway_gate_activation(self.highway_gate(x))
        transformation_part = torch.mul(transform_output, gate_output)
        carry_part = torch.mul(torch.FloatTensor([1.0]).type_as(gate_output
            ) - gate_output, x)
        return torch.add(transformation_part, carry_part)


TAGS = ['@DIGITS', '@EMOTICON', '@FBENTITY', '@MULTIPUNCT', '@NOTRANSLATE',
    '@PERSON', '@PLAIN', '@URL', '@USERNAME']


class CharCNNModel(nn.Module):
    """
    A Conv network to generate word embedding from character embeddings, from
    Character-Aware Neural Language Models, https://arxiv.org/abs/1508.06615.

    Components include convolutional filters, pooling, and
    optional highway network. We also have the ability to use pretrained ELMo
    which corresponds to the byte embeddings, CNN weights and the highway layer.
    """

    def __init__(self, dictionary, num_chars=50, char_embed_dim=32,
        convolutions_params='((128, 3), (128, 5))', nonlinear_fn_type=
        'tanh', num_highway_layers=0, char_cnn_output_dim=-1,
        use_pretrained_weights=False, finetune_pretrained_weights=False,
        weights_file=None):
        super().__init__()
        self.dictionary = dictionary
        self.padding_idx = dictionary.pad()
        self.use_pretrained_weights = use_pretrained_weights
        self.convolutions_params = convolutions_params
        self.num_highway_layers = num_highway_layers
        self.char_embed_dim = char_embed_dim
        self.num_embeddings = num_chars
        self.char_cnn_output_dim = char_cnn_output_dim
        self.filter_dims = sum(f[0] for f in self.convolutions_params)
        if use_pretrained_weights:
            self._weight_file = weights_file
            self._finetune_pretrained_weights = finetune_pretrained_weights
            self._load_weights()
        else:
            if nonlinear_fn_type == 'tanh':
                nonlinear_fn = nn.Tanh
            elif nonlinear_fn_type == 'relu':
                nonlinear_fn = nn.ReLU
            else:
                raise Exception('Invalid nonlinear type: {}'.format(
                    nonlinear_fn_type))
            self.embed_chars = rnn.Embedding(num_embeddings=num_chars,
                embedding_dim=char_embed_dim, padding_idx=self.padding_idx,
                freeze_embed=False)
            self.convolutions = nn.ModuleList([nn.Sequential(nn.Conv1d(
                char_embed_dim, num_filters, kernel_size, padding=
                kernel_size), nonlinear_fn()) for num_filters, kernel_size in
                self.convolutions_params])
            highway_layers = []
            for _ in range(self.num_highway_layers):
                highway_layers.append(HighwayLayer(self.filter_dims))
            self.highway_layers = nn.ModuleList(highway_layers)
            if char_cnn_output_dim != -1:
                self.projection = nn.Linear(self.filter_dims, self.
                    char_cnn_output_dim, bias=True)

    def _load_weights(self):
        """
        Function to load pretrained weights including byte embeddings.
        """
        self.npz_weights = np.load(self._weight_file)
        self._load_byte_embedding()
        self._load_cnn_weights()
        self._load_highway()
        self._load_projection()

    def _load_byte_embedding(self):
        """
        Function to load the pre-trained byte embeddings. We need to ensure that
        the embeddings account for special yoda tags as well.
        """
        char_embed_weights = self.npz_weights['char_embed']
        num_tags = TAGS.__len__()
        weights = np.zeros((char_embed_weights.shape[0] + num_tags + 1,
            char_embed_weights.shape[1]), dtype='float32')
        weights[1:-num_tags, :] = char_embed_weights
        self.embed_chars = rnn.Embedding(num_embeddings=self.num_embeddings,
            embedding_dim=self.char_embed_dim, padding_idx=self.padding_idx,
            freeze_embed=self._finetune_pretrained_weights)
        self.embed_chars.weight.data.copy_(torch.FloatTensor(weights))

    def _load_cnn_weights(self):
        """
        Function to load the weights associated with the pretrained CNN filters.
        For this to work correctly, the cnn params specified in the input arguments
        should match up with the pretrained architecture.
        """
        convolutions = []
        for i, (num_filters, kernel_size) in enumerate(self.convolutions_params
            ):
            conv = torch.nn.Conv1d(in_channels=self.char_embed_dim,
                out_channels=num_filters, kernel_size=kernel_size, padding=
                kernel_size, bias=True)
            weight = self.npz_weights['W_cnn_{}'.format(i)]
            bias = self.npz_weights['b_cnn_{}'.format(i)]
            w_reshaped = np.transpose(weight.squeeze(axis=0), axes=(2, 1, 0))
            if w_reshaped.shape != tuple(conv.weight.data.shape):
                raise ValueError('Invalid weight file')
            conv.weight.data.copy_(torch.div(torch.FloatTensor(w_reshaped),
                kernel_size * 1.0))
            conv.bias.data.copy_(torch.div(torch.FloatTensor(bias), 
                kernel_size * 1.0))
            conv.weight.requires_grad = self._finetune_pretrained_weights
            conv.bias.requires_grad = self._finetune_pretrained_weights
            convolutions.append(nn.Sequential(conv))
        self.convolutions = nn.ModuleList(convolutions)

    def _load_highway(self):
        """
        Function to load the weights associated with the pretrained highway
        network. In order to ensure the norm of the weights match up with the
        rest of the model, we need to normalize the pretrained weights.
        Here we divide by a fixed constant.
        """
        input_dim = sum(f[0] for f in self.convolutions_params)
        highway_layers = []
        for k in range(self.num_highway_layers):
            highway_layer = HighwayLayer(input_dim)
            w_transform = np.transpose(self.npz_weights['W_transform_{}'.
                format(k)])
            b_transform = self.npz_weights['b_transform_{}'.format(k)]
            highway_layer.highway_transform.weight.data.copy_(torch.div(
                torch.FloatTensor(w_transform), 6.0))
            highway_layer.highway_transform.bias.data.copy_(torch.
                FloatTensor(b_transform))
            highway_layer.highway_transform.weight.requires_grad = (self.
                _finetune_pretrained_weights)
            highway_layer.highway_transform.bias.requires_grad = (self.
                _finetune_pretrained_weights)
            w_carry = np.transpose(self.npz_weights['W_carry_{}'.format(k)])
            highway_layer.highway_gate.weight.data.copy_(torch.div(torch.
                FloatTensor(w_carry), 6.0))
            highway_layer.highway_gate.weight.requires_grad = (self.
                _finetune_pretrained_weights)
            b_carry = self.npz_weights['b_carry_{}'.format(k)]
            highway_layer.highway_gate.bias.data.copy_(torch.FloatTensor(
                b_carry))
            highway_layer.highway_gate.bias.requires_grad = (self.
                _finetune_pretrained_weights)
        highway_layers.append(highway_layer)
        self.highway_layers = nn.ModuleList(highway_layers)

    def _load_projection(self):
        """
        Function to load the weights associated with the pretrained projection
        layer. In order to ensure the norm of the weights match up with the
        rest of the model, we need to normalize the pretrained weights.
        Here we divide by a fixed constant.
        """
        input_dim = self.filter_dims
        self.projection = nn.Linear(input_dim, self.char_cnn_output_dim,
            bias=True)
        weight = self.npz_weights['W_proj']
        bias = self.npz_weights['b_proj']
        self.projection.weight.data.copy_(torch.div(torch.FloatTensor(np.
            transpose(weight)), 10.0))
        self.projection.bias.data.copy_(torch.div(torch.FloatTensor(np.
            transpose(bias)), 10.0))
        self.projection.weight.requires_grad = (self.
            _finetune_pretrained_weights)
        self.projection.bias.requires_grad = self._finetune_pretrained_weights

    def forward(self, char_inds_flat):
        x = self.embed_chars(char_inds_flat)
        encoder_padding_mask = char_inds_flat.eq(self.padding_idx)
        char_lengths = torch.sum(~encoder_padding_mask, dim=0)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None
        kernel_outputs = []
        for conv in self.convolutions:
            if encoder_padding_mask is not None:
                x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)
            conv_output = conv(x.permute(1, 2, 0))
            kernel_outputs.append(conv_output)
        pools = [self.pooling(conv, char_lengths, dim=2) for conv in
            kernel_outputs]
        encoder_output = torch.cat([p for p in pools], 1)
        for highway_layer in self.highway_layers:
            encoder_output = highway_layer(encoder_output)
        if self.char_cnn_output_dim != -1:
            encoder_output = self.projection(encoder_output)
        return encoder_output

    def pooling(self, inputs, char_lengths, dim):
        return torch.max(inputs, dim=dim)[0]


class CharRNNModel(nn.Module):
    """Bi-LSTM over characters to produce a word embedding from characters"""

    def __init__(self, dictionary, num_chars, char_embed_dim,
        char_rnn_units, char_rnn_layers):
        super().__init__()
        self.num_chars = num_chars
        self.padding_idx = dictionary.pad()
        self.embed_chars = rnn.Embedding(num_embeddings=num_chars,
            embedding_dim=char_embed_dim, padding_idx=self.padding_idx,
            freeze_embed=False)
        assert char_rnn_units % 2 == 0, 'char_rnn_units must be even (to be divided evenly between directions)'
        self.char_lstm_encoder = rnn.LSTMSequenceEncoder.LSTM(char_embed_dim,
            char_rnn_units // 2, num_layers=char_rnn_layers, bidirectional=True
            )
        self.onnx_export_model = False

    def forward(self, src_tokens, src_lengths, char_inds, word_lengths):
        bsz, seqlen, maxchars = char_inds.size()
        if self.onnx_export_model:
            assert bsz == 1
            maxchars_tensor = torch.onnx.operators.shape_as_tensor(char_inds)[2
                ]
            char_inds_flat_shape = torch.cat((torch.LongTensor([-1]),
                maxchars_tensor.view(1)))
            char_inds_flat = torch.onnx.operators.reshape_from_tensor_shape(
                char_inds, char_inds_flat_shape).t()
            char_rnn_input = self.embed_chars(char_inds_flat)
            packed_char_input = pack_padded_sequence(char_rnn_input,
                word_lengths.view(-1))
        else:
            nonzero_word_locations = word_lengths > 0
            word_lengths_flat = word_lengths[nonzero_word_locations]
            char_inds_flat = char_inds[nonzero_word_locations].t()
            sorted_word_lengths, word_length_order = torch.sort(
                word_lengths_flat, descending=True)
            char_rnn_input = self.embed_chars(char_inds_flat[:, (
                word_length_order)])
            packed_char_input = pack_padded_sequence(char_rnn_input,
                sorted_word_lengths)
        _, (h_last, _) = self.char_lstm_encoder(packed_char_input)
        char_rnn_output = torch.cat((h_last[(-2), :, :], h_last[(-1), :, :]
            ), dim=1)
        if self.onnx_export_model:
            x = char_rnn_output.unsqueeze(1)
        else:
            _, inverted_word_length_order = torch.sort(word_length_order)
            unsorted_rnn_output = char_rnn_output[(
                inverted_word_length_order), :]
            x = char_rnn_output.new(bsz, seqlen, unsorted_rnn_output.shape[1])
            x[nonzero_word_locations] = unsorted_rnn_output
            x = x.transpose(0, 1)
        return x

    def prepare_for_onnx_export_(self, **kwargs):
        self.onnx_export_model = True


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def NonlinearLayer(in_features, out_features, bias=True, activation_fn=nn.ReLU
    ):
    """Weight-normalized non-linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return nn.Sequential(m, activation_fn())


class ContextEmbedding(nn.Module):
    """
    This class implements context-dependent word embeddings as described in
    https://arxiv.org/pdf/1607.00578.pdf
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.nonlinear = NonlinearLayer(embed_dim, embed_dim, bias=True,
            activation_fn=nn.ReLU)
        self.linear = Linear(embed_dim, embed_dim, bias=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, src):
        c = torch.mean(self.nonlinear(src), 1, True)
        return src * self.sigmoid(self.linear(c))


class VariableLengthRecurrent(nn.Module):
    """
    This class acts as a generator of autograd for varying seq lengths with
    different padding behaviors, such as right padding, and order of seq lengths,
    such as descending order.

    The logic is mostly inspired from torch/nn/_functions/rnn.py, so it may be
    merged in the future.
    """

    def __init__(self, rnn_cell, reverse=False):
        super().__init__()
        self.rnn_cell = rnn_cell
        self.reverse = reverse

    def forward(self, x, hidden, batch_size_per_step):
        self.batch_size_per_step = batch_size_per_step
        self.starting_batch_size = batch_size_per_step[-1
            ] if self.reverse else batch_size_per_step[0]
        output = []
        input_offset = x.size(0) if self.reverse else 0
        hiddens = []
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = hidden,
        initial_hidden = hidden
        if self.reverse:
            hidden = tuple(h[:self.batch_size_per_step[-1]] for h in hidden)
        last_batch_size = self.starting_batch_size
        for i in range(len(self.batch_size_per_step)):
            if self.reverse:
                step_batch_size = self.batch_size_per_step[-1 - i]
                step_input = x[input_offset - step_batch_size:input_offset]
                input_offset -= step_batch_size
            else:
                step_batch_size = self.batch_size_per_step[i]
                step_input = x[input_offset:input_offset + step_batch_size]
                input_offset += step_batch_size
            new_pads = last_batch_size - step_batch_size
            if new_pads > 0:
                hiddens.insert(0, tuple(h[-new_pads:] for h in hidden))
                hidden = tuple(h[:-new_pads] for h in hidden)
            if new_pads < 0:
                hidden = tuple(torch.cat((h, ih[last_batch_size:
                    step_batch_size]), 0) for h, ih in zip(hidden,
                    initial_hidden))
            last_batch_size = step_batch_size
            if flat_hidden:
                hidden = self.rnn_cell(step_input, hidden[0]),
            else:
                hidden = self.rnn_cell(step_input, hidden)
            output.append(hidden[0])
        if not self.reverse:
            hiddens.insert(0, hidden)
            hidden = tuple(torch.cat(h, 0) for h in zip(*hiddens))
        assert output[0].size(0) == self.starting_batch_size
        if flat_hidden:
            hidden = hidden[0]
        if self.reverse:
            output.reverse()
        output = torch.cat(output, 0)
        return hidden, output


class RNNLayer(nn.Module):
    """
    A wrapper of rnn cells, with their corresponding forward function.
    If bidirectional, halve the hidden_size for each cell.
    """

    def __init__(self, input_size, hidden_size, cell_type='lstm',
        is_bidirectional=False):
        super().__init__()
        self.is_bidirectional = is_bidirectional
        num_directions = 2 if is_bidirectional else 1
        if cell_type == 'lstm':
            cell_class = rnn_cell.LSTMCell
        elif cell_type == 'milstm':
            cell_class = rnn_cell.MILSTMCell
        elif cell_type == 'layer_norm_lstm':
            cell_class = rnn_cell.LayerNormLSTMCell
        else:
            raise Exception(f'{cell_type} not implemented')
        self.fwd_cell = cell_class(input_size, hidden_size // num_directions)
        if is_bidirectional:
            self.bwd_cell = cell_class(input_size, hidden_size //
                num_directions)
        self.fwd_func = VariableLengthRecurrent(rnn_cell=self.fwd_cell,
            reverse=False)
        if is_bidirectional:
            self.bwd_func = VariableLengthRecurrent(rnn_cell=self.bwd_cell,
                reverse=True)

    def forward(self, x, hidden, batch_size_per_step):
        fwd_hidden, fwd_output = self.fwd_func.forward(x, hidden,
            batch_size_per_step)
        if self.is_bidirectional:
            bwd_hidden, bwd_output = self.bwd_func.forward(x, hidden,
                batch_size_per_step)
            combined_hidden = [fwd_hidden, bwd_hidden]
            bi_hiddens, bi_cells = zip(*combined_hidden)
            next_hidden = torch.cat(bi_hiddens, bi_hiddens[0].dim() - 1
                ), torch.cat(bi_cells, bi_cells[0].dim() - 1)
            output = torch.cat([fwd_output, bwd_output], x.dim() - 1)
        else:
            next_hidden = fwd_hidden
            output = fwd_output
        return next_hidden, output


class Embedding(nn.Embedding):
    """
    A wrapper around the embedding layer, which can be randomly initialized or
    loaded from a .npy file. Also supports normalization of embeddings to have
    zero mean and unit variance (weighted by token frequency) - this is useful
    for example when creating adversarial perturbations of the embeddings that
    should have norms relative to the embeddings' norms.
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx,
        freeze_embed=False, normalize_embed=False, normalize_decay_rate=0.99):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx
            )
        nn.init.uniform_(self.weight, -0.1, 0.1)
        nn.init.constant_(self.weight[padding_idx], 0.0)
        if freeze_embed:
            self.weight.requires_grad = False
        assert 0.0 < normalize_decay_rate < 1.0
        self.normalize = normalize_embed
        self.normalize_decay_rate = normalize_decay_rate
        self.mean = None
        self.var = None
        self.init_normalization_if_needed()

    def forward(self, x):
        x = super().forward(x)
        if self.normalize:
            if self.training:
                self._update_normalize_params(x)
            x = (x - self.mean) / torch.sqrt(self.var + 1e-06)
        return x

    def init_normalization_if_needed(self):
        if not self.normalize:
            return
        self.mean = nn.Parameter(self.weight.mean(dim=0), requires_grad=False)
        self.var = nn.Parameter(self.weight.var(dim=0), requires_grad=False)

    def _update_normalize_params(self, x):
        """
        Updates the observed mean and variance of the token embeddings. Note
        that these will be weighted by the empirical frequency of each token
        (i.e. common tokens will be more heavily weighted in the params).
        """
        assert x.size()[-1:] == self.mean.size()
        x_flattened = x.view(-1, x.size(-1))
        x_mean = x_flattened.mean(dim=0)
        self.mean.data = self.normalize_decay_rate * self.mean.data + (1.0 -
            self.normalize_decay_rate) * x_mean
        x_var = ((x_flattened - self.mean) ** 2).mean(dim=0)
        self.var.data = self.normalize_decay_rate * self.var.data + (1.0 -
            self.normalize_decay_rate) * x_var


class OutputProjection(nn.Module):
    """Output projection layer."""

    def __init__(self, out_embed_dim, vocab_size, vocab_reduction_module=None):
        super().__init__()
        self.out_embed_dim = out_embed_dim
        self.vocab_size = vocab_size
        self.output_projection_w = nn.Parameter(torch.FloatTensor(self.
            vocab_size, self.out_embed_dim).uniform_(-0.1, 0.1))
        self.output_projection_b = nn.Parameter(torch.FloatTensor(self.
            vocab_size).zero_())
        self.vocab_reduction_module = vocab_reduction_module

    def forward(self, x, src_tokens=None, input_tokens=None,
        possible_translation_tokens=None):
        output_projection_w = self.output_projection_w
        output_projection_b = self.output_projection_b
        decoder_input_tokens = input_tokens if self.training else None
        if self.vocab_reduction_module and possible_translation_tokens is None:
            possible_translation_tokens = self.vocab_reduction_module(
                src_tokens, decoder_input_tokens=decoder_input_tokens)
        if possible_translation_tokens is not None:
            output_projection_w = output_projection_w.index_select(dim=0,
                index=possible_translation_tokens)
            output_projection_b = output_projection_b.index_select(dim=0,
                index=possible_translation_tokens)
        batch_time_hidden = torch.onnx.operators.shape_as_tensor(x)
        x_flat_shape = torch.cat((torch.LongTensor([-1]), batch_time_hidden
            [2].view(1)))
        x_flat = torch.onnx.operators.reshape_from_tensor_shape(x, x_flat_shape
            )
        projection_flat = torch.matmul(output_projection_w, x_flat.t()).t()
        logits_shape = torch.cat((batch_time_hidden[:2], torch.LongTensor([
            -1])))
        logits = torch.onnx.operators.reshape_from_tensor_shape(projection_flat
            , logits_shape) + output_projection_b
        return logits, possible_translation_tokens


class TransformerEncoderGivenEmbeddings(nn.Module):

    def __init__(self, args, proj_to_decoder):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.layers.extend([fairseq_transformer.TransformerEncoderLayer(
            args) for i in range(args.encoder_layers)])

    def forward(self, x, positions, encoder_padding_mask):
        for layer in self.layers:
            x = layer(x, encoder_padding_mask)
        return x

    def upgrade_state_dict_named(self, state_dict, name):
        for i in range(len(self.layers)):
            self.layers[i].upgrade_state_dict_named(state_dict,
                f'{name}.layers.{i}')


class TransformerEmbedding(nn.Module):

    def __init__(self, args, embed_tokens):
        super().__init__()
        self.dropout = args.dropout
        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.embed_tokens = embed_tokens
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions = fairseq_transformer.PositionalEmbedding(1024,
            embed_dim, self.padding_idx, learned=args.encoder_learned_pos)

    def forward(self, src_tokens, src_lengths):
        x = self.embed_tokens(src_tokens)
        src_tokens_tensor = pytorch_translate_utils.get_source_tokens_tensor(
            src_tokens)
        x = self.embed_scale * x
        positions = self.embed_positions(src_tokens_tensor)
        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)
        encoder_padding_mask = src_tokens_tensor.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None
        return x, encoder_padding_mask, positions


class DecoderBatchedStepEnsemble(nn.Module):

    def __init__(self, models, tgt_dict, beam_size, word_reward=0,
        unk_reward=0, tile_internal=False):
        super().__init__()
        self.models = models
        for i, model in enumerate(self.models):
            model.prepare_for_onnx_export_()
            if hasattr(model, 'get_student_model'):
                model = model.get_student_model()
                self.models[i] = model
            self._modules[f'model_{i}'] = model
        self.tgt_dict = tgt_dict
        self.beam_size = beam_size
        self.word_reward = word_reward
        self.unk_reward = unk_reward
        vocab_size = len(tgt_dict.indices)
        self.word_rewards = torch.FloatTensor(vocab_size).fill_(word_reward)
        self.word_rewards[tgt_dict.eos()] = 0
        self.word_rewards[tgt_dict.unk()] = word_reward + unk_reward
        self.tile_internal = tile_internal
        self.enable_precompute_reduced_weights = False

    def forward(self, input_tokens, prev_scores, timestep, *inputs,
        src_tuple=None):
        """
        Decoder step inputs correspond one-to-one to encoder outputs.
        HOWEVER: after the first step, encoder outputs (i.e, the first
        len(self.models) elements of inputs) must be tiled k (beam size)
        times on the batch dimension (axis 1).
        """
        input_tokens = input_tokens.unsqueeze(1)
        (log_probs_per_model, attn_weights_per_model, state_outputs,
            beam_axis_per_state, possible_translation_tokens) = (self.
            _get_decoder_outputs(input_tokens, prev_scores, timestep, *
            inputs, src_tuple=src_tuple))
        average_log_probs = torch.mean(torch.cat(log_probs_per_model, dim=1
            ), dim=1, keepdim=True)
        if possible_translation_tokens is None:
            word_rewards = self.word_rewards
        else:
            word_rewards = self.word_rewards.index_select(0,
                possible_translation_tokens)
        word_rewards = word_rewards.unsqueeze(dim=0).unsqueeze(dim=0)
        average_log_probs_with_rewards = average_log_probs + word_rewards
        average_attn_weights = torch.mean(torch.cat(attn_weights_per_model,
            dim=1), dim=1, keepdim=True)
        best_scores_k_by_k, best_tokens_k_by_k = torch.topk(
            average_log_probs_with_rewards.squeeze(1), k=self.beam_size)
        prev_scores_k_by_k = prev_scores.view(-1, 1).expand(-1, self.beam_size)
        total_scores_k_by_k = best_scores_k_by_k + prev_scores_k_by_k
        total_scores_flat = total_scores_k_by_k.view(-1)
        best_tokens_flat = best_tokens_k_by_k.view(-1)
        best_scores, best_indices = torch.topk(total_scores_flat, k=self.
            beam_size)
        best_tokens = best_tokens_flat.index_select(dim=0, index=best_indices
            ).view(-1)
        prev_hypos = best_indices // self.beam_size
        attention_weights = average_attn_weights.index_select(dim=0, index=
            prev_hypos)
        if possible_translation_tokens is not None:
            best_tokens = possible_translation_tokens.index_select(dim=0,
                index=best_tokens)
        self.input_names = ['prev_tokens', 'prev_scores', 'timestep']
        for i in range(len(self.models)):
            self.input_names.append(f'fixed_input_{i}')
        if possible_translation_tokens is not None:
            self.input_names.append('possible_translation_tokens')
        attention_weights = attention_weights.squeeze(1)
        outputs = [best_tokens, best_scores, prev_hypos, attention_weights]
        self.output_names = ['best_tokens_indices', 'best_scores',
            'prev_hypos_indices', 'attention_weights_average']
        for i in range(len(self.models)):
            self.output_names.append(f'fixed_input_{i}')
            if self.tile_internal:
                outputs.append(inputs[i].repeat(1, self.beam_size, 1))
            else:
                outputs.append(inputs[i])
        if possible_translation_tokens is not None:
            self.output_names.append('possible_translation_tokens')
            outputs.append(possible_translation_tokens)
        for i, state in enumerate(state_outputs):
            beam_axis = beam_axis_per_state[i]
            if beam_axis is None:
                next_state = state
            else:
                next_state = state.index_select(dim=beam_axis, index=prev_hypos
                    )
            outputs.append(next_state)
            self.output_names.append(f'state_output_{i}')
            self.input_names.append(f'state_input_{i}')
        return tuple(outputs)

    def _get_decoder_outputs(self, input_tokens, prev_scores, timestep, *
        inputs, src_tuple=None):
        log_probs_per_model = []
        attn_weights_per_model = []
        state_outputs = []
        beam_axis_per_state = []
        reduced_output_weights_per_model = []
        next_state_input = len(self.models)
        batch_size = torch.onnx.operators.shape_as_tensor(input_tokens)[0]
        possible_translation_tokens = None
        if hasattr(self.models[0].decoder, 'vocab_reduction_module'):
            vocab_reduction_module = self.models[0
                ].decoder.vocab_reduction_module
            if vocab_reduction_module is not None:
                possible_translation_tokens = inputs[len(self.models)]
                next_state_input += 1
        futures = []
        for i, model in enumerate(self.models):
            if isinstance(model, rnn.RNNModel) or isinstance(model, rnn.
                DummyPyTextRNNPointerModel) or isinstance(model,
                char_source_model.CharSourceModel) or isinstance(model,
                word_prediction_model.WordPredictionModel):
                encoder_output = inputs[i]
                prev_hiddens = []
                prev_cells = []
                for _ in range(len(model.decoder.layers)):
                    prev_hiddens.append(inputs[next_state_input])
                    prev_cells.append(inputs[next_state_input + 1])
                    next_state_input += 2
                input_feed_shape = torch.cat((batch_size.view(1), torch.
                    LongTensor([-1])))
                prev_input_feed = (torch.onnx.operators.
                    reshape_from_tensor_shape(inputs[next_state_input],
                    input_feed_shape))
                next_state_input += 1
                if self.enable_precompute_reduced_weights and hasattr(model
                    .decoder, '_precompute_reduced_weights'
                    ) and possible_translation_tokens is not None:
                    reduced_output_weights = inputs[next_state_input:
                        next_state_input + 2]
                    next_state_input += 2
                else:
                    reduced_output_weights = None
                reduced_output_weights_per_model.append(reduced_output_weights)
                if src_tuple:
                    src_tokens, src_length = src_tuple
                    src_tokens = src_tokens.t()
                else:
                    src_length_int = int(encoder_output.size()[0])
                    src_length = torch.LongTensor(np.array([src_length_int]))
                    src_tokens = torch.LongTensor(np.array([[0] *
                        src_length_int]))
                src_embeddings = encoder_output.new_zeros(encoder_output.shape)
                encoder_out = (encoder_output, prev_hiddens, prev_cells,
                    src_length, src_tokens, src_embeddings)

                def forked_section(input_tokens, encoder_out,
                    possible_translation_tokens, prev_hiddens, prev_cells,
                    prev_input_feed, reduced_output_weights):
                    model.decoder._is_incremental_eval = True
                    model.eval()
                    incremental_state = {}
                    utils.set_incremental_state(model.decoder,
                        incremental_state, 'cached_state', (prev_hiddens,
                        prev_cells, prev_input_feed))
                    decoder_output = model.decoder(input_tokens,
                        encoder_out, incremental_state=incremental_state,
                        possible_translation_tokens=
                        possible_translation_tokens, reduced_output_weights
                        =reduced_output_weights)
                    logits, attn_scores, _ = decoder_output
                    log_probs = logits if isinstance(model, rnn.
                        DummyPyTextRNNPointerModel) else F.log_softmax(logits,
                        dim=2)
                    log_probs_per_model.append(log_probs)
                    attn_weights_per_model.append(attn_scores)
                    next_hiddens, next_cells, next_input_feed = (utils.
                        get_incremental_state(model.decoder,
                        incremental_state, 'cached_state'))
                    return log_probs, attn_scores, tuple(next_hiddens), tuple(
                        next_cells), next_input_feed
                fut = torch.jit._fork(forked_section, input_tokens,
                    encoder_out, possible_translation_tokens, prev_hiddens,
                    prev_cells, prev_input_feed, reduced_output_weights)
                futures.append(fut)
            elif isinstance(model, transformer.TransformerModel) or isinstance(
                model, char_source_transformer_model.CharSourceTransformerModel
                ):
                encoder_output = inputs[i]
                model.decoder._is_incremental_eval = True
                model.eval()
                states_per_layer = 4
                state_inputs = []
                for i, _ in enumerate(model.decoder.layers):
                    if hasattr(model.decoder, 'decoder_layers_to_keep'
                        ) and i not in model.decoder.decoder_layers_to_keep.keys(
                        ):
                        continue
                    state_inputs.extend(inputs[next_state_input:
                        next_state_input + states_per_layer])
                    next_state_input += states_per_layer
                encoder_out = encoder_output, None, None
                reduced_output_weights = None
                reduced_output_weights_per_model.append(reduced_output_weights)

                def forked_section(input_tokens, encoder_out, state_inputs,
                    possible_translation_tokens, timestep):
                    decoder_output = model.decoder(input_tokens,
                        encoder_out, incremental_state=state_inputs,
                        possible_translation_tokens=
                        possible_translation_tokens, timestep=timestep)
                    logits, attn_scores, _, attention_states = decoder_output
                    log_probs = F.log_softmax(logits, dim=2)
                    return log_probs, attn_scores, tuple(attention_states)
                fut = torch.jit._fork(forked_section, input_tokens,
                    encoder_out, state_inputs, possible_translation_tokens,
                    timestep)
                futures.append(fut)
            elif isinstance(model, levenshtein_transformer.
                LevenshteinTransformerModel):
                encoder_output = inputs[i]
                model.decoder._is_incremental_eval = True
                model.eval()
                states_per_layer = 4
                state_inputs = []
                for _ in model.decoder.layers:
                    state_inputs.extend(inputs[next_state_input:
                        next_state_input + states_per_layer])
                    next_state_input += states_per_layer
                encoder_out = encoder_output, None, None
                reduced_output_weights = None
                reduced_output_weights_per_model.append(reduced_output_weights)

                def forked_section(input_tokens, encoder_out, state_inputs,
                    possible_translation_tokens, timestep):
                    decoder_output = model.decoder(input_tokens,
                        encoder_out, incremental_state=state_inputs,
                        possible_translation_tokens=
                        possible_translation_tokens, timestep=timestep)
                    logits, attn_scores, attention_states = decoder_output
                    log_probs = F.log_softmax(logits, dim=2)
                    return log_probs, attn_scores, tuple(attention_states)
                fut = torch.jit._fork(forked_section, input_tokens,
                    encoder_out, state_inputs, possible_translation_tokens,
                    timestep)
                futures.append(fut)
            elif isinstance(model, latent_var_models.LatentVarModel):
                encoder_output = inputs[i]
                model.decoder._is_incremental_eval = True
                model.eval()
                state_inputs = []
                state_inputs.extend(inputs[next_state_input:
                    next_state_input + 3])
                next_state_input += 3
                for _ in list(model.decoder.decoders.values())[0].layers:
                    state_inputs.extend(inputs[next_state_input:
                        next_state_input + 4])
                    next_state_input += 4
                encoder_out = encoder_output
                reduced_output_weights = None
                reduced_output_weights_per_model.append(reduced_output_weights)

                def forked_section(input_tokens, encoder_out, state_inputs,
                    possible_translation_tokens, timestep):
                    decoder_output = model.decoder(input_tokens,
                        encoder_out, incremental_state=state_inputs)
                    logits, attn_scores, _, _, attention_states = (
                        decoder_output)
                    log_probs = F.log_softmax(logits, dim=2)
                    return log_probs, attn_scores, tuple(attention_states)
                fut = torch.jit._fork(forked_section, input_tokens,
                    encoder_out, state_inputs, possible_translation_tokens,
                    timestep)
                futures.append(fut)
            elif isinstance(model, hybrid_transformer_rnn.
                HybridTransformerRNNModel) or isinstance(model,
                char_source_hybrid.CharSourceHybridModel):
                encoder_output = inputs[i]
                model.decoder._is_incremental_eval = True
                model.eval()
                encoder_out = encoder_output, None, None
                num_states = (1 + model.decoder.num_layers) * 2
                state_inputs = inputs[next_state_input:next_state_input +
                    num_states]
                next_state_input += num_states
                reduced_output_weights = None
                reduced_output_weights_per_model.append(reduced_output_weights)

                def forked_section(input_tokens, encoder_out, state_inputs,
                    possible_translation_tokens, timestep):
                    incremental_state = {}
                    utils.set_incremental_state(model.decoder,
                        incremental_state, 'cached_state', state_inputs)
                    decoder_output = model.decoder(input_tokens,
                        encoder_out, incremental_state=incremental_state,
                        possible_translation_tokens=
                        possible_translation_tokens, timestep=timestep)
                    logits, attn_scores, _ = decoder_output
                    log_probs = F.log_softmax(logits, dim=2)
                    next_states = utils.get_incremental_state(model.decoder,
                        incremental_state, 'cached_state')
                    return log_probs, attn_scores, tuple(next_states)
                fut = torch.jit._fork(forked_section, input_tokens,
                    encoder_out, state_inputs, possible_translation_tokens,
                    timestep)
                futures.append(fut)
            else:
                raise RuntimeError(f'Not a supported model: {type(model)}')
        for i, (model, fut) in enumerate(zip(self.models, futures)):
            if isinstance(model, rnn.RNNModel) or isinstance(model, rnn.
                DummyPyTextRNNPointerModel) or isinstance(model,
                char_source_model.CharSourceModel) or isinstance(model,
                word_prediction_model.WordPredictionModel):
                (log_probs, attn_scores, next_hiddens, next_cells,
                    next_input_feed) = torch.jit._wait(fut)
                for h, c in zip(next_hiddens, next_cells):
                    state_outputs.extend([h, c])
                    beam_axis_per_state.extend([0, 0])
                state_outputs.append(next_input_feed)
                beam_axis_per_state.append(0)
                if reduced_output_weights_per_model[i] is not None:
                    state_outputs.extend(reduced_output_weights_per_model[i])
                    beam_axis_per_state.extend([None for _ in
                        reduced_output_weights_per_model[i]])
            elif isinstance(model, transformer.TransformerModel) or isinstance(
                model, char_source_transformer_model.CharSourceTransformerModel
                ):
                log_probs, attn_scores, attention_states = torch.jit._wait(fut)
                log_probs_per_model.append(log_probs)
                attn_weights_per_model.append(attn_scores)
                state_outputs.extend(attention_states)
                beam_axis_per_state.extend([(0) for _ in attention_states])
            elif isinstance(model, levenshtein_transformer.
                LevenshteinTransformerModel):
                log_probs, attn_scores, attention_states = torch.jit._wait(fut)
                log_probs_per_model.append(log_probs)
                attn_weights_per_model.append(attn_scores)
                state_outputs.extend(attention_states)
                beam_axis_per_state.extend([None for _ in attention_states])
            elif isinstance(model, latent_var_models.LatentVarModel):
                log_probs, attn_scores, attention_states = torch.jit._wait(fut)
                log_probs_per_model.append(log_probs)
                attn_weights_per_model.append(attn_scores)
                state_outputs.extend(attention_states)
                beam_axis_per_state.extend([(0) for _ in attention_states])
            elif isinstance(model, hybrid_transformer_rnn.
                HybridTransformerRNNModel) or isinstance(model,
                char_source_hybrid.CharSourceHybridModel):
                log_probs, attn_scores, next_states = torch.jit._wait(fut)
                log_probs_per_model.append(log_probs)
                attn_weights_per_model.append(attn_scores)
                state_outputs.extend(next_states)
                beam_axis_per_state.extend([(1) for _ in next_states[:-2]])
                beam_axis_per_state.extend([0, 0])
            else:
                raise RuntimeError(f'Not a supported model: {type(model)}')
        return (log_probs_per_model, attn_weights_per_model, state_outputs,
            beam_axis_per_state, possible_translation_tokens)

    @classmethod
    def build_from_checkpoints(cls, checkpoint_filenames, src_dict_filename,
        dst_dict_filename, beam_size, word_reward=0, unk_reward=0,
        lexical_dict_paths=None):
        models, _, tgt_dict = load_models_from_checkpoints(checkpoint_filenames
            , src_dict_filename, dst_dict_filename, lexical_dict_paths)
        return cls(models, tgt_dict, beam_size=beam_size, word_reward=
            word_reward, unk_reward=unk_reward)


class FakeEncoderEnsemble(torch.jit.ScriptModule):

    @torch.jit.script_method
    def forward(self, src_tokens, src_lengths) ->None:
        raise RuntimeError(
            'Called EncoderEnsemble on a BeamSearch thats not word-source')


class FakeCharSourceEncoderEnsemble(torch.jit.ScriptModule):

    @torch.jit.script_method
    def forward(self, src_tokens, src_lengths, char_inds, word_lengths) ->None:
        raise RuntimeError(
            'Called CharSourceEncoderEnsemble on a BeamSearch thats not char-source'
            )


class BeamSearch(torch.jit.ScriptModule):
    __constants__ = ['beam_size', 'is_char_source']

    def __init__(self, model_list, tgt_dict, src_tokens, src_lengths,
        beam_size=1, word_reward=0, unk_reward=0, quantize=False, char_inds
        =None, word_lengths=None):
        super().__init__()
        self.models = model_list
        self.tgt_dict = tgt_dict
        self.beam_size = beam_size
        self.word_reward = word_reward
        self.unk_reward = unk_reward
        if isinstance(self.models[0], char_source_model.CharSourceModel
            ) or isinstance(self.models[0], char_source_transformer_model.
            CharSourceTransformerModel) or isinstance(self.models[0],
            char_source_hybrid.CharSourceHybridModel):
            encoder_ens = CharSourceEncoderEnsemble(self.models)
        else:
            encoder_ens = EncoderEnsemble(self.models)
        encoder_ens.enable_precompute_reduced_weights = True
        if quantize:
            torch.quantization.quantize_dynamic(encoder_ens, {torch.nn.
                Linear}, dtype=torch.qint8, inplace=True)
            encoder_ens = torch.jit.quantized.quantize_rnn_cell_modules(
                encoder_ens)
        if isinstance(self.models[0], char_source_model.CharSourceModel
            ) or isinstance(self.models[0], char_source_transformer_model.
            CharSourceTransformerModel) or isinstance(self.models[0],
            char_source_hybrid.CharSourceHybridModel):
            self.is_char_source = True
            enc_inputs = src_tokens, src_lengths, char_inds, word_lengths
            example_encoder_outs = encoder_ens(*enc_inputs)
            self.encoder_ens = FakeEncoderEnsemble()
            self.encoder_ens_char_source = torch.jit.trace(encoder_ens,
                enc_inputs, _force_outplace=True, check_trace=False)
        else:
            self.is_char_source = False
            enc_inputs = src_tokens, src_lengths
            example_encoder_outs = encoder_ens(*enc_inputs)
            self.encoder_ens = torch.jit.trace(encoder_ens, enc_inputs,
                _force_outplace=True, check_trace=False)
            self.encoder_ens_char_source = FakeCharSourceEncoderEnsemble()
        decoder_ens = DecoderBatchedStepEnsemble(self.models, tgt_dict,
            beam_size, word_reward, unk_reward, tile_internal=False)
        decoder_ens.enable_precompute_reduced_weights = True
        if quantize:
            torch.quantization.quantize_dynamic(decoder_ens, {torch.nn.
                Linear}, dtype=torch.qint8, inplace=True)
            decoder_ens = torch.jit.quantized.quantize_rnn_cell_modules(
                decoder_ens)
            decoder_ens = torch.jit.quantized.quantize_rnn_modules(decoder_ens)
        decoder_ens_tile = DecoderBatchedStepEnsemble(self.models, tgt_dict,
            beam_size, word_reward, unk_reward, tile_internal=True)
        decoder_ens_tile.enable_precompute_reduced_weights = True
        if quantize:
            torch.quantization.quantize_dynamic(decoder_ens_tile, {torch.nn
                .Linear}, dtype=torch.qint8, inplace=True)
            decoder_ens_tile = torch.jit.quantized.quantize_rnn_cell_modules(
                decoder_ens_tile)
            decoder_ens_tile = torch.jit.quantized.quantize_rnn_modules(
                decoder_ens_tile)
        prev_token = torch.LongTensor([0])
        prev_scores = torch.FloatTensor([0.0])
        ts = torch.LongTensor([0])
        _, _, _, _, *tiled_states = decoder_ens_tile(prev_token,
            prev_scores, ts, *example_encoder_outs)
        self.decoder_ens_tile = torch.jit.trace(decoder_ens_tile, (
            prev_token, prev_scores, ts, *example_encoder_outs),
            _force_outplace=True, check_trace=False)
        self.decoder_ens = torch.jit.trace(decoder_ens, (prev_token.repeat(
            self.beam_size), prev_scores.repeat(self.beam_size), ts, *
            tiled_states), _force_outplace=True, check_trace=False)
        self.input_names = ['src_tokens', 'src_lengths', 'prev_token',
            'prev_scores', 'attn_weights', 'prev_hypos_indices', 'num_steps']
        self.output_names = ['all_tokens', 'all_scores', 'all_weights',
            'all_prev_indices']

    @torch.jit.script_method
    def forward(self, src_tokens: torch.Tensor, src_lengths: torch.Tensor,
        prev_token: torch.Tensor, prev_scores: torch.Tensor, attn_weights:
        torch.Tensor, prev_hypos_indices: torch.Tensor, num_steps: int,
        char_inds: Optional[torch.Tensor]=None, word_lengths: Optional[
        torch.Tensor]=None):
        if self.is_char_source:
            if char_inds is None or word_lengths is None:
                raise RuntimeError(
                    'char_inds and word_lengths must be specified for char-source models'
                    )
            char_inds = torch.jit._unwrap_optional(char_inds)
            word_lengths = torch.jit._unwrap_optional(word_lengths)
            enc_states = self.encoder_ens_char_source(src_tokens,
                src_lengths, char_inds, word_lengths)
        else:
            enc_states = self.encoder_ens(src_tokens, src_lengths)
        enc_states = torch.jit._unwrap_optional(enc_states)
        all_tokens = prev_token.repeat(repeats=[self.beam_size]).unsqueeze(dim
            =0)
        all_scores = prev_scores.repeat(repeats=[self.beam_size]).unsqueeze(dim
            =0)
        all_weights = attn_weights.unsqueeze(dim=0).repeat(repeats=[self.
            beam_size, 1]).unsqueeze(dim=0)
        all_prev_indices = prev_hypos_indices.unsqueeze(dim=0)
        (prev_token, prev_scores, prev_hypos_indices, attn_weights, *states
            ) = (self.decoder_ens_tile(prev_token, prev_scores, _to_tensor(
            0), *enc_states))
        all_tokens = torch.cat((all_tokens, prev_token.unsqueeze(dim=0)), dim=0
            )
        all_scores = torch.cat((all_scores, prev_scores.unsqueeze(dim=0)),
            dim=0)
        all_weights = torch.cat((all_weights, attn_weights.unsqueeze(dim=0)
            ), dim=0)
        all_prev_indices = torch.cat((all_prev_indices, prev_hypos_indices.
            unsqueeze(dim=0)), dim=0)
        for i in range(num_steps - 1):
            (prev_token, prev_scores, prev_hypos_indices, attn_weights, *states
                ) = (self.decoder_ens(prev_token, prev_scores, _to_tensor(i +
                1), *states))
            all_tokens = torch.cat((all_tokens, prev_token.unsqueeze(dim=0)
                ), dim=0)
            all_scores = torch.cat((all_scores, prev_scores.unsqueeze(dim=0
                )), dim=0)
            all_weights = torch.cat((all_weights, attn_weights.unsqueeze(
                dim=0)), dim=0)
            all_prev_indices = torch.cat((all_prev_indices,
                prev_hypos_indices.unsqueeze(dim=0)), dim=0)
        return all_tokens, all_scores, all_weights, all_prev_indices

    @classmethod
    def build_from_checkpoints(cls, checkpoint_filenames, src_dict_filename,
        dst_dict_filename, beam_size, word_reward=0, unk_reward=0,
        lexical_dict_paths=None):
        length = 10
        models, _, tgt_dict = load_models_from_checkpoints(checkpoint_filenames
            , src_dict_filename, dst_dict_filename, lexical_dict_paths)
        src_tokens = torch.LongTensor(np.ones((length, 1), dtype='int64'))
        src_lengths = torch.IntTensor(np.array([length], dtype='int32'))
        if isinstance(models[0], char_source_model.CharSourceModel
            ) or isinstance(models[0], char_source_transformer_model.
            CharSourceTransformerModel) or isinstance(models[0],
            char_source_hybrid.CharSourceHybridModel):
            word_length = 3
            char_inds = torch.LongTensor(np.ones((1, length, word_length),
                dtype='int64'))
            word_lengths = torch.IntTensor(np.array([word_length] * length,
                dtype='int32')).reshape((1, length))
        else:
            char_inds = None
            word_lengths = None
        return cls(models, tgt_dict, src_tokens, src_lengths, beam_size=
            beam_size, word_reward=word_reward, unk_reward=unk_reward,
            quantize=True, char_inds=char_inds, word_lengths=word_lengths)

    def save_to_pytorch(self, output_path):

        def pack(s):
            if hasattr(s, '_pack'):
                s._pack()

        def unpack(s):
            if hasattr(s, '_unpack'):
                s._unpack()
        self.apply(pack)
        torch.jit.save(self, output_path)
        self.apply(unpack)


class KnownOutputDecoderStepEnsemble(nn.Module):

    def __init__(self, models, tgt_dict, word_reward=0, unk_reward=0):
        super().__init__()
        self.models = models
        self.tgt_dict = tgt_dict
        for i, model in enumerate(self.models):
            model.prepare_for_onnx_export_()
            self._modules[f'model_{i}'] = model
        self.word_reward = word_reward
        self.unk_reward = unk_reward
        vocab_size = len(tgt_dict.indices)
        self.word_rewards = torch.FloatTensor(vocab_size).fill_(word_reward)
        self.word_rewards[tgt_dict.eos()] = 0
        self.word_rewards[tgt_dict.unk()] = word_reward + unk_reward
        self.vocab_size = vocab_size
        self.unk_token = tgt_dict.unk()
        self.enable_precompute_reduced_weights = False

    def forward(self, input_token, target_token, timestep, *inputs):
        """
        Decoder step inputs correspond one-to-one to encoder outputs.
        """
        log_probs_per_model = []
        state_outputs = []
        next_state_input = len(self.models)
        vocab_reduction_module = self.models[0].decoder.vocab_reduction_module
        if vocab_reduction_module is not None:
            possible_translation_tokens = inputs[len(self.models)]
            next_state_input += 1
        else:
            possible_translation_tokens = None
        for i, model in enumerate(self.models):
            encoder_output = inputs[i]
            prev_hiddens = []
            prev_cells = []
            for _ in range(len(model.decoder.layers)):
                prev_hiddens.append(inputs[next_state_input])
                prev_cells.append(inputs[next_state_input + 1])
                next_state_input += 2
            prev_input_feed = inputs[next_state_input].view(1, -1)
            next_state_input += 1
            if self.enable_precompute_reduced_weights and hasattr(model.
                decoder, '_precompute_reduced_weights'
                ) and possible_translation_tokens is not None:
                reduced_output_weights = inputs[next_state_input:
                    next_state_input + 2]
                next_state_input += 2
            else:
                reduced_output_weights = None
            src_length_int = int(encoder_output.size()[0])
            src_length = torch.LongTensor(np.array([src_length_int]))
            src_tokens = torch.LongTensor(np.array([[0] * src_length_int]))
            src_embeddings = encoder_output.new_zeros(encoder_output.shape)
            encoder_out = (encoder_output, prev_hiddens, prev_cells,
                src_length, src_tokens, src_embeddings)
            model.decoder._is_incremental_eval = True
            model.eval()
            incremental_state = {}
            utils.set_incremental_state(model.decoder, incremental_state,
                'cached_state', (prev_hiddens, prev_cells, prev_input_feed))
            decoder_output = model.decoder(input_token.view(1, 1),
                encoder_out, incremental_state=incremental_state,
                possible_translation_tokens=possible_translation_tokens)
            logits, _, _ = decoder_output
            log_probs = F.log_softmax(logits, dim=2)
            log_probs_per_model.append(log_probs)
            next_hiddens, next_cells, next_input_feed = (utils.
                get_incremental_state(model.decoder, incremental_state,
                'cached_state'))
            for h, c in zip(next_hiddens, next_cells):
                state_outputs.extend([h, c])
            state_outputs.append(next_input_feed)
            if reduced_output_weights is not None:
                state_outputs.extend(reduced_output_weights)
        average_log_probs = torch.mean(torch.cat(log_probs_per_model, dim=0
            ), dim=0, keepdim=True)
        if possible_translation_tokens is not None:
            reduced_indices = torch.zeros(self.vocab_size).long().fill_(self
                .unk_token)
            possible_translation_token_range = torch._dim_arange(like=
                possible_translation_tokens, dim=0)
            reduced_indices[possible_translation_tokens
                ] = possible_translation_token_range
            reduced_index = reduced_indices.index_select(dim=0, index=
                target_token)
            score = average_log_probs.view((-1,)).index_select(dim=0, index
                =reduced_index)
        else:
            score = average_log_probs.view((-1,)).index_select(dim=0, index
                =target_token)
        word_reward = self.word_rewards.index_select(0, target_token)
        score += word_reward
        self.input_names = ['prev_token', 'target_token', 'timestep']
        for i in range(len(self.models)):
            self.input_names.append(f'fixed_input_{i}')
        if possible_translation_tokens is not None:
            self.input_names.append('possible_translation_tokens')
        outputs = [score]
        self.output_names = ['score']
        for i in range(len(self.models)):
            self.output_names.append(f'fixed_input_{i}')
            outputs.append(inputs[i])
        if possible_translation_tokens is not None:
            self.output_names.append('possible_translation_tokens')
            outputs.append(possible_translation_tokens)
        for i, state in enumerate(state_outputs):
            outputs.append(state)
            self.output_names.append(f'state_output_{i}')
            self.input_names.append(f'state_input_{i}')
        return tuple(outputs)


class CharSourceEncoderEnsemble(nn.Module):

    def __init__(self, models, src_dict=None):
        super().__init__()
        self.models = models
        self.src_dict = src_dict
        for i, model in enumerate(self.models):
            model.prepare_for_onnx_export_()
            self._modules[f'model_{i}'] = model
        self.enable_precompute_reduced_weights = False

    def forward(self, src_tokens, src_lengths, char_inds, word_lengths):
        outputs = []
        output_names = []
        states = []
        src_tokens_seq_first = src_tokens.t()
        futures = []
        for model in self.models:
            model.eval()
            futures.append(torch.jit._fork(model.encoder,
                src_tokens_seq_first, src_lengths, char_inds, word_lengths))
        vocab_reduction_module = self.models[0].decoder.vocab_reduction_module
        possible_translation_tokens = None
        if vocab_reduction_module is not None:
            possible_translation_tokens = vocab_reduction_module(src_tokens
                =src_tokens, decoder_input_tokens=None)
        reduced_weights = {}
        for i, model in enumerate(self.models):
            if self.enable_precompute_reduced_weights and hasattr(model.
                decoder, '_precompute_reduced_weights'
                ) and possible_translation_tokens is not None:
                reduced_weights[i] = torch.jit._fork(model.decoder.
                    _precompute_reduced_weights, possible_translation_tokens)
        for i, (model, future) in enumerate(zip(self.models, futures)):
            encoder_out = torch.jit._wait(future)
            encoder_outputs = encoder_out[0]
            outputs.append(encoder_outputs)
            output_names.append(f'encoder_output_{i}')
            if hasattr(model.decoder, '_init_prev_states'):
                states.extend(model.decoder._init_prev_states(encoder_out))
            if self.enable_precompute_reduced_weights and hasattr(model.
                decoder, '_precompute_reduced_weights'
                ) and possible_translation_tokens is not None:
                states.extend(torch.jit._wait(reduced_weights[i]))
        if possible_translation_tokens is not None:
            outputs.append(possible_translation_tokens)
            output_names.append('possible_translation_tokens')
        for i, state in enumerate(states):
            outputs.append(state)
            output_names.append(f'initial_state_{i}')
        self.output_names = output_names
        return tuple(outputs)

    @classmethod
    def build_from_checkpoints(cls, checkpoint_filenames, src_dict_filename,
        dst_dict_filename, lexical_dict_paths=None):
        models, src_dict, _ = load_models_from_checkpoints(checkpoint_filenames
            , src_dict_filename, dst_dict_filename, lexical_dict_paths)
        return cls(models, src_dict=src_dict)


class BeamDecode(torch.jit.ScriptModule):
    """
    Decodes the output of Beam Search to get the top hypotheses
    """

    def __init__(self, eos_token_id, length_penalty, nbest, beam_size,
        stop_at_eos):
        super().__init__()
        self.eos_token_id = torch.jit.Attribute(eos_token_id, int)
        self.length_penalty = torch.jit.Attribute(length_penalty, float)
        self.nbest = torch.jit.Attribute(nbest, int)
        self.beam_size = torch.jit.Attribute(beam_size, int)
        self.stop_at_eos = torch.jit.Attribute(int(stop_at_eos), int)

    @torch.jit.script_method
    @torch.no_grad()
    def forward(self, beam_tokens: Tensor, beam_scores: Tensor,
        token_weights: Tensor, beam_prev_indices: Tensor, num_steps: int
        ) ->List[Tuple[Tensor, float, List[float], Tensor, Tensor]]:
        self._check_dimensions(beam_tokens, beam_scores, token_weights,
            beam_prev_indices, num_steps)
        end_states = self._get_all_end_states(beam_tokens, beam_scores,
            beam_prev_indices, num_steps)
        outputs = torch.jit.annotate(List[Tuple[Tensor, float, List[float],
            Tensor, Tensor]], [])
        for state_idx in range(len(end_states)):
            state = end_states[state_idx]
            hypothesis_score = float(state[0])
            beam_indices = self._get_output_steps_to_beam_indices(state,
                beam_prev_indices)
            beam_output = torch.jit.annotate(List[Tensor], [])
            token_level_scores = torch.jit.annotate(List[float], [])
            position = int(state[1])
            hyp_index = int(state[2])
            best_indices = torch.tensor([position, hyp_index])
            back_alignment_weights = []
            assert position + 1 == len(beam_indices)
            pos = 1
            prev_beam_index = -1
            while pos < len(beam_indices):
                beam_index = beam_indices[pos]
                beam_output.append(beam_tokens[pos][beam_index])
                if pos == 1:
                    token_level_scores.append(float(beam_scores[pos][
                        beam_index]))
                else:
                    token_level_scores.append(float(beam_scores[pos][
                        beam_index]) - float(beam_scores[pos - 1][
                        prev_beam_index]))
                back_alignment_weights.append(token_weights[pos][beam_index
                    ].detach())
                prev_beam_index = beam_index
                pos += 1
            outputs.append((torch.stack(beam_output), hypothesis_score,
                token_level_scores, torch.stack(back_alignment_weights, dim
                =1), best_indices))
        return outputs

    @torch.jit.script_method
    def _get_output_steps_to_beam_indices(self, end_state: Tensor,
        beam_prev_indices: Tensor) ->List[int]:
        """
        Returns a mapping from each output position and the beam index that was
        picked from the beam search results.
        """
        present_position = int(end_state[1])
        beam_index = int(end_state[2])
        beam_indices = torch.jit.annotate(List[int], [])
        while present_position >= 0:
            beam_indices.insert(0, beam_index)
            beam_index = int(beam_prev_indices[present_position][beam_index])
            present_position = present_position - 1
        return beam_indices

    @torch.jit.script_method
    def _add_to_end_states(self, end_states: List[Tensor], min_score: float,
        state: Tensor, min_index: int) ->Tuple[List[Tensor], float, int]:
        """
        Maintains a list of atmost `nbest` highest end states
        """
        if len(end_states) < self.nbest:
            end_states.append(state)
            if float(state[0]) <= min_score:
                min_score = float(state[0])
                min_index = len(end_states) - 1
        elif bool(state[0] > min_score):
            end_states[min_index] = state
            min_index = -1
            min_score = float('inf')
            for idx in range(len(end_states)):
                s = end_states[idx]
                if bool(float(s[0]) <= min_score):
                    min_index = idx
                    min_score = float(s[0])
        return end_states, min_score, min_index

    @torch.jit.script_method
    def _get_all_end_states(self, beam_tokens: Tensor, beam_scores: Tensor,
        beam_prev_indices: Tensor, num_steps: int) ->Tensor:
        """
        Return all end states and hypothesis scores for those end states.
        """
        min_score = float('inf')
        min_index = -1
        end_states = torch.jit.annotate(List[Tensor], [])
        prev_hypo_is_finished = torch.zeros(self.beam_size).byte()
        position = 1
        while bool(position <= num_steps):
            hypo_is_finished = torch.zeros(self.beam_size).byte()
            for hyp_index in range(self.beam_size):
                prev_pos = beam_prev_indices[position][hyp_index]
                hypo_is_finished[hyp_index] = prev_hypo_is_finished[prev_pos]
                if bool(hypo_is_finished[hyp_index] == 0):
                    if bool(beam_tokens[position][hyp_index] == self.
                        eos_token_id) or bool(position == num_steps):
                        if bool(self.stop_at_eos):
                            hypo_is_finished[hyp_index] = 1
                        hypo_score = float(beam_scores[position][hyp_index])
                        if bool(self.length_penalty != 0):
                            hypo_score = hypo_score / float(position) ** float(
                                self.length_penalty)
                        end_states, min_score, min_index = (self.
                            _add_to_end_states(end_states, min_score, torch
                            .tensor([hypo_score, float(position), float(
                            hyp_index)]), min_index))
            prev_hypo_is_finished = hypo_is_finished
            position = position + 1
        end_states = torch.stack(end_states)
        _, sorted_end_state_indices = end_states[:, (0)].sort(dim=0,
            descending=True)
        end_states = end_states[(sorted_end_state_indices), :]
        return end_states

    @torch.jit.script_method
    def _check_dimensions(self, beam_tokens: Tensor, beam_scores: Tensor,
        token_weights: Tensor, beam_prev_indices: Tensor, num_steps: int
        ) ->None:
        assert beam_tokens.size(1
            ) == self.beam_size, 'Dimension of beam_tokens : {} and beam size : {} are not consistent'.format(
            beam_tokens.size(), self.beam_size)
        assert beam_scores.size(1
            ) == self.beam_size, 'Dimension of beam_scores : {} and beam size : {} are not consistent'.format(
            beam_scores.size(), self.beam_size)
        assert token_weights.size(1
            ) == self.beam_size, 'Dimension of token_weights : {} and beam size : {} are not consistent'.format(
            token_weights.size(), self.beam_size)
        assert beam_prev_indices.size(1
            ) == self.beam_size, 'Dimension of beam_prev_indices : {} and beam size : {} '
        """are not consistent""".format(beam_prev_indices.size(), self.
            beam_size)
        assert beam_tokens.size(0
            ) <= num_steps + 1, 'Dimension of beam_tokens : {} and num_steps : {} are not consistent'.format(
            beam_tokens.size(), num_steps)
        assert beam_scores.size(0
            ) <= num_steps + 1, 'Dimension of beam_scores : {} and num_steps : {} are not consistent'.format(
            beam_scores.size(), num_steps)
        assert token_weights.size(0
            ) <= num_steps + 1, 'Dimension of token_weights : {} and num_steps : {} are not consistent'.format(
            token_weights.size(), num_steps)
        assert beam_prev_indices.size(0
            ) <= num_steps + 1, 'Dimension of beam_prev_indices : {} and num_steps : {} are not consistent'.format(
            beam_prev_indices.size(), num_steps)


class BeamSearchAndDecode(torch.jit.ScriptModule):
    """
    Combines the functionality of BeamSearch and BeamDecode
    """

    def __init__(self, models, tgt_dict, src_tokens, src_lengths,
        eos_token_id, length_penalty, nbest, beam_size, stop_at_eos,
        word_reward=0, unk_reward=0, quantize=False):
        super().__init__()
        self.beam_search = BeamSearch(models, tgt_dict, src_tokens,
            src_lengths, beam_size, word_reward, unk_reward, quantize)
        self.beam_decode = BeamDecode(eos_token_id, length_penalty, nbest,
            beam_size, stop_at_eos)
        self.input_names = ['src_tokens', 'src_lengths', 'prev_token',
            'prev_scores', 'attn_weights', 'prev_hypos_indices', 'num_steps']
        self.output_names = ['beam_output', 'hypothesis_score',
            'token_level_scores', 'back_alignment_weights', 'best_indices']

    @torch.jit.script_method
    def forward(self, src_tokens: torch.Tensor, src_lengths: torch.Tensor,
        prev_token: torch.Tensor, prev_scores: torch.Tensor, attn_weights:
        torch.Tensor, prev_hypos_indices: torch.Tensor, num_steps: int) ->List[
        Tuple[Tensor, float, List[float], Tensor, Tensor]]:
        beam_search_out = self.beam_search(src_tokens, src_lengths,
            prev_token, prev_scores, attn_weights, prev_hypos_indices,
            num_steps)
        all_tokens, all_scores, all_weights, all_prev_indices = beam_search_out
        outputs = torch.jit.annotate(List[Tuple[Tensor, float, List[float],
            Tensor, Tensor]], [])
        outputs = self.beam_decode(all_tokens, all_scores, all_weights,
            all_prev_indices, num_steps)
        return outputs

    @classmethod
    def build_from_checkpoints(cls, checkpoint_filenames, src_dict_filename,
        dst_dict_filename, beam_size, length_penalty, nbest, word_reward=0,
        unk_reward=0, lexical_dict_paths=None):
        length = 10
        models, _, tgt_dict = load_models_from_checkpoints(checkpoint_filenames
            , src_dict_filename, dst_dict_filename, lexical_dict_paths)
        src_tokens = torch.LongTensor(np.ones((length, 1), dtype='int64'))
        src_lengths = torch.IntTensor(np.array([length], dtype='int32'))
        eos_token_id = tgt_dict.eos()
        return cls(models, tgt_dict, src_tokens, src_lengths, eos_token_id,
            length_penalty=length_penalty, nbest=nbest, beam_size=beam_size,
            stop_at_eos=True, word_reward=word_reward, unk_reward=
            unk_reward, quantize=True)

    def save_to_pytorch(self, output_path):

        def pack(s):
            if hasattr(s, '_pack'):
                s._pack()

        def unpack(s):
            if hasattr(s, '_unpack'):
                s._unpack()
        self.apply(pack)
        torch.jit.save(self, output_path)
        self.apply(unpack)


class IterativeRefinementGenerateAndDecode(torch.jit.ScriptModule):

    def __init__(self, models, tgt_dict, max_iter=1, quantize=True,
        check_trace=True):
        super().__init__()
        src_tokens = torch.tensor([[4, 2]])
        src_lengths = torch.tensor([2])
        self.models = models
        generator = IterativeRefinementGenerator(self.models, tgt_dict,
            max_iter=max_iter)
        if quantize:
            generator = torch.quantization.quantize_dynamic(generator, {
                torch.nn.Linear}, dtype=torch.qint8, inplace=True)
        enc_inputs = src_tokens, src_lengths
        self.generator = torch.jit.trace(generator, enc_inputs,
            _force_outplace=True, check_trace=check_trace)

    @torch.jit.script_method
    def forward(self, src_tokens: torch.Tensor, src_lengths: torch.Tensor
        ) ->List[Tuple[Tensor, float, Tensor]]:
        return [(x.long(), float(y), at) for x, y, at in list(self.
            generator(src_tokens.t(), src_lengths))]

    def save_to_pytorch(self, output_path):

        def pack(s):
            if hasattr(s, '_pack'):
                s._pack()

        def unpack(s):
            if hasattr(s, '_unpack'):
                s._unpack()
        self.apply(pack)
        torch.jit.save(self, output_path)
        self.apply(unpack)

    @classmethod
    def build_from_checkpoints(cls, checkpoint_filenames, src_dict_filename,
        tgt_dict_filename, lexical_dict_paths=None, max_iter=1):
        models, _, tgt_dict = load_models_from_checkpoints(checkpoint_filenames
            , src_dict_filename, tgt_dict_filename, lexical_dict_paths)
        return cls(models, tgt_dict=tgt_dict, max_iter=max_iter)


@torch.jit.script
def finalize_hypos_loop_attns(finalized_attns_list: List[Tensor],
    finalized_alignments_list: List[Tensor], finalized_idxs, pad_idx: int,
    finalized_tokens, finalized_scores, finalized_attn):
    for i in range(finalized_idxs.size(0)):
        cutoff = finalized_tokens[i].ne(pad_idx)
        hypo_attn = finalized_attn[i][cutoff]
        alignment = hypo_attn.max(dim=1)[1]
        finalized_attns_list[finalized_idxs[i]] = hypo_attn
        finalized_alignments_list[finalized_idxs[i]] = alignment
    return finalized_attns_list, finalized_alignments_list


@torch.jit.script
def finalize_hypos_loop_scores(finalized_scores_list: List[Tensor],
    finalized_idxs, pad_idx: int, finalized_tokens, finalized_scores):
    for i in range(finalized_idxs.size(0)):
        cutoff = finalized_scores[i].ne(pad_idx)
        scores = finalized_scores[i][cutoff]
        finalized_scores_list[finalized_idxs[i]] = scores
    return finalized_scores_list


@torch.jit.script
def finalize_hypos_loop_tokens(finalized_tokens_list: List[Tensor],
    finalized_idxs, pad_idx: int, finalized_tokens, finalized_scores):
    for i in range(finalized_idxs.size(0)):
        cutoff = finalized_tokens[i].ne(pad_idx)
        tokens = finalized_tokens[i][cutoff]
        finalized_tokens_list[finalized_idxs[i]] = tokens
    return finalized_tokens_list


@torch.jit.script
def is_a_loop(pad_idx: int, x, y, s, a):
    b, l_x, l_y = x.size(0), x.size(1), y.size(1)
    if l_x > l_y:
        y = torch.cat([y, torch.zeros([b, l_x - l_y]).to(y).fill_(pad_idx)], 1)
        s = torch.cat([s, torch.zeros([b, l_x - l_y]).to(s)], 1)
        if a.size()[0] > 0:
            a = torch.cat([a, torch.zeros([b, l_x - l_y, a.size(2)]).to(a)], 1)
    elif l_x < l_y:
        x = torch.cat([x, torch.zeros([b, l_y - l_x]).to(x).fill_(pad_idx)], 1)
    return (x == y).all(1), y, s, a


@torch.jit.script
def last_step(step: int, max_iter: int, terminated):
    if step == max_iter:
        terminated.fill_(1)
    return terminated


class IterativeRefinementGenerator(nn.Module):

    def __init__(self, models, tgt_dict, eos_penalty=0.0, max_iter=2,
        max_ratio=2, decoding_format=None, retain_dropout=False, adaptive=True
        ):
        """
        Generates translations based on iterative refinement.

        Args:
            tgt_dict: target dictionary
        """
        super().__init__()
        self.models = models
        self.bos = tgt_dict.bos()
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.eos_penalty = eos_penalty
        self.max_iter = max_iter
        self.max_ratio = max_ratio
        self.decoding_format = decoding_format
        self.retain_dropout = retain_dropout
        self.adaptive = adaptive
        for i, model in enumerate(self.models):
            model.prepare_for_onnx_export_()
            model.eval()
            if hasattr(model, 'get_student_model'):
                model = model.get_student_model()
                self.models[i] = model
            self._modules[f'model_{i}'] = model

    def forward(self, src_tokens: torch.Tensor, src_lengths: torch.Tensor
        ) ->Tuple[Tuple[Tensor, Tensor, Tensor]]:
        o1, o2, o3, _ = self.generate(self.models, src_tokens, src_lengths)
        return tuple((x, y.float().mean(), z) for x, y, z in zip(o1, o2, o3))

    @torch.no_grad()
    def generate(self, models, src_tokens, src_lengths, prefix_tokens=None):
        assert len(models) == 1, 'only support single model'
        model = models[0]
        bsz, src_len = src_tokens.size()
        sent_idxs = torch.arange(bsz)
        encoder_out = model.encoder(src_tokens, src_lengths)
        prev_decoder_out = model.initialize_output_tokens(encoder_out,
            src_tokens)
        prev_output_tokens = prev_decoder_out.output_tokens.clone()
        finalized_tokens_list = [torch.tensor(0) for _ in range(bsz)]
        finalized_scores_list = [torch.tensor(0) for _ in range(bsz)]
        finalized_attns_list = [torch.tensor(0) for _ in range(bsz)]
        finalized_alignments_list = [torch.tensor(0) for _ in range(bsz)]
        for step in range(self.max_iter + 1):
            prev_decoder_out = prev_decoder_out._replace(step=step,
                max_step=self.max_iter + 1)
            decoder_out = model.forward_decoder(prev_decoder_out,
                encoder_out, eos_penalty=self.eos_penalty, max_ratio=self.
                max_ratio, decoding_format=self.decoding_format)
            terminated, output_tokens, output_scores, output_attn = is_a_loop(
                self.pad, prev_output_tokens, decoder_out.output_tokens,
                decoder_out.output_scores, decoder_out.attn)
            decoder_out = decoder_out._replace(output_tokens=output_tokens,
                output_scores=output_scores, attn=output_attn)
            terminated = last_step(step, self.max_iter, terminated)
            finalized_idxs = sent_idxs[terminated]
            finalized_tokens = decoder_out.output_tokens[terminated]
            finalized_scores = decoder_out.output_scores[terminated]
            finalized_attn = (None if decoder_out.attn is None else
                decoder_out.attn[terminated])
            finalized_tokens_list = finalize_hypos_loop_tokens(
                finalized_tokens_list, finalized_idxs, self.pad,
                finalized_tokens, finalized_scores)
            finalized_scores_list = finalize_hypos_loop_scores(
                finalized_scores_list, finalized_idxs, self.pad,
                finalized_tokens, finalized_scores)
            finalized_attns_list, finalized_alignments_list = (
                finalize_hypos_loop_attns(finalized_attns_list,
                finalized_alignments_list, finalized_idxs, self.pad,
                finalized_tokens, finalized_scores, finalized_attn))
            not_terminated = ~terminated
            prev_decoder_out = decoder_out._replace(output_tokens=
                script_skip_tensor(decoder_out.output_tokens,
                not_terminated), output_scores=script_skip_tensor(
                decoder_out.output_scores, not_terminated), attn=
                decoder_out.attn, step=decoder_out.step, max_step=
                decoder_out.max_step)
            encoder_out = EncoderOut(encoder_out=script_skip_tensor(
                encoder_out.encoder_out, ~terminated), encoder_padding_mask
                =None, encoder_embedding=script_skip_tensor(encoder_out.
                encoder_embedding, ~terminated), encoder_states=None,
                src_tokens=None, src_lengths=None)
            sent_idxs = script_skip_tensor(sent_idxs, not_terminated)
            prev_output_tokens = prev_decoder_out.output_tokens.clone()
        return (finalized_tokens_list, finalized_scores_list,
            finalized_attns_list, finalized_alignments_list)


class MultiDecoderCombinationStrategy(nn.Module):
    """Strategy for combining decoder networks.

    This is an abstract strategy (GoF) which defines the mapping from multiple
    (unprojected) decoder outputs to the fully expanded logits.
    """

    def __init__(self, out_embed_dims, vocab_size, vocab_reduction_module=None
        ):
        super().__init__()
        self.out_embed_dims = out_embed_dims
        self.vocab_size = vocab_size
        self.vocab_reduction_module = vocab_reduction_module

    @abc.abstractmethod
    def forward(self, unprojected_outs, src_tokens=None, input_tokens=None,
        possible_translation_tokens=None, select_single=None):
        """Combine decoder outputs and project.

        Args:
            unprojected_outs (list): List of tensors with the same length as
                self.out_embed_dims containing the unprojected decoder outputs
                from each decoder network.
            src_tokens (Tensor): Tensor with source sentence tokens for vocab
                reduction.
            input_tokens (Tensor): Tensor with target-side decoder input tokens
                for vocab reduction.
            possible_translation_tokens: For vocab reduction.
            select_single (None or int): Only use the n-th decoder output.

        Return:
            A tuple (logits, possible_translation_tokens), where logits is a
            [batch_size, seq_len, vocab_size] tensor with the final combined
            output logits, and possible_translation_tokens the short list from
            vocab reduction.
        """
        raise NotImplementedError()


def combine_heads(X):
    """
    Combine heads (the inverse of split heads):
    1) Transpose X from (batch size, nheads, sequence length, d_head) to
        (batch size, sequence length, nheads, d_head)
    2) Combine (reshape) last 2 dimensions (nheads, d_head) into 1 (d_model)

    Inputs:
      X : [batch size * nheads, sequence length, d_head]
      nheads : integer
      d_head : integer

    Outputs:
      [batch_size, seq_len, d_model]

    """
    X = X.transpose(1, 2)
    nheads, d_head = X.shape[-2:]
    return X.contiguous().view(list(X.shape[:-2]) + [nheads * d_head])


def create_src_lengths_mask(batch_size, src_lengths):
    max_srclen = src_lengths.max()
    src_indices = torch.arange(0, max_srclen).unsqueeze(0).type_as(src_lengths)
    src_indices = src_indices.expand(batch_size, max_srclen)
    src_lengths = src_lengths.unsqueeze(dim=1).expand(batch_size, max_srclen)
    return (src_indices < src_lengths).int().detach()


def apply_masks(scores, batch_size, unseen_mask, src_lengths):
    seq_len = scores.shape[-1]
    sequence_mask = torch.ones(seq_len, seq_len).unsqueeze(0).int()
    if unseen_mask:
        sequence_mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=0
            ).unsqueeze(0).int()
    if src_lengths is not None:
        src_lengths_mask = create_src_lengths_mask(batch_size=batch_size,
            src_lengths=src_lengths).unsqueeze(-2)
        sequence_mask = sequence_mask & src_lengths_mask
    sequence_mask = sequence_mask.unsqueeze(1)
    scores = scores.masked_fill(sequence_mask == 0, -np.inf)
    return scores


def scaled_dot_prod_attn(query, key, value, unseen_mask=False, src_lengths=None
    ):
    """
    Scaled Dot Product Attention

    Implements equation:
    Attention(Q, K, V) = softmax(QK^T/\\sqrt{d_k})V

    Inputs:
      query : [batch size, nheads, sequence length, d_k]
      key : [batch size, nheads, sequence length, d_k]
      value : [batch size, nheads, sequence length, d_v]
      unseen_mask: if True, only attend to previous sequence positions
      src_lengths_mask: if True, mask padding based on src_lengths

    Outputs:
      attn: [batch size, sequence length, d_v]

    Note that in this implementation d_q = d_k = d_v = dim
    """
    d_k = query.shape[-1]
    scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(d_k)
    if unseen_mask or src_lengths is not None:
        scores = apply_masks(scores=scores, batch_size=query.shape[0],
            unseen_mask=unseen_mask, src_lengths=src_lengths)
    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn


def split_heads(X, nheads):
    """
    Split heads:
    1) Split (reshape) last dimension (size d_model) into nheads, d_head
    2) Transpose X from (batch size, sequence length, nheads, d_head) to
        (batch size, nheads, sequence length, d_head)

    Inputs:
      X : [batch size, sequence length, nheads * d_head]
      nheads : integer
    Outputs:
      [batch size,  nheads, sequence length, d_head]

    """
    last_dim = X.shape[-1]
    assert last_dim % nheads == 0
    X_last_dim_split = X.view(list(X.shape[:-1]) + [nheads, last_dim // nheads]
        )
    return X_last_dim_split.transpose(1, 2)


class MultiheadAttention(nn.Module):
    """
    Multiheaded Scaled Dot Product Attention

    Implements equation:
    MultiHead(Q, K, V) = Concat(head_1,...,head_h)W^O
        where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

    Similarly to the above, d_k = d_v = d_model / h

    Inputs
      init:
        nheads : integer # of attention heads
        d_model : model dimensionality
        d_head : dimensionality of a single head

      forward:
        query : [batch size, sequence length, d_model]
        key: [batch size, sequence length, d_model]
        value: [batch size, sequence length, d_model]
      unseen_mask: if True, only attend to previous sequence positions
      src_lengths_mask: if True, mask padding based on src_lengths

    Output
      result : [batch_size, sequence length, d_model]
    """

    def __init__(self, nheads, d_model):
        """Take in model size and number of heads."""
        super(MultiheadAttention, self).__init__()
        assert d_model % nheads == 0
        self.d_head = d_model // nheads
        self.nheads = nheads
        self.Q_fc = nn.Linear(d_model, d_model, bias=False)
        self.K_fc = nn.Linear(d_model, d_model, bias=False)
        self.V_fc = nn.Linear(d_model, d_model, bias=False)
        self.output_fc = nn.Linear(d_model, d_model, bias=False)
        self.attn = None

    def forward(self, query, key, value, unseen_mask=False, src_lengths=None):
        query = split_heads(self.Q_fc(query), self.nheads)
        key = split_heads(self.K_fc(key), self.nheads)
        value = split_heads(self.V_fc(value), self.nheads)
        x, self.attn = scaled_dot_prod_attn(query=query, key=key, value=
            value, unseen_mask=unseen_mask, src_lengths=src_lengths)
        x = combine_heads(x)
        return self.output_fc(x)


def LayerNorm(embedding_dim):
    m = nn.LayerNorm(embedding_dim)
    return m


class TransformerDecoderLayerPhase2(nn.Module):
    """Second phase of decoder layer block
    This layer will take the input from the ecoder and phirst pass decoder.
    papers.nips.cc/paper/6775-deliberation-networks-sequence-generation-beyond-one-pass-decoding.pdf
    """

    def __init__(self, args, no_encoder_decoder_attn=False, add_bias_kv=
        False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.self_attn = MultiheadAttention(embed_dim=self.embed_dim,
            num_heads=args.decoder_attention_heads, dropout=args.
            attention_dropout, add_bias_kv=add_bias_kv, add_zero_attn=
            add_zero_attn, self_attention=True)
        self.dropout = args.dropout
        self.activation_fn = utils.get_activation_fn(activation=getattr(
            args, 'activation_fn', 'relu'))
        self.activation_dropout = getattr(args, 'activation_dropout', 0)
        if self.activation_dropout == 0:
            self.activation_dropout = getattr(args, 'relu_dropout', 0)
        self.normalize_before = args.decoder_normalize_before
        export = getattr(args, 'char_inputs', False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
        if no_encoder_decoder_attn:
            self.encoder_attn = None
            self.decoder_attn = None
            self.encoder_layer_norm = None
            self.decoder_layer_norm = None
        else:
            self.encoder_attn = MultiheadAttention(self.embed_dim, args.
                decoder_attention_heads, dropout=args.attention_dropout,
                encoder_decoder_attention=True)
            self.decoder_attn = MultiheadAttention(self.embed_dim, args.
                decoder_attention_heads, dropout=args.attention_dropout,
                encoder_decoder_attention=True)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export
                =export)
            self.decoder_attn_layer_norm = LayerNorm(self.embed_dim, export
                =export)
        self.fc1 = Linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = Linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True
        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def forward(self, x, encoder_out=None, encoder_padding_mask=None,
        decoder_out=None, incremental_state=None, prev_self_attn_state=None,
        prev_encoder_attn_state=None, prev_decoder_attn_state=None,
        self_attn_mask=None, self_attn_padding_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        residual = x
        x_self_attention = self.maybe_layer_norm(self.self_attn_layer_norm,
            x, before=True)
        if prev_self_attn_state is not None:
            if incremental_state is None:
                incremental_state = {}
            prev_key, prev_value = prev_self_attn_state
            saved_state = {'prev_key': prev_key, 'prev_value': prev_value}
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        x_self_attention, attn = self.self_attn(query=x, key=x, value=x,
            key_padding_mask=self_attn_padding_mask, incremental_state=
            incremental_state, need_weights=False, attn_mask=self_attn_mask)
        x_self_attention = F.dropout(x_self_attention, p=self.dropout,
            training=self.training)
        x_self_attention = residual + x_self_attention
        x_self_attention = self.maybe_layer_norm(self.self_attn_layer_norm,
            x_self_attention, after=True)
        if self.encoder_attn is not None:
            residual = x
            x_encoder_attention = self.maybe_layer_norm(self.
                encoder_attn_layer_norm, x, before=True)
            if prev_encoder_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_encoder_attn_state
                saved_state = {'prev_key': prev_key, 'prev_value': prev_value}
                self.encoder_attn._set_input_buffer(incremental_state,
                    saved_state)
            x_encoder_attention, attn = self.encoder_attn(query=
                x_encoder_attention, key=encoder_out, value=encoder_out,
                key_padding_mask=encoder_padding_mask, incremental_state=
                incremental_state, static_kv=True, need_weights=not self.
                training and self.need_attn)
            x_encoder_attention = F.dropout(x_encoder_attention, p=self.
                dropout, training=self.training)
            x_encoder_attention = residual + x_encoder_attention
            x_encoder_attention = self.maybe_layer_norm(self.
                encoder_attn_layer_norm, x_encoder_attention, after=True)
        if self.decoder_attn is not None:
            residual = x
            x_decoder_attention = self.maybe_layer_norm(self.
                decoder_attn_layer_norm, x, before=True)
            if prev_decoder_attn_state is not None:
                if incremental_state is None:
                    incremental_state = {}
                prev_key, prev_value = prev_decoder_attn_state
                saved_state = {'prev_key': prev_key, 'prev_value': prev_value}
                self.encoder_attn._set_input_buffer(incremental_state,
                    saved_state)
            x_decoder_attention, attn = self.decoder_attn(query=
                x_decoder_attention, key=decoder_out, value=decoder_out,
                incremental_state=incremental_state, static_kv=True,
                need_weights=not self.training and self.need_attn)
            x_decoder_attention = F.dropout(x_decoder_attention, p=self.
                dropout, training=self.training)
            x_decoder_attention = residual + x_decoder_attention
            x_decoder_attention = self.maybe_layer_norm(self.
                encoder_attn_layer_norm, x_decoder_attention, after=True)
        x = x_self_attention + x_encoder_attention + x_decoder_attention
        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            self_attn_state = saved_state['prev_key'], saved_state['prev_value'
                ]
            return x, attn, self_attn_state
        return x, attn

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


class MultiSourceSequenceGenerator(torch.nn.Module):
    align_to = 0

    def __init__(self, models, tgt_dict, beam_size=1, minlen=1, maxlen=None,
        stop_early=True, normalize_scores=True, len_penalty=0, unk_reward=0,
        lexicon_reward=0, retain_dropout=False, word_reward=0,
        model_weights=None, use_char_source=False, align_to=1):
        """Generates translations from multiple source sentences

        This only supports one model for now.

        Args:
            models: List of FairseqModel objects. Each one must implement
                expand_encoder_output() method to replicate encoder outputs.
                For now only one model is supported
            min/maxlen: The length of the generated output will be bounded by
                minlen and maxlen (not including the end-of-sentence marker).
            stop_early: Stop generation immediately after we finalize beam_size
                hypotheses, even though longer hypotheses might have better
                normalized scores.
            normalize_scores: Normalize scores by the length of the output.
            word_reward: add this value to score each token except EOS
                (an alternative method to len_penalty for encouraging longer
                output)
            model_weights: None or list of Python floats of the same length as
                `models` with ensemble interpolation weights.
            use_char_source: if True, encoder inputs consist of (src_tokens,
                src_lengths, char_inds, word_lengths)
        """
        self.models = models
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        self.minlen = minlen
        max_decoder_len = min(m.max_decoder_positions() for m in self.models)
        self.maxlen = max_decoder_len if maxlen is None else min(maxlen,
            max_decoder_len)
        self.stop_early = stop_early
        self.normalize_scores = normalize_scores
        self.len_penalty = len_penalty
        self.unk_reward = unk_reward
        self.lexicon_reward = lexicon_reward
        self.lexicon_indices = tgt_dict.lexicon_indices_list()
        self.retain_dropout = retain_dropout
        self.word_reward = word_reward
        if model_weights is not None:
            assert len(models) == len(model_weights)
            self.model_weights = model_weights
        else:
            self.model_weights = [1.0 / len(models)] * len(models)
        self.use_char_source = use_char_source

    def cuda(self):
        for model in self.models:
            model
        return self

    def generate_batched_itr(self, data_itr, beam_size=None, maxlen_a=0.0,
        maxlen_b=None, cuda=False, timer=None, prefix_size=0):
        """Iterate over a batched dataset and yield individual translations.

        Args:
            maxlen_a/b: generate sequences of maximum length ax + b,
                where x is the source sentence length.
            cuda: use GPU for generation
            timer: StopwatchMeter for timing generations.
        """
        if maxlen_b is None:
            maxlen_b = self.maxlen
        for sample in data_itr:
            if cuda:
                s = utils.move_to_cuda(sample)
            input = s['net_input']
            srclen = input['src_tokens'].size(1)
            if self.use_char_source:
                raise ValueError(
                    'Character level encoder is not supported yet for multisource sentences.'
                    )
            encoder_inputs = input['src_tokens'], input['src_lengths']
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(encoder_inputs, srcs_ids=input[
                    'src_ids'], beam_size=beam_size, maxlen=int(maxlen_a *
                    srclen + maxlen_b), prefix_tokens=s['target'][:, :
                    prefix_size] if prefix_size > 0 else None)
            if timer is not None:
                timer.stop(s['ntokens'])
            for i, id in enumerate(s['id']):
                src = input['src_tokens'].index_select(0, input['src_ids'][
                    self.align_to])
                ref = utils.strip_pad(s['target'][(i), :], self.pad)
                yield id, src, ref, hypos[i]

    def generate(self, encoder_inputs, srcs_ids, beam_size=None, maxlen=
        None, prefix_tokens=None, src_weights=None):
        """Generate a batch of translations."""
        with torch.no_grad():
            return self._generate(encoder_inputs, srcs_ids, beam_size,
                maxlen, prefix_tokens, src_weights)

    def _generate(self, encoder_inputs, srcs_ids, beam_size=None, maxlen=
        None, prefix_tokens=None, src_weights=None):
        """Generates a translation from multiple source sentences"""
        n_srcs = len(srcs_ids)
        srcs_tokens = encoder_inputs[0]
        align_src_tokens = srcs_tokens.index_select(0, srcs_ids[self.align_to])
        bsz, srclen = align_src_tokens.size()
        maxlen = min(maxlen, self.maxlen
            ) if maxlen is not None else self.maxlen
        beam_size = beam_size if beam_size is not None else self.beam_size
        assert beam_size < self.vocab_size, 'Beam size must be smaller than target vocabulary'
        encoder_outs = self._encode(encoder_inputs, beam_size, srcs_ids)
        incremental_states = self._init_incremental_states(n_srcs)
        scores = align_src_tokens.new(bsz * beam_size, maxlen + 1).float(
            ).fill_(0)
        scores_buf = scores.clone()
        tokens = align_src_tokens.new(bsz * beam_size, maxlen + 2).fill_(self
            .pad)
        tokens_buf = tokens.clone()
        tokens[:, (0)] = self.eos
        src_encoding_len = encoder_outs[self.align_to][0][0].size(0)
        attn = scores.new(bsz * beam_size, src_encoding_len, maxlen + 2)
        attn_buf = attn.clone()
        finalized = [[] for i in range(bsz)]
        finished = [(False) for i in range(bsz)]
        worst_finalized = [{'idx': None, 'score': -math.inf} for i in range
            (bsz)]
        num_remaining_sent = bsz
        cand_size = 2 * beam_size
        bbsz_offsets = (torch.arange(0, bsz) * beam_size).unsqueeze(1).type_as(
            tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)
        buffers = {}

        def buffer(name, type_of=tokens):
            if name not in buffers:
                buffers[name] = type_of.new()
            return buffers[name]

        def is_finished(sent, step, unfinalized_scores=None):
            """
            Check whether we've finished generation for a given sentence, by
            comparing the worst score among finalized hypotheses to the best
            possible score among unfinalized hypotheses.
            """
            assert len(finalized[sent]) <= beam_size
            if len(finalized[sent]) == beam_size:
                if (self.stop_early or step == maxlen or unfinalized_scores is
                    None):
                    return True
                best_unfinalized_score = unfinalized_scores[sent].max()
                if self.normalize_scores:
                    best_unfinalized_score /= (maxlen + 1) ** self.len_penalty
                if worst_finalized[sent]['score'] >= best_unfinalized_score:
                    return True
            return False

        def finalize_hypos(step, bbsz_idx, eos_scores, unfinalized_scores=None
            ):
            """
            Finalize the given hypotheses at this step, while keeping the total
            number of finalized hypotheses per sentence <= beam_size.

            Note: the input must be in the desired finalization order, so that
            hypotheses that appear earlier in the input are preferred to those
            that appear later.

            Args:
                step: current time step
                bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                    indicating which hypotheses to finalize
                eos_scores: A vector of the same size as bbsz_idx containing
                    scores for each hypothesis
                unfinalized_scores: A vector containing scores for all
                    unfinalized hypotheses
            """
            assert bbsz_idx.numel() == eos_scores.numel()
            tokens_clone = tokens.index_select(0, bbsz_idx)
            tokens_clone = tokens_clone[:, 1:step + 2]
            tokens_clone[:, (step)] = self.eos
            attn_clone = attn.index_select(0, bbsz_idx)[:, :, 1:step + 2]
            pos_scores = scores.index_select(0, bbsz_idx)[:, :step + 1]
            pos_scores[:, (step)] = eos_scores
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]
            if self.normalize_scores:
                eos_scores /= (step + 1) ** self.len_penalty
            sents_seen = set()
            for i, (idx, score) in enumerate(zip(bbsz_idx.tolist(),
                eos_scores.tolist())):
                sent = idx // beam_size
                sents_seen.add(sent)

                def get_hypo():
                    _, alignment = attn_clone[i].max(dim=0)
                    return {'tokens': tokens_clone[i], 'score': score,
                        'attention': attn_clone[i], 'alignment': alignment,
                        'positional_scores': pos_scores[i]}
                if len(finalized[sent]) < beam_size:
                    finalized[sent].append(get_hypo())
                elif not self.stop_early and score > worst_finalized[sent][
                    'score']:
                    worst_idx = worst_finalized[sent]['idx']
                    if worst_idx is not None:
                        finalized[sent][worst_idx] = get_hypo()
                    idx, s = min(enumerate(finalized[sent]), key=lambda r:
                        r[1]['score'])
                    worst_finalized[sent] = {'score': s['score'], 'idx': idx}
            num_finished = 0
            for sent in sents_seen:
                if not finished[sent] and is_finished(sent, step,
                    unfinalized_scores):
                    finished[sent] = True
                    num_finished += 1
            return num_finished
        reorder_state = None
        for step in range(maxlen + 1):
            if reorder_state is not None:
                for model_id, model in enumerate(self.models):
                    if isinstance(model.decoder, FairseqIncrementalDecoder):
                        for src_id in range(n_srcs):
                            model.decoder.reorder_incremental_state(
                                incremental_states[src_id, model_id],
                                reorder_state)
            logprobs, avg_attn, possible_translation_tokens = self._decode(
                tokens[:, :step + 1], encoder_outs, incremental_states, n_srcs)
            if step == 0:
                logprobs = logprobs.unfold(0, 1, beam_size).squeeze(2
                    ).contiguous()
                scores = scores.type_as(logprobs)
                scores_buf = scores_buf.type_as(logprobs)
            else:
                logprobs.add_(scores[:, (step - 1)].view(-1, 1))
            logprobs[:, (self.pad)] = -math.inf
            if possible_translation_tokens is None:
                unk_index = self.unk
            else:
                unk_index = torch.nonzero(possible_translation_tokens ==
                    self.unk)[0, 0]
            logprobs[:, (unk_index)] += self.unk_reward
            logprobs[:, (self.lexicon_indices)] += self.lexicon_reward
            logprobs += self.word_reward
            logprobs[:, (self.eos)] -= self.word_reward
            attn[:, :, (step + 1)].copy_(avg_attn)
            cand_scores = buffer('cand_scores', type_of=scores)
            cand_indices = buffer('cand_indices')
            cand_beams = buffer('cand_beams')
            eos_bbsz_idx = buffer('eos_bbsz_idx')
            eos_scores = buffer('eos_scores', type_of=scores)
            if step < maxlen:
                if prefix_tokens is not None and step < prefix_tokens.size(1):
                    logprobs_slice = logprobs.view(bsz, -1, logprobs.size(-1))[
                        :, (0), :]
                    cand_scores = torch.gather(logprobs_slice, dim=1, index
                        =prefix_tokens[:, (step)].view(-1, 1)).expand(-1,
                        cand_size)
                    cand_indices = prefix_tokens[:, (step)].view(-1, 1).expand(
                        bsz, cand_size)
                    cand_beams.resize_as_(cand_indices).fill_(0)
                else:
                    torch.topk(logprobs.view(bsz, -1), k=min(cand_size, 
                        logprobs.view(bsz, -1).size(1) - 1), out=(
                        cand_scores, cand_indices))
                    possible_tokens_size = self.vocab_size
                    if possible_translation_tokens is not None:
                        possible_tokens_size = (possible_translation_tokens
                            .size(0))
                    torch.div(cand_indices, possible_tokens_size, out=
                        cand_beams)
                    cand_indices.fmod_(possible_tokens_size)
                    if possible_translation_tokens is not None:
                        possible_translation_tokens = (
                            possible_translation_tokens.view(1,
                            possible_tokens_size).expand(cand_indices.size(
                            0), possible_tokens_size))
                        cand_indices = torch.gather(possible_translation_tokens
                            , dim=1, index=cand_indices, out=cand_indices)
            else:
                torch.sort(logprobs[:, (self.eos)], descending=True, out=(
                    eos_scores, eos_bbsz_idx))
                num_remaining_sent -= finalize_hypos(step, eos_bbsz_idx,
                    eos_scores)
                assert num_remaining_sent == 0
                break
            cand_bbsz_idx = cand_beams.add_(bbsz_offsets)
            eos_mask = cand_indices.eq(self.eos)
            if step >= self.minlen:
                torch.masked_select(cand_bbsz_idx[:, :beam_size], mask=
                    eos_mask[:, :beam_size], out=eos_bbsz_idx)
                if eos_bbsz_idx.numel() > 0:
                    torch.masked_select(cand_scores[:, :beam_size], mask=
                        eos_mask[:, :beam_size], out=eos_scores)
                    num_remaining_sent -= finalize_hypos(step, eos_bbsz_idx,
                        eos_scores, cand_scores)
            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            assert step < maxlen
            active_mask = buffer('active_mask')
            torch.add(eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[:eos_mask.size(1)], out=active_mask)
            active_hypos, _ignore = buffer('active_hypos'), buffer('_ignore')
            torch.topk(active_mask, k=beam_size, dim=1, largest=False, out=
                (_ignore, active_hypos))
            active_bbsz_idx = buffer('active_bbsz_idx')
            torch.gather(cand_bbsz_idx, dim=1, index=active_hypos, out=
                active_bbsz_idx)
            active_scores = torch.gather(cand_scores, dim=1, index=
                active_hypos, out=scores[:, (step)].view(bsz, beam_size))
            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)
            torch.index_select(tokens[:, :step + 1], dim=0, index=
                active_bbsz_idx, out=tokens_buf[:, :step + 1])
            torch.gather(cand_indices, dim=1, index=active_hypos, out=
                tokens_buf.view(bsz, beam_size, -1)[:, :, (step + 1)])
            if step > 0:
                torch.index_select(scores[:, :step], dim=0, index=
                    active_bbsz_idx, out=scores_buf[:, :step])
            torch.gather(cand_scores, dim=1, index=active_hypos, out=
                scores_buf.view(bsz, beam_size, -1)[:, :, (step)])
            torch.index_select(attn[:, :, :step + 2], dim=0, index=
                active_bbsz_idx, out=attn_buf[:, :, :step + 2])
            tokens, tokens_buf = tokens_buf, tokens
            scores, scores_buf = scores_buf, scores
            attn, attn_buf = attn_buf, attn
            reorder_state = active_bbsz_idx
        for sent in range(bsz):
            finalized[sent] = sorted(finalized[sent], key=lambda r: r[
                'score'], reverse=True)
        return finalized

    def _init_incremental_states(self, n_srcs):
        incremental_states = {}
        for src_id in range(n_srcs):
            for model_id, model in enumerate(self.models):
                if isinstance(model.decoder, FairseqIncrementalDecoder):
                    incremental_states[src_id, model_id] = {}
                else:
                    incremental_states[src_id, model_id] = None
        return incremental_states

    def _encode(self, encoder_inputs, beam_size, srcs_ids):
        encoder_outs = [[] for _ in range(len(srcs_ids))]

        def pick_src_encodings(encoder_out, src_ids):
            (unpacked_output, final_hiddens, final_cells, src_lengths,
                src_tokens) = encoder_out
            unpacked_output = unpacked_output.index_select(1, src_ids)
            final_hiddens = final_hiddens.index_select(1, src_ids)
            final_cells = final_cells.index_select(1, src_ids)
            src_lengths = src_lengths.index_select(0, src_ids)
            src_tokens = src_tokens.index_select(0, src_ids)
            max_src_len = src_lengths.data.max()
            return unpacked_output[:max_src_len, :, :
                ], final_hiddens, final_cells, src_lengths, src_tokens[:, :
                max_src_len]
        for model in self.models:
            if not self.retain_dropout:
                model.eval()
            encoder_out = model.encoder(*encoder_inputs)
            for k, src_ids in enumerate(srcs_ids):
                encoder_out_k = pick_src_encodings(encoder_out, src_ids)
                encoder_out_k = model.expand_encoder_output(encoder_out_k,
                    beam_size)
                encoder_outs[k].append(encoder_out_k)
        return encoder_outs

    def _decode(self, tokens, encoder_outs, incremental_states, n_srcs=1):
        srcs_weights = [1 / n_srcs] * n_srcs
        avg_probs = None
        avg_attn = None
        for src_id, src_weight in enumerate(srcs_weights):
            for model_id, (model_weight, model) in enumerate(zip(self.
                model_weights, self.models)):
                with torch.no_grad():
                    encoder_out = encoder_outs[src_id][model_id]
                    incremental_state = incremental_states[src_id, model_id]
                    decoder_out = list(model.decoder(tokens, encoder_out,
                        incremental_state))
                    decoder_out[0] = decoder_out[0][:, (-1), :]
                    attn = decoder_out[1]
                    if len(decoder_out) == 3:
                        possible_translation_tokens = decoder_out[2]
                    else:
                        possible_translation_tokens = None
                probs = src_weight * model_weight * model.get_normalized_probs(
                    decoder_out, log_probs=False)
                if avg_probs is None:
                    avg_probs = probs
                else:
                    avg_probs.add_(probs)
                if attn is not None and src_id == self.align_to:
                    attn = attn[:, (-1), :]
                    if avg_attn is None:
                        avg_attn = attn
                    else:
                        avg_attn.add_(attn)
        avg_probs.log_()
        if avg_attn is not None:
            avg_attn.div_(len(self.models))
        return avg_probs, avg_attn, possible_translation_tokens


class BiLSTM(nn.Module):
    """Wrapper for nn.LSTM

    Differences include:
    * weight initialization
    * the bidirectional option makes the first layer bidirectional only
    (and in that case the hidden dim is divided by 2)
    """

    @staticmethod
    def LSTM(input_size, hidden_size, **kwargs):
        m = nn.LSTM(input_size, hidden_size, **kwargs)
        for name, param in m.named_parameters():
            if 'weight' in name or 'bias' in name:
                param.data.uniform_(-0.1, 0.1)
        return m

    def __init__(self, num_layers, bidirectional, embed_dim, hidden_dim,
        dropout, residual_level):
        super().__init__()
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        if bidirectional:
            assert hidden_dim % 2 == 0, 'hidden_dim should be even if bidirectional'
        self.hidden_dim = hidden_dim
        self.residual_level = residual_level
        self.layers = nn.ModuleList([])
        for layer in range(num_layers):
            is_layer_bidirectional = bidirectional and layer == 0
            if is_layer_bidirectional:
                assert hidden_dim % 2 == 0, 'hidden_dim must be even if bidirectional (to be divided evenly between directions)'
            self.layers.append(BiLSTM.LSTM(embed_dim if layer == 0 else
                hidden_dim, hidden_dim // 2 if is_layer_bidirectional else
                hidden_dim, num_layers=1, dropout=dropout, bidirectional=
                is_layer_bidirectional))

    def forward(self, embeddings, lengths, enforce_sorted=True):
        bsz = embeddings.size()[1]
        packed_input = pack_padded_sequence(embeddings, lengths,
            enforce_sorted=enforce_sorted)
        final_hiddens, final_cells = [], []
        for i, rnn_layer in enumerate(self.layers):
            if self.bidirectional and i == 0:
                h0 = embeddings.new(2, bsz, self.hidden_dim // 2).zero_()
                c0 = embeddings.new(2, bsz, self.hidden_dim // 2).zero_()
            else:
                h0 = embeddings.new(1, bsz, self.hidden_dim).zero_()
                c0 = embeddings.new(1, bsz, self.hidden_dim).zero_()
            current_output, (h_last, c_last) = rnn_layer(packed_input, (h0, c0)
                )
            if self.bidirectional and i == 0:
                h_last = torch.cat((h_last[(0), :, :], h_last[(1), :, :]),
                    dim=1)
                c_last = torch.cat((c_last[(0), :, :], c_last[(1), :, :]),
                    dim=1)
            else:
                h_last = h_last.squeeze(dim=0)
                c_last = c_last.squeeze(dim=0)
            final_hiddens.append(h_last)
            final_cells.append(c_last)
            if self.residual_level is not None and i >= self.residual_level:
                packed_input[0] = packed_input.clone()[0] + current_output[0]
            else:
                packed_input = current_output
        final_hiddens = torch.cat(final_hiddens, dim=0).view(self.
            num_layers, *final_hiddens[0].size())
        final_cells = torch.cat(final_cells, dim=0).view(self.num_layers, *
            final_cells[0].size())
        unpacked_output, _ = pad_packed_sequence(packed_input)
        return unpacked_output, final_hiddens, final_cells


class MILSTMCellBackend(nn.RNNCell):

    def __init__(self, input_size, hidden_size, bias=True):
        super(MILSTMCellBackend, self).__init__(input_size, hidden_size,
            bias=False)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size)
            )
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size,
            hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(4 * hidden_size))
        else:
            self.register_parameter('bias', None)
        self.alpha = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.beta_h = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.beta_i = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        hx, cx = hidden
        Wx = F.linear(x, self.weight_ih)
        Uz = F.linear(hx, self.weight_hh)
        gates = (self.alpha * Wx * Uz + self.beta_i * Wx + self.beta_h * Uz +
            self.bias)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
        cy = forgetgate * cx + ingate * cellgate
        hy = outgate * F.tanh(cy)
        return hy, cy


class LayerNormLSTMCellBackend(nn.LSTMCell):

    def __init__(self, input_dim, hidden_dim, bias=True, epsilon=1e-05):
        super(LayerNormLSTMCellBackend, self).__init__(input_dim,
            hidden_dim, bias)
        self.epsilon = epsilon

    def _layerNormalization(self, x):
        mean = x.mean(1, keepdim=True).expand_as(x)
        std = x.std(1, keepdim=True).expand_as(x)
        return (x - mean) / (std + self.epsilon)

    def forward(self, x, hidden):
        hx, cx = hidden
        gates = F.linear(x, self.weight_ih, self.bias_ih) + F.linear(hx,
            self.weight_hh, self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = F.sigmoid(self._layerNormalization(ingate))
        forgetgate = F.sigmoid(self._layerNormalization(forgetgate))
        cellgate = F.tanh(self._layerNormalization(cellgate))
        outgate = F.sigmoid(self._layerNormalization(outgate))
        cy = forgetgate * cx + ingate * cellgate
        hy = outgate * F.tanh(cy)
        return hy, cy


class FeedForwardNetwork(nn.Module):

    def __init__(self, input_size, hidden_size=None, output_size=None,
        num_layers=2, dropout=0.0):
        super().__init__()
        self.input_size = input_size
        self.output_size = (self.input_size if output_size is None else
            output_size)
        self.hidden_size = (self.output_size if hidden_size is None else
            hidden_size)
        self.num_layers = num_layers
        self.activation_type = 'relu'
        self.dropout = dropout
        self.layers = nn.ModuleList()
        if num_layers == 1:
            self.layers.append(Linear(self.input_size, self.output_size))
        else:
            self.layers.append(Linear(self.input_size, self.hidden_size))
            for _ in range(1, num_layers - 1):
                self.layers.append(Linear(self.hidden_size, self.hidden_size))
            self.layers.append(Linear(self.hidden_size, self.output_size))

    def forward(self, x):
        x = self.layers[0](x)
        for layer in self.layers[1:]:
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = layer(x)
        return x

    def extra_repr(self):
        return 'activation_type={}, dropout={}'.format(self.activation_type,
            self.dropout)


logger = logging.getLogger(__name__)


def select_top_candidate_per_word(source_index, target_indices_with_prob,
    counter_per_word, max_translation_candidates_per_word,
    translation_candidates, translation_candidates_set):
    translation_candidates_saved = 0
    target_indices_with_prob.sort(key=lambda x: x[1], reverse=True)
    for target_index_with_prob in target_indices_with_prob:
        if counter_per_word[source_index
            ] >= max_translation_candidates_per_word:
            break
        translation_candidates[source_index, counter_per_word[source_index]
            ] = target_index_with_prob[0]
        translation_candidates_set.update((source_index,
            target_index_with_prob[0]))
        counter_per_word[source_index] += 1
        translation_candidates_saved += 1
    return translation_candidates_saved


def get_translation_candidates(src_dict, dst_dict, lexical_dictionaries,
    num_top_words, max_translation_candidates_per_word):
    """
    Reads a lexical dictionary file, where each line is (source token, possible
    translation of source token, probability). The file is generally grouped
    by source tokens, but within the group, the probabilities are not
    necessarily sorted.

    A a 0.3
    A c 0.1
    A e 0.05
    A f 0.01
    B b 0.6
    B b 0.2
    A z 0.001
    A y 0.002
    ...

    Returns: translation_candidates
        Matrix of shape (src_dict, max_translation_candidates_per_word) where
        each row corresponds to a source word in the vocab and contains token
        indices of translation candidates for that source word
    """
    translation_candidates = np.zeros([len(src_dict),
        max_translation_candidates_per_word], dtype=np.int32)
    counter_per_word = np.zeros(len(src_dict), dtype=np.int32)
    translation_candidates_set = set()
    for lexical_dictionary in lexical_dictionaries:
        logger.info(f'Processing dictionary file {lexical_dictionary}')
        translation_candidates_saved = 0
        with codecs.open(lexical_dictionary, 'r', 'utf-8'
            ) as lexical_dictionary_file:
            current_source_index = None
            current_target_indices = []
            for line in lexical_dictionary_file.readlines():
                alignment_data = line.split()
                if len(alignment_data) != 3:
                    logger.warning(
                        f'Malformed line in lexical dictionary: {line}')
                    continue
                source_word, target_word, prob = alignment_data
                prob = float(prob)
                source_index = src_dict.index(source_word)
                target_index = dst_dict.index(target_word)
                if (source_index not in src_dict.lexicon_indices and 
                    target_index in dst_dict.lexicon_indices):
                    continue
                if source_index is not None and target_index is not None:
                    if source_index != current_source_index:
                        translation_candidates_saved += (
                            select_top_candidate_per_word(
                            current_source_index, current_target_indices,
                            counter_per_word,
                            max_translation_candidates_per_word,
                            translation_candidates, translation_candidates_set)
                            )
                        current_source_index = source_index
                        current_target_indices = []
                    if target_index >= num_top_words and (source_index,
                        target_index) not in translation_candidates_set:
                        current_target_indices.append((target_index, prob))
        translation_candidates_saved += select_top_candidate_per_word(
            current_source_index, current_target_indices, counter_per_word,
            max_translation_candidates_per_word, translation_candidates,
            translation_candidates_set)
        logger.info(
            f'Loaded {translation_candidates_saved} translationcandidates from dictionary {lexical_dictionary}'
            )
    return translation_candidates


class VocabReduction(nn.Module):

    def __init__(self, src_dict, dst_dict, vocab_reduction_params,
        predictor=None, fp16: bool=False):
        super().__init__()
        self.src_dict = src_dict
        self.dst_dict = dst_dict
        self.vocab_reduction_params = vocab_reduction_params
        self.predictor = predictor
        self.fp16 = fp16
        self.translation_candidates = None
        if (self.vocab_reduction_params is not None and self.
            vocab_reduction_params['max_translation_candidates_per_word'] > 0):
            translation_candidates = get_translation_candidates(self.
                src_dict, self.dst_dict, self.vocab_reduction_params[
                'lexical_dictionaries'], self.vocab_reduction_params[
                'num_top_words'], self.vocab_reduction_params[
                'max_translation_candidates_per_word'])
            self.translation_candidates = nn.Parameter(torch.Tensor(
                translation_candidates).long(), requires_grad=False)

    def forward(self, src_tokens, encoder_output=None, decoder_input_tokens
        =None):
        assert self.dst_dict.pad(
            ) == 0, f'VocabReduction only works correctly when the padding ID is 0 (to ensure its position in possible_translation_tokens is also 0), instead of {self.dst_dict.pad()}.'
        vocab_list = [src_tokens.new_tensor([self.dst_dict.pad()])]
        if decoder_input_tokens is not None:
            flat_decoder_input_tokens = decoder_input_tokens.view(-1)
            vocab_list.append(flat_decoder_input_tokens)
        if self.translation_candidates is not None:
            reduced_vocab = self.translation_candidates.index_select(dim=0,
                index=src_tokens.view(-1)).view(-1)
            vocab_list.append(reduced_vocab)
        if (self.vocab_reduction_params is not None and self.
            vocab_reduction_params['num_top_words'] > 0):
            top_words = torch.arange(self.vocab_reduction_params[
                'num_top_words'], device=vocab_list[0].device).long()
            vocab_list.append(top_words)
        if self.predictor is not None:
            assert encoder_output is not None
            pred_output = self.predictor(encoder_output)
            topk_indices = self.predictor.get_topk_predicted_tokens(pred_output
                , src_tokens, log_probs=True)
            topk_indices = topk_indices.view(-1)
            vocab_list.append(topk_indices.detach())
        all_translation_tokens = torch.cat(vocab_list, dim=0)
        possible_translation_tokens = torch.unique(all_translation_tokens,
            sorted=True, return_inverse=False).type_as(src_tokens)
        len_mod_eight = possible_translation_tokens.shape[0] % 8
        if self.training and self.fp16 and len_mod_eight != 0:
            possible_translation_tokens = torch.cat([
                possible_translation_tokens, possible_translation_tokens.
                new_tensor([self.dst_dict.pad()] * (8 - len_mod_eight))])
        return possible_translation_tokens


class WordPredictor(nn.Module):

    def __init__(self, encoder_output_dim, hidden_dim, output_dim,
        topk_labels_per_source_token=None, use_self_attention=False):
        super().__init__()
        self.encoder_output_dim = encoder_output_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.topk_labels_per_source_token = topk_labels_per_source_token
        self.use_self_attention = use_self_attention
        if self.use_self_attention:
            self.init_layer = nn.Linear(encoder_output_dim, encoder_output_dim)
            self.attn_layer = nn.Linear(2 * encoder_output_dim, 1)
            self.hidden_layer = nn.Linear(2 * encoder_output_dim, hidden_dim)
            self.output_layer = nn.Linear(hidden_dim, output_dim)
        else:
            self.hidden_layer = nn.Linear(encoder_output_dim, hidden_dim)
            self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, encoder_output):
        encoder_hiddens, *_ = encoder_output
        assert encoder_hiddens.dim()
        if self.use_self_attention:
            init_state = self._get_init_state(encoder_hiddens)
            attn_scores = self._attention(encoder_hiddens, init_state)
            attned_state = (encoder_hiddens * attn_scores).sum(0)
            pred_input = torch.cat([init_state, attned_state], 1)
            pred_hidden = F.relu(self.hidden_layer(pred_input))
            logits = self.output_layer(pred_hidden)
        else:
            hidden = F.relu(self.hidden_layer(encoder_hiddens))
            mean_hidden = torch.mean(hidden, 0)
            max_hidden = torch.max(hidden, 0)[0]
            logits = self.output_layer(mean_hidden + max_hidden)
        return logits

    def _get_init_state(self, encoder_hiddens):
        x = torch.mean(encoder_hiddens, 0)
        x = F.relu(self.init_layer(x))
        return x

    def _attention(self, encoder_hiddens, init_state):
        init_state = init_state.unsqueeze(0).expand_as(encoder_hiddens)
        attn_input = torch.cat([init_state, encoder_hiddens], 2)
        attn_scores = F.relu(self.attn_layer(attn_input))
        attn_scores = F.softmax(attn_scores, 0)
        return attn_scores

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output
        if log_probs:
            return F.log_softmax(logits, dim=1)
        else:
            return F.softmax(logits, dim=1)

    def get_topk_predicted_tokens(self, net_output, src_tokens, log_probs: bool
        ):
        """
        Get self.topk_labels_per_source_token top predicted words for vocab
        reduction (per source token).
        """
        assert isinstance(self.topk_labels_per_source_token, int
            ) and self.topk_labels_per_source_token > 0, 'topk_labels_per_source_token must be a positive int, or None'
        k = src_tokens.size(1) * self.topk_labels_per_source_token
        probs = self.get_normalized_probs(net_output, log_probs)
        _, topk_indices = torch.topk(probs, k, dim=1)
        return topk_indices


import torch
from _paritybench_helpers import _mock_config, _mock_layer, _paritybench_base, _fails_compile

class Test_pytorch_translate(_paritybench_base):
    pass
    def test_000(self):
        self._check(ContextEmbedding(*[], **{'embed_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_001(self):
        self._check(FeedForwardNetwork(*[], **{'input_size': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_002(self):
        self._check(HighwayLayer(*[], **{'input_dim': 4}), [torch.rand([4, 4, 4, 4])], {})

    @_fails_compile()
    def test_003(self):
        self._check(WordPredictor(*[], **{'encoder_output_dim': 4, 'hidden_dim': 4, 'output_dim': 4}), [torch.rand([4, 4, 4, 4])], {})
