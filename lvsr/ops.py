from __future__ import print_function
import math
import numpy
import theano
import itertools
from theano import tensor, Op
from theano.gradient import disconnected_type
from fuel.utils import do_not_pickle_attributes
from picklable_itertools.extras import equizip
from collections import defaultdict, deque

from toposort import toposort_flatten

from lvsr.error_rate import (
    reward_matrix, gain_matrix, edit_distance, _edit_distance_matrix, _bleu)


class RewardOp(Op):
    __props__ = ()

    def __init__(self, eos_label, alphabet_size):
        """Computes matrices of rewards and gains."""
        self.eos_label = eos_label
        self.alphabet_size = alphabet_size

    def perform(self, node, inputs, output_storage):
        groundtruth, recognized = inputs
        if (groundtruth.ndim != 2 or recognized.ndim != 2
                or groundtruth.shape[1] != recognized.shape[1]):
            raise ValueError
        batch_size = groundtruth.shape[1]
        all_rewards = numpy.zeros(
            recognized.shape + (self.alphabet_size,), dtype='int64')
        all_gains = numpy.zeros(
            recognized.shape + (self.alphabet_size,), dtype='int64')
        alphabet = list(range(self.alphabet_size))
        for index in range(batch_size):
            y = list(groundtruth[:, index])
            y_hat = list(recognized[:, index])
            try:
                eos_pos = y.index(self.eos_label)
                y = y[:eos_pos + 1]
            except:
                # Sometimes groundtruth is in fact also a prediction
                # and in this case it might not have EOS label
                pass
            if self.eos_label in y_hat:
                y_hat_eos_pos = y_hat.index(self.eos_label)
                y_hat_trunc = y_hat[:y_hat_eos_pos + 1]
            else:
                y_hat_trunc = y_hat
            rewards_trunc = reward_matrix(
                y, y_hat_trunc, alphabet, self.eos_label)
            # pass freshly computed rewards to gain_matrix to speed things up
            # a bit
            gains_trunc = gain_matrix(y, y_hat_trunc, alphabet,
                                      given_reward_matrix=rewards_trunc)
            gains = numpy.ones((len(y_hat), len(alphabet))) * -1000
            gains[:(gains_trunc.shape[0] - 1), :] = gains_trunc[:-1, :]

            rewards = numpy.ones((len(y_hat), len(alphabet))) * -1
            rewards[:(rewards_trunc.shape[0] - 1), :] = rewards_trunc[:-1, :]
            all_rewards[:, index, :] = rewards
            all_gains[:, index, :] = gains

        output_storage[0][0] = all_rewards
        output_storage[1][0] = all_gains

    def grad(self, *args, **kwargs):
        return disconnected_type(), disconnected_type()

    def make_node(self, groundtruth, recognized):
        recognized = tensor.as_tensor_variable(recognized)
        groundtruth = tensor.as_tensor_variable(groundtruth)
        return theano.Apply(
            self, [groundtruth, recognized], [tensor.ltensor3(), tensor.ltensor3()])


def trim(y, mask):
    try:
        return y[:mask.index(0.)]
    except ValueError:
        return y


class EditDistanceOp(Op):
    __props__ = ()

    def __init__(self, bos_label, eos_label, deltas=False):
        self.bos_label = bos_label
        self.eos_label = eos_label
        self.deltas = deltas

    def perform(self, node, inputs, output_storage):
        prediction, prediction_mask, groundtruth, groundtruth_mask = inputs
        if (groundtruth.ndim != 2 or prediction.ndim != 2
                or groundtruth.shape[1] != prediction.shape[1]):
            raise ValueError
        batch_size = groundtruth.shape[1]

        results = numpy.zeros_like(prediction[:, :, None])
        for index in range(batch_size):
            y = trim(list(groundtruth[:, index]),
                     list(groundtruth_mask[:, index]))
            y_hat = trim(list(prediction[:, index]),
                         list(prediction_mask[:, index]))
            if self.deltas:
                matrix = _edit_distance_matrix(
                    y, y_hat, special_tokens={self.bos_label, self.eos_label})

                row = matrix[-1, :].copy()
                results[:len(y_hat), index, 0] = row[1:] - matrix[-1, :-1]
            else:
                results[len(y_hat) - 1, index, 0] = edit_distance(y, y_hat)

        output_storage[0][0] = results

    def grad(self, *args, **kwargs):
        return theano.gradient.disconnected_type()

    def make_node(self, prediction, prediction_mask,
                  groundtruth, groundtruth_mask):
        prediction = tensor.as_tensor_variable(prediction)
        prediction_mask = tensor.as_tensor_variable(prediction_mask)
        groundtruth = tensor.as_tensor_variable(groundtruth)
        groundtruth_mask = tensor.as_tensor_variable(groundtruth_mask)
        return theano.Apply(
            self, [prediction, prediction_mask,
                   groundtruth, groundtruth_mask], [tensor.ltensor3()])


class BleuOp(Op):
    __props__ = ()

    def __init__(self, bos_label, eos_label, deltas=False):
        self.n = 4
        self.deltas = deltas
        self.special_tokens = set([bos_label, eos_label])

    def grad(self, *args, **kwargs):
            return [theano.gradient.disconnected_type()] * 4

    def perform(self, node, inputs, output_storage):
        prediction, prediction_mask, groundtruth, groundtruth_mask = inputs
        if (groundtruth.ndim != 2 or prediction.ndim != 2
                or groundtruth.shape[1] != prediction.shape[1]):
            raise ValueError
        batch_size = groundtruth.shape[1]

        results = numpy.zeros_like(prediction[:, :, None]).astype('float32')
        for index in range(batch_size):
            y = trim(list(groundtruth[:, index]),
                     list(groundtruth_mask[:, index]))
            y_no_special = [token for token in y
                            if token not in self.special_tokens]
            y_hat = trim(list(prediction[:, index]),
                         list(prediction_mask[:, index]))
            y_hat_no_special = [token for token in y_hat
                                if token not in self.special_tokens]
            blues, _, _, _ = _bleu(y_no_special, y_hat_no_special, self.n)
            reward = blues[:, self.n - 1].copy()
            if self.deltas:
                reward[1:] = reward[1:] - reward[:-1]
                pos = -1
                for i in range(len(y_hat)):
                    if y_hat[i] not in self.special_tokens:
                        pos = pos + 1
                        results[i, index, 0] = reward[pos]
                    else:
                        results[i, index, 0] = 0.
            elif len(reward):
                results[len(y_hat) - 1, index, 0] = reward[-1]
        output_storage[0][0] = results

    def make_node(self, prediction, prediction_mask,
                  groundtruth, groundtruth_mask):
        prediction = tensor.as_tensor_variable(prediction)
        prediction_mask = tensor.as_tensor_variable(prediction_mask)
        groundtruth = tensor.as_tensor_variable(groundtruth)
        groundtruth_mask = tensor.as_tensor_variable(groundtruth_mask)
        return theano.Apply(
            self,
            [prediction, prediction_mask,
             groundtruth, groundtruth_mask],
            [tensor.tensor3()])
