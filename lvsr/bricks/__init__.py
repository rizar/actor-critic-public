import logging
import numpy
from theano import tensor

from blocks.bricks import (
    Brick, Initializable, Linear, Sequence, Tanh)
from blocks.bricks.base import lazy, application
from blocks.bricks.parallel import Fork, Distribute
from blocks.bricks.recurrent import Bidirectional
from blocks.utils import dict_union, dict_subset, shared_floatx_nans
from blocks.roles import WEIGHT, add_role

from lvsr.ops import EditDistanceOp, BleuOp

logger = logging.getLogger(__name__)


class RecurrentWithFork(Initializable):

    @lazy(allocation=['input_dim'])
    def __init__(self, recurrent, input_dim, **kwargs):
        super(RecurrentWithFork, self).__init__(**kwargs)
        self.recurrent = recurrent
        self.input_dim = input_dim
        self.fork = Fork(
            [name for name in self.recurrent.sequences
             if name != 'mask'],
             prototype=Linear())
        self.children = [recurrent.brick, self.fork]

    def _push_allocation_config(self):
        self.fork.input_dim = self.input_dim
        self.fork.output_dims = [self.recurrent.brick.get_dim(name)
                                 for name in self.fork.output_names]

    @application(inputs=['input_', 'mask'])
    def apply(self, input_, mask=None, **kwargs):
        return self.recurrent(
            mask=mask, **dict_union(self.fork.apply(input_, as_dict=True),
                                    kwargs))

    @apply.property('outputs')
    def apply_outputs(self):
        return self.recurrent.states


class RecurrentWithExtraInput(Initializable):

    @lazy(allocation=['extra_input_dim'])
    def __init__(self, recurrent, extra_input_name, extra_input_dim, **kwargs):
        self.recurrent = recurrent
        self.extra_input_name = extra_input_name
        self.extra_input_dim = extra_input_dim
        self._normal_inputs = [
            name for name in self.recurrent.apply.sequences if name != 'mask']
        self.distribute = Distribute(self._normal_inputs, self.extra_input_name)
        children = [self.recurrent, self.distribute]
        super(RecurrentWithExtraInput, self).__init__(children=children, **kwargs)

        self.apply.sequences = self.recurrent.apply.sequences + [self.extra_input_name]
        self.apply.outputs = self.recurrent.apply.outputs
        self.apply.states = self.recurrent.apply.states
        self.apply.contexts = self.recurrent.apply.contexts
        self.initial_states.outputs = self.recurrent.initial_states.outputs

    def _push_allocation_config(self):
        self.distribute.source_dim = self.extra_input_dim
        self.distribute.target_dims = self.recurrent.get_dims(self.distribute.target_names)

    @application
    def apply(self, **kwargs):
        # Should handle both "iterate=True" and "iterate=False"
        extra_input = kwargs.pop(self.extra_input_name)
        mask = kwargs.pop('mask', None)
        normal_inputs = dict_subset(kwargs, self._normal_inputs, pop=True)
        normal_inputs = self.distribute.apply(
            as_dict=True, **dict_union(normal_inputs, {self.extra_input_name: extra_input}))
        return self.recurrent.apply(mask=mask, **dict_union(normal_inputs, kwargs))

    @application
    def initial_states(self, *args, **kwargs):
        return self.recurrent.initial_states(*args, **kwargs)

    def get_dim(self, name):
        if name == self.extra_input_name:
            return self.extra_input_dim
        return self.recurrent.get_dim(name)


class InitializableSequence(Sequence, Initializable):
    pass


class Encoder(Initializable):

    def __init__(self, enc_transition, dims, dim_input, subsample, bidir, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.subsample = subsample

        if dims:
            dims_under = [dim_input] + list((2 if bidir else 1) * numpy.array(dims))
            for layer_num, (dim_under, dim) in enumerate(zip(dims_under, dims)):
                layer = RecurrentWithFork(
                        enc_transition(dim=dim, activation=Tanh()).apply,
                        dim_under,
                        name='with_fork{}'.format(layer_num))
                if bidir:
                    layer = Bidirectional(layer, name='bidir{}'.format(layer_num))
                self.children.append(layer)
            self.dim_encoded = (2 if bidir else 1) * dims[-1]
        else:
            self.dim_encoded = dim_input

    @application(outputs=['encoded', 'encoded_mask'])
    def apply(self, input_, mask=None):
        for layer, take_each in zip(self.children, self.subsample):
            input_ = layer.apply(input_, mask)
            input_ = input_[::take_each]
            if mask:
                mask = mask[::take_each]
        return input_, (mask if mask else tensor.ones_like(input_[:, :, 0]))

    def get_dim(self, name):
        if name == self.apply.outputs[0]:
            return self.dim_encoded
        return super(Encoder, self).get_dim(name)


class ConvEncoder(Initializable):

    def __init__(self, max_length, dim, width, **kwargs):
        self.max_length = max_length
        self.dim = dim
        self.width = width
        if self.width % 2 != 1:
            raise ValueError
        super(ConvEncoder, self).__init__(**kwargs)

    def _allocate(self):
        self.padding = shared_floatx_nans(
            (self.dim,), name='padding')
        add_role(self.padding, WEIGHT)
        self.location_component = shared_floatx_nans(
            (self.max_length, self.dim), name='loc_comp')
        add_role(self.location_component, WEIGHT)
        self.parameters = [self.padding, self.location_component]

    def _initialize(self):
        self.weights_init.initialize(self.padding, self.rng)
        self.weights_init.initialize(self.location_component, self.rng)

    @application(outputs=['encoder', 'encoded_mask'])
    def apply(self, input_, mask=None):
        w = self.width
        hw = self.width / 2

        # Step 1: pad input with padding
        if mask:
            input_ = (mask[:, :, None] * input_ +
                      (1 - mask)[:, :, None] * self.padding[None, None, :])
        padded = tensor.alloc(
            self.padding, input_.shape[0] + w, input_.shape[1], self.dim)
        padded = tensor.set_subtensor(padded[hw + 1:-hw], input_)

        # Step 2: add location-based component
        padded += self.location_component[:padded.shape[0], None, :]

        # Step 3: compute sums across the windows
        cumsum = padded.cumsum(axis=0)
        output = (cumsum[w:] - cumsum[:-w]) / w

        return output, (mask if mask else tensor.ones_like(output[:, :, 0]))

    def get_dim(self, name):
        if name == self.apply.outputs[0]:
            return self.dim
        return super(ConvEncoder, self).get_dim(name)


class EditDistanceReward(Brick):

    def __init__(self, bos_label, eos_label, deltas=False,  **kwargs):
        super(EditDistanceReward, self).__init__(**kwargs)
        self.op = EditDistanceOp(bos_label, eos_label, deltas)

    @application
    def apply(self, prediction, prediction_mask,
              groundtruth, groundtruth_mask):
        return -self.op(prediction, prediction_mask,
                        groundtruth, groundtruth_mask)


class BleuReward(Brick):

    def __init__(self, bos_label, eos_label, deltas=False,  **kwargs):
        super(BleuReward, self).__init__(**kwargs)
        self.op = BleuOp(bos_label, eos_label, deltas)

    @application
    def apply(self, prediction, prediction_mask,
              groundtruth, groundtruth_mask):
        blues = self.op(prediction, prediction_mask,
                        groundtruth, groundtruth_mask)
        return blues * groundtruth_mask.sum(axis=0)[None, :, None]
