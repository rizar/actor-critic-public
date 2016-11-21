import numpy
from numpy.testing import assert_equal, assert_allclose
import theano
from theano import tensor

from blocks.initialization import Uniform
from lvsr.bricks.readouts import ReinforceReadout
from lvsr.bricks import EditDistanceReward


class TestReadouts(object):

    def setUp(self):
        self.readout = ReinforceReadout(
            reward_brick=EditDistanceReward(0, 2),
            input_names=['states', 'attended', 'attended_mask'],
            num_tokens=4, input_dims=[2, 3, 2],
            weights_init=Uniform(width=1.0),
            biases_init=Uniform(width=1.0),
            seed=1)
        self.readout.initialize()

        self.states = numpy.array(
            [[[1., 2.]], [[2., 1.]]],
            dtype=theano.config.floatX)

    def test_scores(self):
        assert self.readout.scores.inputs == ['states',
                                              'attended',
                                              'attended_mask']
        # TODO: implement me

    def test_sample(self):
        assert self.readout.scores.inputs == ['states',
                                              'attended',
                                              'attended_mask']
        # TODO: check the returned value
        attended = tensor.tensor3('attended')
        attended_mask = tensor.tensor3('attended_mask')
        self.readout.sample(states=self.states[0],
                            attended=attended,
                            attended_mask=attended_mask)
