import copy
import logging

import numpy
import theano
from theano import tensor
from theano.gradient import disconnected_grad

from blocks.bricks import (
    Bias, Identity, Initializable, MLP, Tanh, Softmax, Random)
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.base import application
from blocks.bricks.recurrent import (
    BaseRecurrent, RecurrentStack, recurrent)
from blocks_extras.bricks.sequence_generator2 import (
    SequenceGenerator, SoftmaxReadout, Feedback)
from blocks_extras.bricks.attention2 import AttentionRecurrent
from blocks.bricks.lookup import LookupTable
from blocks.graph import ComputationGraph
from blocks.model import Model
from blocks.filter import VariableFilter
from blocks.serialization import load_parameters
from blocks.utils import dict_union, dict_subset

from lvsr.bricks import (
    Encoder, InitializableSequence, EditDistanceReward, BleuReward,
    RecurrentWithExtraInput, ConvEncoder)
from lvsr.bricks.readouts import (
    ReinforceReadout, CriticReadout, ActorCriticReadout)
from lvsr.bricks.attention import SequenceContentAndConvAttention
from lvsr.utils import global_push_initialization_config
from lvsr.beam_search import BeamSearch

logger = logging.getLogger(__name__)


class Bottom(Initializable):
    """
    A bottom class that mergers possibly many input sources into one
    sequence.

    The bottom is responsible for allocating variables for single and
    multiple sequences in a batch.

    In speech recognition this will typically be the identity transformation
    ro a small MLP.

    Attributes
    ----------
    vector_input_sources : list of str
    discrete_input_sources : list of str

    Parameters
    ----------
    input_dims : dict
        Maps input source to their dimensions, only for vector sources.
    input_num_chars : dict
        Maps input source to their range of values, only for discrete sources.

    """
    vector_input_sources = []
    discrete_input_sources = []

    def __init__(self, input_dims, input_num_chars, **kwargs):
        super(Bottom, self).__init__(**kwargs)
        self.input_dims = input_dims
        self.input_num_chars = input_num_chars


class LookupBottom(Bottom):
    discrete_input_sources = ['inputs']

    def __init__(self, dim, **kwargs):
        super(LookupBottom, self).__init__(**kwargs)
        self.dim = dim

        self.mask = tensor.matrix('inputs_mask')
        self.batch_inputs = {
            'inputs': tensor.lmatrix('inputs')}
        self.single_inputs = {
            'inputs': tensor.lvector('inputs')}

        self.children = [LookupTable(self.input_num_chars['inputs'], self.dim)]

    @application(inputs=['inputs'], outputs=['outputs'])
    def apply(self, inputs):
        return self.children[0].apply(inputs)

    def batch_size(self, inputs):
        return inputs.shape[1]

    def num_time_steps(self, inputs):
        return inputs.shape[0]

    def single_to_batch_inputs(self, inputs):
        # Note: this code supports many inputs, which are all sequences
        inputs = {n: v[:, None, :] if v.ndim == 2 else v[:, None]
                  for (n, v) in inputs.items()}
        inputs_mask = tensor.ones((self.num_time_steps(**inputs),
                                   self.batch_size(**inputs)))
        return inputs, inputs_mask

    def get_dim(self, name):
        if name == 'outputs':
            return self.dim
        return super(LookupBottom, self).get_dim(name)


class SpeechBottom(Bottom):
    """
    A Bottom specialized for speech recognition that accets only one input
    - the recordings.
    """
    vector_input_sources = ['recordings']

    def __init__(self, activation, dims=None, **kwargs):
        super(SpeechBottom, self).__init__(**kwargs)
        self.num_features = self.input_dims['recordings']

        if activation is None:
            activation = Tanh()

        if dims:
            child = MLP([activation] * len(dims),
                        [self.num_features] + dims,
                        name="bottom")
            self.output_dim = child.output_dim
        else:
            child = Identity(name='bottom')
            self.output_dim = self.num_features
        self.children.append(child)

        self.mask = tensor.matrix('recordings_mask')
        self.batch_inputs = {
            'recordings': tensor.tensor3('recordings')}
        self.single_inputs = {
            'recordings': tensor.matrix('recordings')}

    @application(inputs=['recordings'], outputs=['outputs'])
    def apply(self, recordings):
        return self.children[0].apply(recordings)

    def batch_size(self, recordings):
        return recordings.shape[1]

    def num_time_steps(self, recordings):
        return recordings.shape[0]

    def single_to_batch_inputs(self, inputs):
        # Note: this code supports many inputs, which are all sequences
        inputs = {n: v[:, None, :] if v.ndim == 2 else v[:, None]
                  for (n, v) in inputs.items()}
        inputs_mask = tensor.ones((self.num_time_steps(**inputs),
                                   self.batch_size(**inputs)))
        return inputs, inputs_mask

    def get_dim(self, name):
        if name == 'outputs':
            return self.output_dim
        return super(SpeechBottom, self).get_dim(name)


def _downsize_dim(value, times):
    if isinstance(value, int):
        return value / times
    elif isinstance(value, list):
        value = list(value)
        for i in range(len(value)):
            value[i] /= times
        return value
    raise ValueError


def _downsize_config(config, times):
    for option in ['dim_dec', 'dim_matcher', 'dim_output_embedding',
                   'dims_bidir', 'post_merge_dims']:
        value = config.get(option)
        if value is not None:
            config[option] = _downsize_dim(value, times)
    for option in ['dim', 'dims']:
        value = config['bottom'].get(option)
        if value is not None:
            config['bottom'][option] = _downsize_dim(value, times)
    return config


class EncoderDecoder(Initializable, Random):
    """Encapsulate all reusable logic.

    This class plays a few roles: (a) it's a top brick that knows
    how to combine bottom, bidirectional and recognizer network, (b)
    it has the inputs variables and can build whole computation graphs
    starting with them (c) it hides compilation of Theano functions
    and initialization of beam search. I find it simpler to have it all
    in one place for research code.

    Parameters
    ----------
    All defining the structure and the dimensions of the model. Typically
    receives everything from the "net" section of the config.

    """

    def __init__(self,
                 input_dims,
                 input_num_chars,
                 bos_label, eos_label,
                 num_labels,
                 dim_dec, dims_bidir,
                 enc_transition, dec_transition,
                 use_states_for_readout,
                 attention_type,
                 criterion,
                 bottom,
                 lm=None, token_map=None,
                 bidir=True, window_size=None,
                 max_length=None, subsample=None,
                 dims_top=None, extra_input_dim=None,
                 prior=None, conv_n=None,
                 post_merge_activation=None,
                 post_merge_dims=None,
                 dim_matcher=None,
                 embed_outputs=True,
                 dim_output_embedding=None,
                 reuse_bottom_lookup_table=False,
                 dec_stack=1,
                 conv_num_filters=1,
                 data_prepend_eos=True,
                 # softmax is the default set in SequenceContentAndConvAttention
                 energy_normalizer=None,
                 # for speech this is the approximate phoneme duration in frames
                 max_decoded_length_scale=1,
                 # for criterions involving generation of outputs, whether
                 # or not they should be generated by the recognizer itself
                 generate_predictions=True,
                 compute_targets=True,
                 compute_policy=True,
                 extra_generation_steps=3,
                 **kwargs):
        all_arguments = copy.deepcopy(locals())
        all_arguments.update(copy.deepcopy(kwargs))
        del all_arguments['kwargs']
        del all_arguments['self']

        if post_merge_activation is None:
            post_merge_activation = Tanh()
        super(EncoderDecoder, self).__init__(**kwargs)
        self.bos_label = bos_label
        self.eos_label = eos_label
        self.data_prepend_eos = data_prepend_eos

        self.rec_weights_init = None
        self.initial_states_init = None

        self.enc_transition = enc_transition
        self.dec_transition = dec_transition
        self.dec_stack = dec_stack

        self.criterion = criterion
        self.generate_predictions = generate_predictions
        self.extra_generation_steps = extra_generation_steps
        self.compute_targets = compute_targets
        self.compute_policy = compute_policy

        self.max_decoded_length_scale = max_decoded_length_scale

        post_merge_activation = post_merge_activation

        if dim_matcher is None:
            dim_matcher = dim_dec

        # The bottom part, before BiRNN
        bottom_class = bottom.pop('bottom_class')
        bottom = bottom_class(
            input_dims=input_dims, input_num_chars=input_num_chars,
            name='bottom',
            **bottom)

        # BiRNN
        if dims_bidir:
            if not subsample:
                subsample = [1] * len(dims_bidir)
            encoder = Encoder(self.enc_transition, dims_bidir,
                            bottom.get_dim(bottom.apply.outputs[0]),
                            subsample, bidir=bidir)
        elif window_size:
            encoder = ConvEncoder(
                max_length, bottom.get_dim(bottom.apply.outputs[0]), window_size)
        else:
            raise ValueError("Don't know which Encoder to use")
        dim_encoded = encoder.get_dim(encoder.apply.outputs[0])

        # The top part, on top of BiRNN but before the attention
        if dims_top:
            top = MLP([Tanh()],
                      [dim_encoded] + dims_top + [dim_encoded], name="top")
        else:
            top = Identity(name='top')

        if dec_stack == 1:
            transition = self.dec_transition(
                dim=dim_dec, activation=Tanh(), name="transition")
        else:
            assert not extra_input_dim
            transitions = [self.dec_transition(dim=dim_dec,
                                               activation=Tanh(),
                                               name="transition_{}".format(trans_level))
                           for trans_level in xrange(dec_stack)]
            transition = RecurrentStack(transitions=transitions,
                                        skip_connections=True)
        # Choose attention mechanism according to the configuration
        if attention_type == "content":
            attention = SequenceContentAttention(
                state_names=transition.apply.states,
                attended_dim=dim_encoded, match_dim=dim_matcher,
                name="cont_att")
        elif attention_type == "content_and_conv":
            attention = SequenceContentAndConvAttention(
                state_names=transition.apply.states,
                conv_n=conv_n,
                conv_num_filters=conv_num_filters,
                attended_dim=dim_encoded, match_dim=dim_matcher,
                prior=prior,
                energy_normalizer=energy_normalizer,
                name="conv_att")
        else:
            raise ValueError("Unknown attention type {}"
                             .format(attention_type))
        if not embed_outputs:
            raise ValueError("embed_outputs=False is not supported any more")
        if not reuse_bottom_lookup_table:
            embedding = LookupTable(num_labels + 1,
                            dim_dec if
                            dim_output_embedding is None
                            else dim_output_embedding)
        else:
            embedding = bottom.children[0]
        feedback = Feedback(
            embedding=embedding,
            output_names=[s for s in transition.apply.sequences
                           if s != 'mask'])

        # Create a readout
        readout_config = dict(
            num_tokens=num_labels,
            input_names=(transition.apply.states if use_states_for_readout else [])
                         + [attention.take_glimpses.outputs[0]],
            name="readout")
        if post_merge_dims:
            readout_config['merge_dim'] = post_merge_dims[0]
            readout_config['post_merge'] = InitializableSequence([
                Bias(post_merge_dims[0]).apply,
                post_merge_activation.apply,
                MLP([post_merge_activation] * (len(post_merge_dims) - 1) + [Identity()],
                    # MLP was designed to support Maxout is activation
                    # (because Maxout in a way is not one). However
                    # a single layer Maxout network works with the trick below.
                    # For deeper Maxout network one has to use the
                    # Sequence brick.
                    [d//getattr(post_merge_activation, 'num_pieces', 1)
                     for d in post_merge_dims] + [num_labels]).apply,
            ], name='post_merge')
        if 'reward' in criterion and criterion['name'] != 'log_likelihood':
            if criterion['reward'] == 'edit_distance':
                readout_config['reward_brick'] = EditDistanceReward(
                    self.bos_label, self.eos_label)
            elif criterion['reward'] == 'delta_edit_distance':
                readout_config['reward_brick'] = EditDistanceReward(
                    self.bos_label, self.eos_label, deltas=True)
            elif criterion['reward'] == 'bleu':
                readout_config['reward_brick'] = BleuReward(
                    self.bos_label, self.eos_label, deltas=False)
            elif criterion['reward'] == 'delta_bleu':
                readout_config['reward_brick'] = BleuReward(
                    self.bos_label, self.eos_label, deltas=True)
            else:
                raise ValueError("Unknown reward type")
        if criterion['name'] == 'log_likelihood':
            readout_class = SoftmaxReadout
        elif criterion['name'] == 'critic':
            readout_class = CriticReadout
            criterion_copy = dict(criterion)
            del criterion_copy['name']
            readout_config.update(**criterion_copy)
        elif criterion['name'] == 'reinforce':
            readout_class = ReinforceReadout
            readout_config['merge_names'] = list(readout_config['input_names'])
            readout_config['entropy'] = criterion.get('entropy')
            readout_config['input_names'] += ['attended', 'attended_mask']
        elif criterion['name'] in ['sarsa', 'actor_critic']:
            readout_class = ActorCriticReadout
            if criterion['name'] == 'actor_critic':
                critic_arguments = dict(all_arguments)
                # No worries, critic will not compute log likelihood values.
                # We
                critic_arguments['criterion'] = {
                    'name': 'critic',
                    'value_softmax': criterion.get('value_softmax'),
                    'same_value_for_wrong': criterion.get('same_value_for_wrong')}
                critic_arguments['name'] = 'critic'
                if criterion.get('critic_uses_actor_states'):
                    critic_arguments['extra_input_dim'] = dim_dec
                if criterion.get('value_softmax') or criterion.get('same_value_for_wrong'):
                    # Add an extra output for the critic
                    critic_arguments['num_labels'] = num_labels + 1
                if criterion.get('force_bidir'):
                    critic_arguments['dims_bidir'] = [dim_dec]
                critic_arguments['reuse_bottom_lookup_table'] = True
                critic_arguments['input_num_chars'] = {'inputs': num_labels}
                if criterion.get('downsize_critic'):
                    critic_arguments = _downsize_config(
                        critic_arguments, criterion['downsize_critic'])
                critic = EncoderDecoder(**critic_arguments)
                readout_config['critic'] = critic
            readout_config['merge_names'] = list(readout_config['input_names'])
            readout_config['freeze_actor'] = criterion.get('freeze_actor')
            readout_config['freeze_critic'] = criterion.get('freeze_critic')
            readout_config['critic_uses_actor_states'] = criterion.get('critic_uses_actor_states')
            readout_config['critic_uses_groundtruth'] = criterion.get('critic_uses_groundtruth')
            readout_config['critic_burnin_steps'] = criterion.get('critic_burnin_steps')
            readout_config['discount'] = criterion.get('discount')
            readout_config['entropy_reward_coof'] = criterion.get('entropy_reward_coof')
            readout_config['cross_entropy_reward_coof'] = criterion.get('cross_entropy_reward_coof')
            readout_config['value_penalty'] = criterion.get('value_penalty')
            readout_config['critic_policy_t'] = criterion.get('critic_policy_t')
            readout_config['bos_token'] = bos_label
            readout_config['accumulate_outputs'] = criterion.get('accumulate_outputs')
            readout_config['use_value_biases'] = criterion.get('use_value_biases')
            readout_config['actor_grad_estimate'] = criterion.get('actor_grad_estimate')
            readout_config['input_names'] += ['attended', 'attended_mask']
            # Note, that settings below are for the "clean" mode.
            # When get_cost_graph() is run with training=True, they
            # are temporarily overriden with the "real" settings from
            # "criterion"
            readout_config['compute_targets'] = True
            readout_config['compute_policy'] = True
            readout_config['solve_bellman'] = True
        else:
            raise ValueError("Unknown criterion {}".format(criterion['name']))
        readout = readout_class(**readout_config)

        if lm:
            raise ValueError("LM is currently not supported")

        recurrent = AttentionRecurrent(transition, attention)
        if extra_input_dim:
            recurrent = RecurrentWithExtraInput(
                recurrent, "extra_inputs", extra_input_dim, name="with_extra_inputs")
        generator = SequenceGenerator(
            recurrent=recurrent, readout=readout, feedback=feedback,
            name="generator")

        # Remember child bricks
        self.encoder = encoder
        self.bottom = bottom
        self.top = top
        self.generator = generator
        self.softmax = Softmax()
        self.children = [encoder, top, bottom, generator, self.softmax]

        # Create input variables
        self.inputs = self.bottom.batch_inputs
        self.inputs_mask = self.bottom.mask

        self.labels = tensor.lmatrix('labels')
        self.labels_mask = tensor.matrix("labels_mask")

        self.predicted_labels = tensor.lmatrix('predicted_labels')
        self.predicted_mask = tensor.matrix('predicted_mask')

        self.single_inputs = self.bottom.single_inputs
        self.single_labels = tensor.lvector('labels')
        self.single_predicted_labels = tensor.lvector('predicted_labels')
        self.n_steps = tensor.lscalar('n_steps')

        # Configure mixed_generate
        if criterion['name'] == 'actor_critic':
            critic = self.generator.readout.critic
            self.mixed_generate.sequences = []
            self.mixed_generate.states = (
                ['step'] +
                self.generator.recurrent.apply.states +
                ['critic_' + name for name in critic.generator.recurrent.apply.states])
            self.mixed_generate.outputs = (
                ['samples', 'step'] +
                self.generator.recurrent.apply.outputs +
                ['critic_' + name for name in critic.generator.recurrent.apply.outputs])
            self.mixed_generate.contexts = (
                self.generator.recurrent.apply.contexts +
                ['critic_' + name for name in critic.generator.recurrent.apply.contexts]
                + ['groundtruth', 'groundtruth_mask'])
            self.initial_states.outputs = self.mixed_generate.states


    def push_initialization_config(self):
        super(EncoderDecoder, self).push_initialization_config()
        if self.rec_weights_init:
            rec_weights_config = {'weights_init': self.rec_weights_init,
                                  'recurrent_weights_init': self.rec_weights_init}
            global_push_initialization_config(self,
                                              rec_weights_config,
                                              BaseRecurrent)
        if self.initial_states_init:
            global_push_initialization_config(self,
                                              {'initial_states_init': self.initial_states_init})

    @application
    def costs(self, **kwargs):
        # pop inputs we know about
        prediction = kwargs.pop('prediction')
        prediction_mask = kwargs.pop('prediction_mask')
        groundtruth = kwargs.pop('groundtruth', None)
        groundtruth_mask = kwargs.pop('groundtruth_mask', None)
        inputs_mask = kwargs.pop('inputs_mask')
        extra_inputs = kwargs.pop('extra_inputs', None)

        # the rest is for bottom
        bottom_processed = self.bottom.apply(**kwargs)
        encoded, encoded_mask = self.encoder.apply(
            input_=bottom_processed, mask=inputs_mask)
        encoded = self.top.apply(encoded)
        costs_kwargs = dict(
            prediction=prediction, prediction_mask=prediction_mask,
            groundtruth=groundtruth, groundtruth_mask=groundtruth_mask,
            attended=encoded, attended_mask=encoded_mask)
        if extra_inputs:
            costs_kwargs['extra_inputs'] = extra_inputs
        return self.generator.costs(**costs_kwargs)

    @application
    def generate(self, return_initial_states=False, **kwargs):
        inputs_mask = kwargs.pop('inputs_mask')
        n_steps = kwargs.pop('n_steps')

        encoded, encoded_mask = self.encoder.apply(
            input_=self.bottom.apply(**kwargs),
            mask=inputs_mask)
        encoded = self.top.apply(encoded)
        return self.generator.generate(
            n_steps=n_steps if n_steps is not None else self.n_steps,
            batch_size=encoded.shape[1],
            attended=encoded,
            attended_mask=encoded_mask,
            return_initial_states=return_initial_states,
            as_dict=True)

    @recurrent
    def mixed_generate(self, return_initial_states=True, **kwargs):
        critic = self.generator.readout.critic
        groundtruth = kwargs.pop('groundtruth')
        groundtruth_mask = kwargs.pop('groundtruth_mask')
        step = kwargs.pop('step')

        sampling_inputs = dict_subset(
            kwargs, self.generator.readout.sample.inputs)
        actor_scores = self.generator.readout.scores(**sampling_inputs)

        critic_inputs = {
            name: kwargs['critic_' + name]
            for name in critic.generator.readout.merge_names}
        critic_outputs = critic.generator.readout.outputs(
            groundtruth, groundtruth_mask, **critic_inputs)

        epsilon = numpy.array(self.generator.readout.epsilon,
                              dtype=theano.config.floatX)
        actor_probs = tensor.exp(actor_scores)
        # This is a poor man's 1-hot argmax
        critic_probs = self.softmax.apply(critic_outputs * 1000)
        probs = (actor_probs * (tensor.constant(1) - epsilon)
                 + critic_probs * epsilon)

        x = self.theano_rng.uniform(size=(probs.shape[0],))
        samples = (tensor.gt(x[:, None], tensor.cumsum(probs, axis=1))
                   .astype(theano.config.floatX)
                   .sum(axis=1)
                   .astype('int64'))
        samples = tensor.minimum(samples, probs.shape[1] - 1)

        actor_feedback = self.generator.feedback.apply(samples, as_dict=True)
        actor_states_contexts = dict_subset(
            kwargs,
            self.generator.recurrent.apply.states
            + self.generator.recurrent.apply.contexts)
        actor_states_outputs = self.generator.recurrent.apply(
            as_dict=True, iterate=False,
            **dict_union(actor_feedback, actor_states_contexts))

        critic_feedback = critic.generator.feedback.apply(samples, as_dict=True)
        critic_states_contexts = {
            name: kwargs['critic_' + name]
            for name in
            critic.generator.recurrent.apply.states
            + critic.generator.recurrent.apply.contexts}
        critic_apply_kwargs = dict(
            as_dict=True, iterate=False,
            **dict_union(critic_feedback, critic_states_contexts))
        if self.generator.readout.critic_uses_actor_states:
            critic_apply_kwargs['extra_inputs'] = actor_states_outputs['states']
        critic_states_outputs = critic.generator.recurrent.apply(**critic_apply_kwargs)
        return ([samples, step + 1]
                + actor_states_outputs.values()
                + critic_states_outputs.values())

    @application
    def initial_states(self, batch_size, *args, **kwargs):
        critic = self.generator.readout.critic
        return ([tensor.zeros((batch_size,), dtype='int64')]
                + self.generator.initial_states(batch_size, *args, **kwargs)
                + critic.generator.initial_states(
                      batch_size, ** {name[7:]: kwargs[name]
                                      for name in kwargs if name.startswith('critic_')}))

    def get_dim(self, name):
        critic = self.generator.readout.critic
        if name.startswith('critic_'):
            return critic.generator.get_dim(name[7:])
        elif name == 'step':
            return 0
        else:
            return self.generator.get_dim(name)

    @application
    def mask_for_prediction(self, prediction):
        prediction_mask = tensor.lt(
            tensor.cumsum(tensor.eq(prediction, self.eos_label)
                          .astype(theano.config.floatX), axis=0),
            1).astype(theano.config.floatX)
        prediction_mask = tensor.roll(prediction_mask, 1, 0)
        prediction_mask = tensor.set_subtensor(
            prediction_mask[0, :], tensor.ones_like(prediction_mask[0, :]))
        return prediction_mask

    def load_params(self, path):
        cg = self.get_cost_graph()
        with open(path, 'r') as src:
            param_values = load_parameters(src)
        Model(cg.outputs).set_parameter_values(param_values)

    def get_generate_graph(self, use_mask=True, n_steps=None,
                           return_initial_states=False,
                           use_softmax_t=False):
        if use_softmax_t:
            self.generator.readout.softmax_t = self.criterion.get('softmax_t', 1.0)
        inputs_mask = None
        if use_mask:
            inputs_mask = self.inputs_mask
        result = self.generate(
            n_steps=n_steps, inputs_mask=inputs_mask,
            return_initial_states=return_initial_states,
             **self.inputs)

        self.generator.readout.softmax_t = 1.
        return result

    def get_mixed_generate_graph(self, n_steps=None,
                                 return_initial_states=False):
        critic = self.generator.readout.critic

        attended, attended_mask = self.encoder.apply(
            input_=self.bottom.apply(**self.inputs),
            mask=self.inputs_mask)
        attended = self.top.apply(attended)

        critic_attended, critic_attended_mask = critic.encoder.apply(
            input_=critic.bottom.apply(inputs=self.labels),
            mask=self.labels_mask)
        critic_attended = critic.top.apply(critic_attended)

        return self.mixed_generate(
            n_steps=n_steps, batch_size=attended.shape[1],
            return_initial_states=return_initial_states, as_dict=True,
            attended=attended, attended_mask=attended_mask,
            critic_attended=critic_attended, critic_attended_mask=critic_attended_mask,
            groundtruth=self.labels, groundtruth_mask=self.labels_mask)


    def get_cost_graph(self, batch=True, use_prediction=False,
                       training=False, groundtruth_as_predictions=False,
                       with_mixed_generation=False):
        # "use_predictions" means use the Theano input variable
        # for predictions.
        readout = self.generator.readout
        if training and self.criterion['name'] == 'actor_critic':
            logger.debug("Switching to training mode")
            readout.compute_targets = self.compute_targets
            readout.compute_policy = self.compute_policy
            if 'solve_bellman' in self.criterion:
                readout.solve_bellman = self.criterion['solve_bellman']
        if with_mixed_generation and 'epsilon' in self.criterion:
            readout.epsilon = self.criterion['epsilon']

        if batch:
            inputs, inputs_mask = self.inputs, self.inputs_mask
            groundtruth, groundtruth_mask = self.labels, self.labels_mask
            prediction, prediction_mask = self.predicted_labels, self.predicted_mask
        else:
            inputs, inputs_mask = self.bottom.single_to_batch_inputs(
                self.single_inputs)
            groundtruth = self.single_labels[:, None]
            groundtruth_mask = self.mask_for_prediction(groundtruth)
            prediction = self.single_predicted_labels[:, None]
            prediction_mask = self.mask_for_prediction(prediction)
        if self.cost_involves_generation() and not groundtruth_as_predictions:
            if ((training and self.generate_predictions) or
                    (not training and not use_prediction)):
                generation_routine = (self.get_mixed_generate_graph
                                      if with_mixed_generation
                                      else self.get_generate_graph)
                generated = generation_routine(
                    n_steps=self.labels.shape[0] + self.extra_generation_steps)
                prediction = disconnected_grad(generated['samples'])
                prediction_mask = self.mask_for_prediction(prediction)
            else:
                logger.debug("Using provided predictions")
            cost = self.costs(inputs_mask=inputs_mask,
                 prediction=prediction, prediction_mask=prediction_mask,
                 groundtruth=groundtruth, groundtruth_mask=groundtruth_mask,
                 **inputs)
        else:
            if use_prediction:
                cost = self.costs(inputs_mask=inputs_mask,
                    prediction=prediction, prediction_mask=prediction_mask,
                    **inputs)
            else:
                cost = self.costs(inputs_mask=inputs_mask,
                    prediction=groundtruth, prediction_mask=groundtruth_mask,
                    groundtruth=groundtruth, groundtruth_mask=groundtruth_mask,
                    **inputs)
        cost_cg = ComputationGraph(cost)

        # This *has to* be done only when
        # "training" or "with_mixed_generation" is True,
        # but it does not hurt to do it every time.
        logger.debug("Switching back to the normal mode")
        readout = self.generator.readout
        readout.compute_targets = True
        readout.compute_policy = True
        readout.solve_bellman = True
        readout.epsilon = 0.

        return cost_cg

    def analyze(self, inputs, groundtruth, prediction):
        """Compute cost and aligment."""
        if not hasattr(self, "_analyze"):
            input_variables = list(self.single_inputs.values())
            input_variables.append(self.single_labels)
            input_variables.append(self.single_predicted_labels)

            cg = self.get_cost_graph(batch=False, use_prediction=True)
            costs = cg.outputs[0]

            weights, = VariableFilter(
                bricks=[self.generator], name="weights")(cg)
            energies = VariableFilter(
                bricks=[self.generator], name="energies")(cg)
            energies_output = [energies[0][:, 0, :] if energies
                               else tensor.zeros_like(weights)]

            self._analyze = theano.function(
                input_variables,
                [costs[0], weights[:, 0, :]] + energies_output,
                on_unused_input='warn')

        input_values_dict = dict(inputs)
        input_values_dict['labels'] = groundtruth
        input_values_dict['predicted_labels'] = prediction
        return self._analyze(**input_values_dict)

    def init_beam_search(self, beam_size):
        """Compile beam search and set the beam size.

        See Blocks issue #500.

        """
        if hasattr(self, '_beam_search') and self.beam_size == beam_size:
            # Only recompile if the user wants a different beam size
            return
        self.beam_size = beam_size
        generated = self.get_generate_graph(use_mask=False, n_steps=3)
        cg = ComputationGraph(generated.values())
        samples, = VariableFilter(
            applications=[self.generator.generate], name="samples")(cg)
        self._beam_search = BeamSearch(beam_size, samples)
        self._beam_search.compile()

    def beam_search(self, inputs, **kwargs):
        # When a recognizer is unpickled, self.beam_size is available
        # but beam search has to be recompiled.

        self.init_beam_search(self.beam_size)
        inputs = dict(inputs)
        max_length = int(self.bottom.num_time_steps(**inputs) /
                         self.max_decoded_length_scale)
        search_inputs = {}
        for var in self.inputs.values():
            search_inputs[var] = inputs.pop(var.name)[:, numpy.newaxis, ...]
        if inputs:
            raise Exception(
                'Unknown inputs passed to beam search: {}'.format(
                    inputs.keys()))
        outputs, search_costs = self._beam_search.search(
            search_inputs, self.eos_label,
            max_length,
            ignore_first_eol=self.data_prepend_eos,
            **kwargs)
        return outputs, search_costs

    def init_generate(self):
        generated = self.get_generate_graph(use_mask=False)
        cg = ComputationGraph(generated['samples'])
        self._do_generate = cg.get_theano_function()

    def sample(self, inputs, n_steps=None):
        if not hasattr(self, '_do_generate'):
            self.init_generate()
        batch, unused_mask = self.bottom.single_to_batch_inputs(inputs)
        batch['n_steps'] = n_steps if n_steps is not None \
            else int(self.bottom.num_time_steps(**batch) /
                     self.max_decoded_length_scale)
        sample = self._do_generate(**batch)[0]
        sample = list(sample[:, 0])
        if self.eos_label in sample:
            sample = sample[:sample.index(self.eos_label) + 1]
        return sample

    def __getstate__(self):
        state = dict(self.__dict__)
        for attr in ['_analyze', '_beam_search']:
            state.pop(attr, None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # To use bricks used on a GPU first on a CPU later
        try:
            emitter = self.generator.readout.emitter
            del emitter._theano_rng
        except:
            pass

    def cost_involves_generation(self):
        return self.criterion['name'] in ['reinforce', 'sarsa', 'actor_critic']
