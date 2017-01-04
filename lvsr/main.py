from __future__ import print_function
import time
import logging
import pprint
import math
import os
import re
import cPickle as pickle
import sys
import yaml
import copy
from collections import OrderedDict

import numpy
from lvsr.algorithms import BurnIn
from blocks_extras.extensions.embed_ipython import EmbedIPython
import theano
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams
from blocks.bricks.lookup import LookupTable
from blocks.graph import ComputationGraph, apply_dropout, apply_noise
from blocks.algorithms import (GradientDescent,
                               StepClipping, CompositeRule,
                               Momentum, RemoveNotFinite, AdaDelta,
                               Restrict, VariableClipping, RMSProp,
                               Adam)
from blocks.monitoring import aggregation
from blocks.monitoring.aggregation import MonitoredQuantity
from blocks.theano_expressions import l2_norm
from blocks.extensions import (
    FinishAfter, Printing, Timing, ProgressBar, SimpleExtension,
    TrainingExtension)
from blocks.extensions.saveload import Checkpoint, Load
from blocks.extensions.monitoring import (
    TrainingDataMonitoring, DataStreamMonitoring)
from blocks_extras.extensions.plot import Plot
from blocks.extensions.training import TrackTheBest
from blocks.extensions.predicates import OnLogRecord
from blocks.log import TrainingLog
from blocks.model import Model
from blocks.main_loop import MainLoop
from blocks.filter import VariableFilter, get_brick
from blocks.roles import WEIGHT, OUTPUT
from blocks.utils import reraise_as, dict_subset
from blocks.select import Selector

from lvsr.bricks.recognizer import EncoderDecoder
from lvsr.datasets import Data
from lvsr.expressions import (
    monotonicity_penalty, entropy, weights_std)
from lvsr.extensions import (
    CGStatistics, AdaptiveClipping, GeneratePredictions, Patience,
    CodeVersion)
from lvsr.error_rate import wer, _bleu
from lvsr.graph import apply_adaptive_noise
from lvsr.utils import rename
from blocks.serialization import load_parameters, continue_training
from lvsr.log_backends import NDarrayLog
from lvsr.beam_search import BeamSearch

floatX = theano.config.floatX
logger = logging.getLogger(__name__)


def _gradient_norm_is_none(log):
    return math.isnan(log.current_row.get('total_gradient_norm', 0))


class PhonemeErrorRate(MonitoredQuantity):

    def __init__(self, recognizer, data, metric, beam_size,
                 char_discount=None, round_to_inf=None, stop_on=None,
                 consider_all_eos=None,
                 **kwargs):
        self.recognizer = recognizer
        self.metric = metric
        self.beam_size = beam_size
        self.char_discount = char_discount
        self.round_to_inf = round_to_inf
        self.stop_on = stop_on
        self.consider_all_eos = consider_all_eos
        # Will only be used to decode generated outputs,
        # which is necessary for correct scoring.
        self.data = data
        kwargs.setdefault('name', 'per')
        kwargs.setdefault('requires', (self.recognizer.single_inputs.values() +
                                       [self.recognizer.single_labels]))
        super(PhonemeErrorRate, self).__init__(**kwargs)

        if not self.metric in ['per', 'bleu']:
            raise ValueError

        self.recognizer.init_beam_search(self.beam_size)

    def initialize(self):
        self.total_length = 0.
        self.total_recognized_length = 0.
        self.decoded = []

        # BLEU
        self.total_correct = numpy.zeros(4)
        self.total_possible = numpy.zeros(4)

        # Edit distance
        self.total_errors = 0.
        self.num_examples = 0


    def aggregate(self, *args):
        input_vars = self.requires[:-1]
        beam_inputs = {var.name: val for var, val in zip(input_vars,
                                                         args[:-1])}
        transcription = args[-1]
        data = self.data
        groundtruth = data.decode(transcription)
        search_kwargs = dict(
            char_discount=self.char_discount,
            round_to_inf=self.round_to_inf,
            stop_on=self.stop_on,
            validate_solution_function=getattr(
                data.info_dataset, 'validate_solution', None),
            consider_all_eos=self.consider_all_eos)
        # We rely on the defaults hard-coded in BeamSearch
        search_kwargs = {k: v for k, v in search_kwargs.items() if v}
        outputs, search_costs = self.recognizer.beam_search(
            beam_inputs, **search_kwargs)
        recognized = data.decode(outputs[0])
        self.decoded.append(outputs[0])

        self.total_length += len(groundtruth)
        self.total_recognized_length += len(recognized)
        self.num_examples += 1
        if self.metric == 'per':
            error = min(1, wer(groundtruth, recognized))
            self.total_errors += error * len(groundtruth)
            self.mean_error = self.total_errors / self.total_length
        elif self.metric == 'bleu':
            _, correct, _, _  = _bleu(groundtruth, recognized)
            self.total_correct += correct
            self.total_possible += numpy.array([len(recognized) - i for i in range(4)])

    def get_aggregated_value(self):
        if self.metric == 'per':
            return self.mean_error
        elif self.metric == 'bleu':
            logger.debug('Precisions {}'.format(self.total_correct / self.total_possible))
            logger.debug('Total recognized length: {}'.format(self.total_recognized_length))
            logger.debug('Total groundtruth length: {}'.format(self.total_length))
            brevity_penalty = numpy.exp(min(0.0, 1 - self.total_length / float(self.total_recognized_length)))
            logger.debug('Brevity penalty: {}'.format(brevity_penalty))
            bleu = (self.total_correct / self.total_possible).prod() ** 0.25
            bleu *= brevity_penalty
            return 100 *bleu


class _OutputDecoded(SimpleExtension):

    def __init__(self, data, per, decoded_path, **kwargs):
        self.data = data
        self.per = per
        self.decoded_path = decoded_path
        super(_OutputDecoded, self).__init__(**kwargs)

    def do(self, *args, **kwargs):
        if not os.path.exists(self.decoded_path):
            os.mkdir(self.decoded_path)
        with open(os.path.join(
            self.decoded_path,
            str(self.main_loop.status['iterations_done'])), 'w') as dst:
                for output in self.per.decoded:
                    print(self.data.pretty_print(output, None), file=dst)


class Strings(MonitoredQuantity):
    def __init__(self, data, **kwargs):
        super(Strings, self).__init__(**kwargs)
        self.data = data

    def initialize(self):
        self.result = None

    def aggregate(self, string):
        self.result = [
            self.data.monospace_print(string[:, i])
            for i in range(string.shape[1])]

    def get_aggregated_value(self):
        return self.result


class SwitchOffLengthFilter(SimpleExtension):

    def __init__(self, length_filter, **kwargs):
        self.length_filter = length_filter
        super(SwitchOffLengthFilter, self).__init__(**kwargs)

    def do(self, *args, **kwargs):
        self.length_filter.max_length = None
        self.main_loop.log.current_row['length_filter_switched'] = True


class LoadLog(TrainingExtension):
    """Loads a the log from the checkoint.

    Makes a `LOADED_FROM` record in the log with the dump path.

    Parameters
    ----------
    path : str
        The path to the folder with dump.

    """
    def __init__(self, path, **kwargs):
        super(LoadLog, self).__init__(**kwargs)
        self.path = path[:-4] + '_log.zip'

    def load_to(self, main_loop):

        with open(self.path, "rb") as source:
            loaded_log = pickle.load(source)
            #TODO: remove and fix the printing issue!
            loaded_log.status['resumed_from'] = None
            #make sure that we start a new epoch
            if loaded_log.status.get('epoch_started'):
                logger.warn('Loading a snaphot taken during an epoch. '
                            'Iteration information will be destroyed!')
                loaded_log.status['epoch_started'] = False
        main_loop.log = loaded_log

    def before_training(self):
        if not os.path.exists(self.path):
            logger.warning("No log dump found")
            return
        logger.info("loading log from {}".format(self.path))
        try:
            self.load_to(self.main_loop)
            #self.main_loop.log.current_row[saveload.LOADED_FROM] = self.path
        except Exception:
            reraise_as("Failed to load the state")


def create_model(config, data,
                 load_path=None,
                 test_tag=False):
    """
    Build the main brick and initialize or load all parameters.

    Parameters
    ----------

    config : dict
        the configuration dict

    data : object of class Data
        the dataset creation object

    load_path : str or None
        if given a string, it will be used to load model parameters. Else,
        the parameters will be randomly initalized by calling
        recognizer.initialize()

    test_tag : bool
        if true, will add tag the input variables with test values

    """
    # First tell the recognizer about required data sources
    net_config = dict(config["net"])
    train_config = dict(config["training"])
    bottom_class = net_config['bottom']['bottom_class']
    input_dims = {
        source: data.num_features(source)
        for source in bottom_class.vector_input_sources}
    input_num_chars = {
        source: len(data.token_map(source))
        for source in bottom_class.discrete_input_sources}

    recognizer = EncoderDecoder(
        input_dims=input_dims,
        input_num_chars=input_num_chars,
        bos_label=data.bos_label,
        eos_label=data.eos_label,
        num_labels=data.num_labels,
        name="recognizer",
        data_prepend_eos=data.prepend_eos,
        token_map=data.token_map('labels'),
        generate_predictions=not train_config.get('external_predictions', False),
        compute_targets=not train_config.get('external_targets', False),
        extra_generation_steps=train_config.get('extra_generation_steps'),
        **net_config)
    if load_path:
        recognizer.load_params(load_path)
    else:
        for brick_path, attribute_dict in sorted(
                config['initialization'].items(),
                key=lambda (k, v): k.count('/')):
            for attribute, value in attribute_dict.items():
                brick, = Selector(recognizer).select(brick_path).bricks
                setattr(brick, attribute, value)
                brick.push_initialization_config()
        recognizer.initialize()

    if test_tag:
        stream = data.get_stream("train")
        data = next(stream.get_epoch_iterator(as_dict=True))
        for var in (recognizer.inputs.values() +
                      [recognizer.inputs_mask, recognizer.labels, recognizer.labels_mask]):
            var.tag.test_value = data[var.name]
        theano.config.compute_test_value = 'warn'
        theano.config.print_test_value = True
    return recognizer


def initialize_all(config, save_path, bokeh_name,
                   params, bokeh_server, bokeh, test_tag, use_load_ext,
                   load_log, fast_start):
    root_path, extension = os.path.splitext(save_path)

    data = Data(**config['data'])
    train_conf = config['training']
    mon_conf = config['monitoring']
    recognizer = create_model(config, data,
                              test_tag=test_tag)
    step_number = theano.shared(0)

    # Separate attention_params to be handled differently
    # when regularization is applied
    attention = recognizer.generator.recurrent.attention
    attention_params = Selector(attention).get_parameters().values()

    logger.info(
        "Initialization schemes for all bricks.\n"
        "Works well only in my branch with __repr__ added to all them,\n"
        "there is an issue #463 in Blocks to do that properly.")
    def show_init_scheme(cur):
        result = dict()
        for attr in dir(cur):
            if attr.endswith('_init'):
                result[attr] = getattr(cur, attr)
        for child in cur.children:
            result[child.name] = show_init_scheme(child)
        return result
    logger.info(pprint.pformat(show_init_scheme(recognizer)))

    cg = recognizer.get_cost_graph(batch=True, training=True)
    labels, = VariableFilter(
        applications=[recognizer.costs], name='prediction')(cg)
    labels_mask, = VariableFilter(
        applications=[recognizer.costs], name='prediction_mask')(cg)
    batch_cost = cg.outputs[0].sum()
    batch_size = rename(recognizer.labels.shape[1], "batch_size")
    # Assumes constant batch size. `aggregation.mean` is not used because
    # of Blocks #514.
    cost = batch_cost / batch_size
    cost.name = "sequence_total_cost"
    logger.info("Cost graph is built")

    # Fetch variables useful for debugging.
    # It is important not to use any aggregation schemes here,
    # as it's currently impossible to spread the effect of
    # regularization on their variables, see Blocks #514.
    cg = ComputationGraph(cost)
    r = recognizer
    bottom_output = VariableFilter(
        # We need name_regex instead of name because LookupTable calls itsoutput output_0
        applications=[r.bottom.apply], name_regex="output")(
            cg)[-1]
    attended, = VariableFilter(
        applications=[r.generator.recurrent.apply], name="attended")(
            cg)
    attended_mask, = VariableFilter(
        applications=[r.generator.recurrent.apply], name="attended_mask")(
            cg)
    weights, = VariableFilter(
        applications=[r.generator.costs], name="weights")(
            cg)
    max_recording_length = rename(bottom_output.shape[0],
                                  "max_recording_length")
    # To exclude subsampling related bugs
    max_attended_mask_length = rename(attended_mask.shape[0],
                                      "max_attended_mask_length")
    max_attended_length = rename(attended.shape[0],
                                 "max_attended_length")
    max_num_phonemes = rename(labels.shape[0],
                              "max_num_phonemes")
    mean_attended = rename(abs(attended).mean(),
                           "mean_attended")
    mean_bottom_output = rename(abs(bottom_output).mean(),
                                "mean_bottom_output")
    weights_penalty = rename(monotonicity_penalty(weights, labels_mask),
                             "weights_penalty")
    weights_entropy = rename(entropy(weights, labels_mask),
                             "weights_entropy")
    mask_density = rename(labels_mask.mean(),
                          "mask_density")

    # Observables:
    primary_observables = []  # monitored each batch
    secondary_observables = []  # monitored every 10 batches
    validation_observables = []  # monitored on the validation set
    verbosity = config['monitoring'].get('verbosity', 0.)

    secondary_observables = [
        weights_penalty, weights_entropy,
        mean_attended, mean_bottom_output,
        batch_size, max_num_phonemes,
        mask_density]

    # Regularization. It is applied explicitly to all variables
    # of interest, it could not be applied to the cost only as it
    # would not have effect on auxiliary variables, see Blocks #514.
    reg_config = config.get('regularization', dict())
    regularized_cg = ComputationGraph([cost] + secondary_observables)
    if reg_config.get('dropout'):
        logger.info('apply dropout')
        regularized_cg = apply_dropout(cg, [bottom_output], 0.5)
    if reg_config.get('noise'):
        logger.info('apply noise')
        noise_subjects = [p for p in cg.parameters if p not in attention_params]
        regularized_cg = apply_noise(cg, noise_subjects, reg_config['noise'])

    train_cost = regularized_cg.outputs[0]
    if reg_config.get("penalty_coof", .0) > 0:
        # big warning!!!
        # here we assume that:
        # regularized_weights_penalty = regularized_cg.outputs[1]
        train_cost = (train_cost +
                      reg_config.get("penalty_coof", .0) *
                      regularized_cg.outputs[1] / batch_size)
    if reg_config.get("decay", .0) > 0:
        logger.debug("Using weight decay of {}".format(reg_config['decay']))
        train_cost = (train_cost + reg_config.get("decay", .0) *
                      l2_norm(VariableFilter(roles=[WEIGHT])(cg.parameters)) ** 2)

    gradients = None
    if reg_config.get('adaptive_noise'):
        logger.info('apply adaptive noise')
        if ((reg_config.get("penalty_coof", .0) > 0) or
                (reg_config.get("decay", .0) > 0)):
            logger.error('using  adaptive noise with alignment weight panalty '
                         'or weight decay is probably stupid')
        train_cost, regularized_cg, gradients, noise_brick = apply_adaptive_noise(
            cg, cg.outputs[0],
            variables=cg.parameters,
            num_examples=data.get_dataset('train').num_examples,
            parameters=Model(regularized_cg.outputs[0]).get_parameter_dict().values(),
            **reg_config.get('adaptive_noise')
        )
        adapt_noise_cg = ComputationGraph(train_cost)
        model_prior_mean = rename(
            VariableFilter(applications=[noise_brick.apply],
                           name='model_prior_mean')(adapt_noise_cg)[0],
            'model_prior_mean')
        model_cost = rename(
            VariableFilter(applications=[noise_brick.apply],
                           name='model_cost')(adapt_noise_cg)[0],
            'model_cost')
        model_prior_variance = rename(
            VariableFilter(applications=[noise_brick.apply],
                           name='model_prior_variance')(adapt_noise_cg)[0],
            'model_prior_variance')
        regularized_cg = ComputationGraph(
            [train_cost, model_cost] +
            regularized_cg.outputs +
            [model_prior_mean, model_prior_variance])
        primary_observables += [
            regularized_cg.outputs[1],  # model cost
            regularized_cg.outputs[2],  # task cost
            regularized_cg.outputs[-2],  # model prior mean
            regularized_cg.outputs[-1]]  # model prior variance

    # Additional components of the costs required for some criterions
    if config['net']['criterion']['name'] == 'reinforce':
        readout = r.generator.readout
        baselines, = VariableFilter(
            bricks=[readout], name='baselines')(regularized_cg)
        baseline_errors, = VariableFilter(
            bricks=[readout], name='baseline_errors')(regularized_cg)
        mean_baseline = rename(baselines.mean(),
                               'mean_baseline')
        mean_baseline_error = rename(baseline_errors.sum(axis=0).mean(),
                                     'mean_baseline_error')
        train_cost = (train_cost * config['net']['criterion'].get('train_cost_coof', 1.0)
                      + mean_baseline_error)

    # Add log-likelihood of the groundtruth to the cost
    if config['net']['criterion'].get('also_minimize_loglikelihood'):
        logger.info("Also add log-likelihood to the cost")
        groundtruth_cg = recognizer.get_cost_graph(
            training=False, use_prediction=False, groundtruth_as_predictions=True)
        prediction_log_probs, = VariableFilter(
            bricks=[r.generator.readout], name='prediction_log_probs')(groundtruth_cg)
        groundtruth_mask, = VariableFilter(
            bricks=[r.generator.readout], name='groundtruth_mask')(groundtruth_cg)
        log_likelihood = (prediction_log_probs * groundtruth_mask).sum(axis=0).mean()
        train_cost -= log_likelihood

    # Build the model and load parameters if necessary
    train_cost.name = 'train_cost'
    model = Model(train_cost)
    if params:
        logger.info("Load parameters from " + params)
        # please note: we cannot use recognizer.load_params
        # as it builds a new computation graph that dies not have
        # shapred variables added by adaptive weight noise
        with open(params, 'r') as src:
            param_values = load_parameters(src)
        model.set_parameter_values(param_values)
    parameters = model.get_parameter_dict()
    def _log_parameters(message, ps):
        logger.info(message + "\n" +
                    pprint.pformat(
                        [(key, parameters[key].get_value().shape) for key
                        in sorted(ps.keys())],
                        width=120))
    _log_parameters("Parameters", parameters)

    # Define the training algorithm.
    trainable_parameters = OrderedDict(
        [(key, value) for key, value in parameters.items()
         if re.match(train_conf.get('trainable_regexp', '.*'), key)])
    if trainable_parameters.keys() != parameters.keys():
        _log_parameters("Trainable parameters", trainable_parameters)

    if train_conf['gradient_threshold']:
        clipping = StepClipping(train_conf['gradient_threshold'])
        clipping.threshold.name = "gradient_norm_threshold"
        clipping = [clipping]
    else:
        clipping = []
    rule_names = train_conf.get('rules', ['momentum'])
    core_rules = []
    if 'momentum' in rule_names:
        logger.info("Using scaling and momentum for training")
        core_rules.append(
            Momentum(train_conf['scale'], train_conf['momentum']))
    if 'adadelta' in rule_names:
        logger.info("Using AdaDelta for training")
        core_rules.append(
            AdaDelta(train_conf['decay_rate'], train_conf['epsilon']))
    if 'rmsprop' in rule_names:
        logger.info("Using RMSProp for training")
        core_rules.append(
            RMSProp(train_conf['scale'], train_conf['decay_rate'],
                    train_conf['max_scaling']))
    if 'adam' in rule_names:
        logger.info("Using ADAM for training")
        core_rules.append(Adam(
            train_conf['scale'],
            1 - train_conf['momentum'],
            1 - train_conf['decay_rate'],
            epsilon=train_conf['epsilon']))
    max_norm_rules = []
    if reg_config.get('max_norm', False) > 0:
        logger.info("Apply MaxNorm")
        maxnorm_subjects = VariableFilter(roles=[WEIGHT])(trainable_parameters)
        if reg_config.get('max_norm_exclude_lookup', False):
            maxnorm_subjects = [v for v in maxnorm_subjects
                                if not isinstance(get_brick(v), LookupTable)]
        logger.info(
            "Parameters covered by MaxNorm:\n"
             + pprint.pformat([name for name, p in trainable_parameters.items()
                               if p in maxnorm_subjects]))
        logger.info(
            "Parameters NOT covered by MaxNorm:\n"
             + pprint.pformat([name for name, p in trainable_parameters.items()
                               if not p in maxnorm_subjects]))
        max_norm_rules = [
            Restrict(VariableClipping(reg_config['max_norm'], axis=0),
                     maxnorm_subjects)]
    burn_in = []
    if train_conf.get('burn_in_steps', 0):
        burn_in.append(
            BurnIn(num_steps=train_conf['burn_in_steps']))
    algorithm = GradientDescent(
        cost=train_cost,
        parameters=trainable_parameters.values(),
        gradients=gradients,
        step_rule=CompositeRule(
            clipping + core_rules + max_norm_rules +
            # Parameters are not changed at all
            # when nans are encountered.
            [RemoveNotFinite(0.0)] + burn_in),
        on_unused_sources='warn')
    if regularized_cg.updates:
        logger.debug("There are updates in the computation graph")
        algorithm.updates.extend(regularized_cg.updates.items())
    algorithm.updates.append((step_number, step_number + 1))

    logger.debug("Scan Ops in the gradients")
    gradient_cg = ComputationGraph(algorithm.gradients.values())
    for op in ComputationGraph(gradient_cg).scans:
        logger.debug(op)

    # More variables for debugging: some of them can be added only
    # after the `algorithm` object is created.
    primary_observables += [
        train_cost,
        max_recording_length,
        max_attended_length, max_attended_mask_length]
    if clipping:
        primary_observables += [
            algorithm.total_gradient_norm,
            algorithm.total_step_norm,
            clipping[0].threshold]

    secondary_observables = list(regularized_cg.outputs)
    if not 'train_cost' in [v.name for v in secondary_observables]:
        secondary_observables += [train_cost]
    if clipping:
        secondary_observables += [
            algorithm.total_step_norm, algorithm.total_gradient_norm,
            clipping[0].threshold]
    if mon_conf.get('monitor_parameters'):
        for name, param in parameters.items():
            num_elements = numpy.product(param.get_value().shape)
            norm = param.norm(2) / num_elements ** 0.5
            grad_norm = algorithm.gradients[param].norm(2) / num_elements ** 0.5
            step_norm = algorithm.steps[param].norm(2) / num_elements ** 0.5
            stats = tensor.stack(norm, grad_norm, step_norm, step_norm / grad_norm)
            stats.name = name + '_stats'
            secondary_observables.append(stats)

    # Fetch variables that make sense only for some criteria.
    # Add respective observables
    cost_to_track = cost
    choose_best = min
    if r.cost_involves_generation():
        readout = r.generator.readout
        rewards, = VariableFilter(
            bricks=[readout], name='rewards')(regularized_cg)
        mean_total_reward = rename(rewards.sum(axis=0).mean(), 'mean_total_reward')
        primary_observables += [mean_total_reward]
        if verbosity >= 1:
            primary_observables += [aggregation.take_last(rewards)]
        secondary_observables += [
            Strings(data, requires=[r.labels], name='groundtruth'),
            Strings(data, requires=[labels], name='predictions')]
    if r.criterion['name'] == 'reinforce':
        baselines, = VariableFilter(
            bricks=[readout], name='baselines')(regularized_cg)
        log_probs, = VariableFilter(
            bricks=[readout], name='log_probs')(regularized_cg)
        baseline_errors, = VariableFilter(
            bricks=[readout], name='baseline_errors')(regularized_cg)
        est_entropy = rename(log_probs.sum(axis=0).mean(), 'entropy')
        primary_observables += [est_entropy, mean_baseline, mean_baseline_error]
        if verbosity >= 1:
            primary_observables += [aggregation.take_last(baselines)]
        rewards, = VariableFilter(
            bricks=[readout], name='rewards')(regularized_cg)
        validation_observables += [
            rename(rewards.sum(axis=0).mean(), 'mean_total_reward')]
        validation_updates = cg.updates
    if r.criterion['name'] in ['sarsa', 'actor_critic']:
        value_biases, = VariableFilter(
            bricks=[readout], name='value_biases')(regularized_cg)
        prediction_mask, = VariableFilter(
            bricks=[readout], name='prediction_mask')(regularized_cg)
        prediction_values, = VariableFilter(
            bricks=[readout], name='prediction_values')(regularized_cg)
        prediction_outputs, = VariableFilter(
            bricks=[readout], name='prediction_outputs')(regularized_cg)
        probs, = VariableFilter(
            applications=[readout.costs], name='probs')(regularized_cg)
        value_targets, = VariableFilter(
            bricks=[readout], name='value_targets')(regularized_cg)
        values, = VariableFilter(
            applications=[readout.costs], name='values')(regularized_cg)
        outputs, = VariableFilter(
            bricks=[readout], name='outputs')(regularized_cg)
        last_character_costs, = VariableFilter(
            bricks=[readout], name='last_character_costs')(regularized_cg)
        mean_expected_reward, = VariableFilter(
            bricks=[readout], name='mean_expected_reward')(regularized_cg)
        mean_last_character_cost = rename(
            last_character_costs.mean(),
            'mean_last_character_cost')
        mean_action_entropy, = VariableFilter(
            bricks=[readout], name='mean_actor_entropy')(regularized_cg)
        mean2_output, = VariableFilter(
            bricks=[readout], name='mean2_output')(regularized_cg)
        max_output, = VariableFilter(
            bricks=[readout], name='max_output')(regularized_cg)
        primary_observables += [
            mean_expected_reward, mean_last_character_cost,
            mean2_output, max_output,
            mean_action_entropy]
        if verbosity >= 1:
            primary_observables += map(aggregation.take_last,
                [prediction_mask, prediction_values, prediction_outputs,
                 probs, value_biases, outputs, values, value_targets])
        # Note, that we build a "clean" cg for the validation.
        # In particular, it contains not dangling free variables
        # like "value_targets", probs, etc.
        clean_cg = recognizer.get_cost_graph(batch=True)
        clean_rewards, = VariableFilter(
            bricks=[readout], name='rewards')(clean_cg)
        validation_observables += [
            rename(clean_rewards.sum(axis=0).mean(), 'mean_total_reward')]
        cost_to_track = validation_observables[-1]
        choose_best = max
        validation_updates = clean_cg.updates
        # In addition we monitoring the rewards of a mixed policy
        mixed_cg = recognizer.get_cost_graph(batch=True, with_mixed_generation=True)
        mixed_rewards, = VariableFilter(
            bricks=[readout], name='rewards')(mixed_cg)
        mixed_validation_observables = [
                rename(mixed_rewards.sum(axis=0).mean(), 'mean_total_reward')
            ]
        mixed_validation_updates = mixed_cg.updates
    if r.criterion['name'] == 'actor_critic':
        mean_critic_cost, = VariableFilter(
            bricks=[readout], name='mean_critic_cost')(regularized_cg)
        mean_critic_monte_carlo_cost, = VariableFilter(
            bricks=[readout], name='mean_critic_monte_carlo_cost')(regularized_cg)
        mean_actor_cost, = VariableFilter(
            bricks=[readout], name='mean_actor_cost')(regularized_cg)
        primary_observables += [
            mean_critic_cost, mean_critic_monte_carlo_cost, mean_actor_cost]
    if r.criterion['name'] in ['log_likelihood', 'reinforce']:
        energies, = VariableFilter(
            applications=[r.generator.readout.all_scores], roles=[OUTPUT])(
                cg)
        min_energy = rename(energies.min(), "min_energy")
        max_energy = rename(energies.max(), "max_energy")
        secondary_observables += [min_energy, max_energy]
    if r.criterion['name'] == 'log_likelihood':
        validation_observables += [
            rename(aggregation.mean(batch_cost, batch_size), cost.name),
            weights_entropy, weights_penalty]
        validation_updates = cg.updates

    def attach_aggregation_schemes(variables):
        # Attaches non-trivial aggregation schemes to
        # secondary and validation observables
        result = []
        for var in variables:
            if var.name == 'weights_penalty':
                result.append(rename(aggregation.mean(var, batch_size),
                                     'weights_penalty_per_recording'))
            elif var.name == 'weights_entropy':
                result.append(rename(aggregation.mean(var, labels_mask.sum()),
                                     'weights_entropy_per_label'))
            else:
                result.append(var)
        return result

    if verbosity >= 2:
        # Override the frequencies
        mon_conf['primary_freq'] = 1
        mon_conf['secondary_freq'] = 1


    # Build main loop.
    logger.info("Initialize extensions")
    extensions = []
    if use_load_ext and params:
        extensions.append(Load(params, load_iteration_state=True, load_log=True))
    if load_log and params:
        extensions.append(LoadLog(params))
    extensions += [
        Timing(every_n_batches=mon_conf['primary_freq']),
        CGStatistics(),
        CodeVersion(['lvsr']),
    ]

    # Monitoring
    extensions.append(TrainingDataMonitoring(
        primary_observables,
        every_n_batches=mon_conf.get('primary_freq', 1)))
    average_monitoring = TrainingDataMonitoring(
        attach_aggregation_schemes(secondary_observables),
        prefix="average",
        every_n_batches=mon_conf.get('secondary_freq', 10))
    extensions.append(average_monitoring)
    validation_requested = (
        mon_conf['validate_every_epochs'] or
        mon_conf['validate_every_batches'])
    if validation_requested:
        validation = DataStreamMonitoring(
            attach_aggregation_schemes(validation_observables),
            data.get_stream("valid", shuffle=False),
            prefix="valid", updates=validation_updates).set_conditions(
                before_first_epoch=not fast_start,
                every_n_epochs=mon_conf['validate_every_epochs'],
                every_n_batches=mon_conf['validate_every_batches'],
                after_training=False)
        track_the_best_cost = TrackTheBest(
            validation.record_name(cost_to_track),
            choose_best=choose_best).set_conditions(
                before_first_epoch=True,
                every_n_epochs=mon_conf['validate_every_epochs'],
                every_n_batches=mon_conf['validate_every_batches'])
        extensions.append(validation)
        extensions.append(track_the_best_cost)
        if r.criterion['name'] == 'actor_critic':
            mixed_validation = DataStreamMonitoring(
                mixed_validation_observables,
                data.get_stream("valid", shuffle=False),
                prefix="mixed_valid", updates=mixed_validation_updates).set_conditions(
                    before_first_epoch=not fast_start,
                    every_n_epochs=mon_conf['validate_every_epochs'],
                    every_n_batches=mon_conf['validate_every_batches'],
                    after_training=False)
            extensions.append(mixed_validation)
    search_config = config['monitoring'].get('search')
    search_requested = (search_config and (
        mon_conf['search_every_epochs'] or
        mon_conf['search_every_batches']))
    if search_requested:
        per = PhonemeErrorRate(
            recognizer, data,
            **config['monitoring']['search'])
        frequency_kwargs = dict(
            before_first_epoch=not fast_start,
            every_n_epochs=mon_conf['search_every_epochs'],
            every_n_batches=mon_conf['search_every_batches'],
            after_training=False)
        per_monitoring = DataStreamMonitoring(
            [per], data.get_stream("valid", batches=False, shuffle=False),
            prefix="valid").set_conditions(**frequency_kwargs)
        extensions.append(per_monitoring)
        track_the_best_per = TrackTheBest(
            per_monitoring.record_name(per),
            choose_best=min if search_config['metric'] == 'per' else max).set_conditions(
                **frequency_kwargs)
        extensions.append(track_the_best_per)
        extensions.append(_OutputDecoded(
            data, per, root_path + '_decoded',
            **frequency_kwargs
            ))

        if mon_conf.get('search_on_training'):
            # We reuse PhonemeErrorRate object here, should not cause problems
            training_per_monitoring = DataStreamMonitoring(
                [per], data.get_stream(
                    "train", batches=False, shuffle=False,
                    num_examples=mon_conf['search_on_training']),
                prefix="train").set_conditions(**frequency_kwargs)
            track_the_best_training_per = TrackTheBest(
                training_per_monitoring.record_name(per),
                choose_best=min if search_config['metric'] == 'per' else max).set_conditions(
                    **frequency_kwargs)
            extensions.append(training_per_monitoring)
            extensions.append(track_the_best_training_per)
            extensions.append(_OutputDecoded(
                data, per, root_path + '_train_decoded',
                **frequency_kwargs))


    # Training control
    if train_conf.get('external_predictions'):
        extensions.append(GeneratePredictions(
            train_conf['extra_generation_steps'],
            train_conf.get('external_targets'),
            config['net']['criterion'].get('trpo_coef', 0.0),
            train_conf.get('force_generate_groundtruth'),
            train_conf.get('catching_up_coof'),
            train_conf.get('catching_up_freq')))
    if clipping:
        extensions.append(AdaptiveClipping(
            algorithm.total_gradient_norm,
            clipping[0], train_conf['gradient_threshold'],
            decay_rate=0.998, burnin_period=500))
    extensions += [
        FinishAfter(after_n_batches=train_conf.get('num_batches'),
                    after_n_epochs=train_conf.get('num_epochs'))
            .add_condition(["after_batch"], _gradient_norm_is_none),
    ]

    if bokeh:
        channels = [
            # Plot 1: training and validation costs
            [average_monitoring.record_name(train_cost),
             validation.record_name(cost)],
            # Plot 2: gradient norm,
            [average_monitoring.record_name(algorithm.total_gradient_norm),
             average_monitoring.record_name(clipping[0].threshold)]]
            # Plot 3: phoneme error rate
        if search_config:
            channels += [per_monitoring.record_name(per)]
        channels += [
            # Plot 4: training and validation mean weight entropy
            [average_monitoring._record_name('weights_entropy_per_label'),
             validation._record_name('weights_entropy_per_label')],
            # Plot 5: training and validation monotonicity penalty
            [average_monitoring._record_name('weights_penalty_per_recording'),
             validation._record_name('weights_penalty_per_recording')]]
        extensions += [
            Plot(bokeh_name if bokeh_name
                 else os.path.basename(save_path),
                 channels,
                 every_n_batches=10,
                 server_url=bokeh_server),]
    checkpoint = Checkpoint(
        save_path,
        before_first_epoch=not fast_start,
        every_n_epochs=train_conf.get('save_every_epochs'),
        every_n_batches=train_conf.get('save_every_batches'),
        save_main_loop=True,
        save_separately=["log"],
        use_cpickle=True)
    if search_requested:
        checkpoint.add_condition(
                ['after_batch'],
                OnLogRecord(track_the_best_per.notification_name),
                (root_path + "_best" + extension,))
    if validation_requested:
        checkpoint.add_condition(
            ['after_batch'],
            OnLogRecord(track_the_best_cost.notification_name),
            (root_path + "_best_ll" + extension,)),
    extensions += [
        checkpoint,
        EmbedIPython(use_main_loop_run_caller_env=True)]

    if train_conf.get('patience'):
        patience_conf = train_conf['patience']
        if not patience_conf.get('notification_names'):
            # setdefault will not work for empty list
            patience_conf['notification_names'] = [
                track_the_best_per.notification_name,
                track_the_best_cost.notification_name]
        extensions.append(Patience(**patience_conf))

    extensions.append(Printing(every_n_batches=mon_conf['primary_freq']))

    return model, algorithm, data, extensions


def train(config, save_path, bokeh_name,
          params, bokeh_server, bokeh, test_tag, use_load_ext,
          load_log, fast_start, debug_mode):

    model, algorithm, data, extensions = initialize_all(
        config, save_path, bokeh_name,
        params, bokeh_server, bokeh, test_tag, use_load_ext,
        load_log, fast_start)

    num_examples = config['training'].get('num_examples', None)

    # Save the config into the status
    log = TrainingLog()
    log.status['_config'] = repr(config)
    if debug_mode:
        data_stream = data.get_stream(
            "train", shuffle=False, num_examples=data.batch_size)
    else:
        data_stream = data.get_stream(
            "train", shuffle=config['training'].get('shuffle', True),
            num_examples=num_examples)
    main_loop = MainLoop(
        model=model, log=log, algorithm=algorithm,
        data_stream=data_stream,
        extensions=extensions)
    main_loop.run()

    if (main_loop.log.status['batch_interrupt_received']
            or main_loop.log.status['epoch_interrupt_received']):
        return 'interrupted'
    return 'success'


def train_multistage(config, save_path, bokeh_name, params,
                     start_stage, final_stage, **kwargs):
    """Run multiple stages of the training procedure."""
    if os.environ.get('SLURM_RESTART_COUNT') is not None:
        logger.debug('This is a SLURM restart')
        params = None
        start_stage = None

    if not config.multi_stage:
        main_save_path = os.path.join(save_path, 'main.tar')
        if os.path.exists(main_save_path):
            logger.info("Training will be resumed")
            params = main_save_path
            kwargs['use_load_ext'] = True
        train(config, main_save_path, bokeh_name, params, **kwargs)
        return

    stages = list(config.ordered_stages.items())
    current_stage_path = save_path + '/current_stage.txt'

    # Prepare the start stage
    if start_stage:
        start_stage = config.stage_number(start_stage)
    elif os.path.exists(current_stage_path):
        # If start stage has not be provided explicitly, assume that
        # the current stage has to be continued
        with open(current_stage_path) as file_:
            start_stage_name = file_.read().strip()
            start_stage = config.stage_number(start_stage_name)
        logger.info("Training is resumed from stage " + start_stage_name)
        # To continue the current stage we tell the training routine
        # to use the log, the parameters and etc. from the old main loop dump
        kwargs['use_load_ext'] = True
        params = '{}/{}.tar'.format(save_path, start_stage_name)
    else:
        start_stage = 0

    if final_stage is not None:
        final_stage = config.stage_number(final_stage)
    else:
        final_stage = len(stages) - 1

    # Run all stages
    for number in range(start_stage, final_stage + 1):
        stage_name, stage_config = stages[number]
        logger.info("Stage \"{}\" config:\n".format(stage_name)
                        + pprint.pformat(stage_config, width=120))
        stage_save_path = '{}/{}.tar'.format(save_path, stage_name)
        stage_bokeh_name = '{}_{}'.format(save_path, stage_name)
        if params:
            stage_params = params
            # Avoid loading the params twice
            params = None
        elif number > 0:
            stage_params = '{}/{}{}.tar'.format(
                save_path, stages[number - 1][0],
                stage_config['training'].get('restart_from', ''))
        else:
            stage_params = None

        with open(current_stage_path, 'w') as dst:
            print(stage_name, file=dst)
        exit_code = train(
            stage_config, stage_save_path, stage_bokeh_name, stage_params, **kwargs)
        if exit_code != 'success':
            return

        # Using load only makes sense at the first stage of the stage loop.
        kwargs['use_load_ext'] = False


def search(config, params, load_path, part, decode_only, report,
           decoded_save, nll_only, seed):
    from matplotlib import pyplot
    from lvsr.notebook import show_alignment

    data = Data(**config['data'])
    search_conf = config['monitoring']['search']

    logger.info("Recognizer initialization started")
    recognizer = create_model(config, data, load_path)
    recognizer.init_beam_search(search_conf['beam_size'])
    logger.info("Recognizer is initialized")

    has_uttids = 'uttids' in data.info_dataset.provides_sources
    add_sources = ('uttids',) if has_uttids else ()
    dataset = data.get_dataset(part, add_sources)
    stream = data.get_stream(
        part, batches=False,
        shuffle=
            config['training']['shuffle'] if part == 'train' else False,
        add_sources=add_sources,
        num_examples=
            config['monitoring']['search_on_training'] if part == 'train' else None,
        seed=seed)
    it = stream.get_epoch_iterator(as_dict=True)
    if decode_only is not None:
        decode_only = eval(decode_only)

    weights = tensor.matrix('weights')
    weight_statistics = theano.function(
        [weights],
        [weights_std(weights.dimshuffle(0, 'x', 1)),
            monotonicity_penalty(weights.dimshuffle(0, 'x', 1))])

    print_to = sys.stdout
    if report:
        alignments_path = os.path.join(report, "alignments")
        if not os.path.exists(report):
            os.mkdir(report)
            os.mkdir(alignments_path)
        print_to = open(os.path.join(report, "report.txt"), 'w')
    if decoded_save:
        print_to = open(decoded_save, 'w')

    num_examples = .0
    total_nll = .0
    total_errors = .0
    total_length = .0
    total_wer_errors = .0
    total_word_length = 0.

    if config.get('vocabulary'):
        with open(os.path.expandvars(config['vocabulary'])) as f:
            vocabulary = dict(line.split() for line in f.readlines())

        def to_words(chars):
            words = chars.split()
            words = [vocabulary[word] if word in vocabulary
                     else vocabulary['<UNK>'] for word in words]
            return words

    for number, example in enumerate(it):
        if decode_only and number not in decode_only:
            continue
        uttids = example.pop('uttids', None)
        raw_groundtruth = example.pop('labels')
        required_inputs = dict_subset(example, recognizer.inputs.keys())

        print("Utterance {} ({})".format(number, uttids), file=print_to)

        groundtruth = dataset.decode(raw_groundtruth)
        groundtruth_text = dataset.pretty_print(raw_groundtruth, example)
        costs_groundtruth, weights_groundtruth = recognizer.analyze(
            inputs=required_inputs,
            groundtruth=raw_groundtruth,
            prediction=raw_groundtruth)[:2]
        weight_std_groundtruth, mono_penalty_groundtruth = weight_statistics(
            weights_groundtruth)
        total_nll += costs_groundtruth.sum()
        num_examples += 1
        print("Groundtruth:", groundtruth_text, file=print_to)
        print("Groundtruth cost:", costs_groundtruth.sum(), file=print_to)
        print("Groundtruth weight std:", weight_std_groundtruth, file=print_to)
        print("Groundtruth monotonicity penalty:", mono_penalty_groundtruth,
              file=print_to)
        print("Average groundtruth cost: {}".format(total_nll / num_examples),
              file=print_to)
        if nll_only:
            print_to.flush()
            continue

        before = time.time()
        search_kwargs = dict(
            char_discount=search_conf.get('char_discount'),
            round_to_inf=search_conf.get('round_to_inf'),
            stop_on=search_conf.get('stop_on'),
            validate_solution_function=getattr(
                data.info_dataset, 'validate_solution', None),
            consider_all_eos=search_conf.get('consider_all_eos'))
        search_kwargs = {k: v for k, v in search_kwargs.items() if v}
        outputs, search_costs = recognizer.beam_search(
            required_inputs, **search_kwargs)

        took = time.time() - before
        recognized = dataset.decode(outputs[0])
        recognized_text = dataset.pretty_print(outputs[0], example)
        if recognized:
            # Theano scan doesn't work with 0 length sequences
            costs_recognized, weights_recognized = recognizer.analyze(
                inputs=required_inputs,
                groundtruth=raw_groundtruth,
                prediction=outputs[0])[:2]
            weight_std_recognized, mono_penalty_recognized = weight_statistics(
                weights_recognized)
            error = min(1, wer(groundtruth, recognized))
        else:
            error = 1
        total_errors += len(groundtruth) * error
        total_length += len(groundtruth)

        if config.get('vocabulary'):
            wer_error = min(1, wer(to_words(groundtruth_text),
                                   to_words(recognized_text)))
            total_wer_errors += len(groundtruth) * wer_error
            total_word_length += len(groundtruth)

        if report and recognized:
            show_alignment(weights_groundtruth, groundtruth, bos_symbol=True)
            pyplot.savefig(os.path.join(
                alignments_path, "{}.groundtruth.png".format(number)))
            show_alignment(weights_recognized, recognized, bos_symbol=True)
            pyplot.savefig(os.path.join(
                alignments_path, "{}.recognized.png".format(number)))


        print("Decoding took:", took, file=print_to)
        print("Beam search cost:", search_costs[0], file=print_to)
        print("Recognized:", recognized_text, file=print_to)
        if recognized:
            print("Recognized cost:", costs_recognized.sum(), file=print_to)
            print("Recognized weight std:", weight_std_recognized,
                  file=print_to)
            print("Recognized monotonicity penalty:", mono_penalty_recognized,
                  file=print_to)
        print("CER:", error, file=print_to)
        print("Average CER:", total_errors / total_length, file=print_to)
        if config.get('vocabulary'):
            print("WER:", wer_error, file=print_to)
            print("Average WER:", total_wer_errors / total_word_length, file=print_to)
        print_to.flush()

        #assert_allclose(search_costs[0], costs_recognized.sum(), rtol=1e-5)


def sample(config, params, load_path, part):
    data = Data(**config['data'])
    recognizer = create_model(config, data, load_path)

    stream = data.get_stream(part, batches=False, shuffle=False)
    it = stream.get_epoch_iterator(as_dict=True)

    print_to = sys.stdout
    for number, example in enumerate(it):
        uttids = example.pop('uttids', None)
        print("Utterance {} ({})".format(number, uttids),
              file=print_to)
        raw_groundtruth = example.pop('labels')
        groundtruth_text = data.pretty_print(raw_groundtruth, example)
        print("Groundtruth:", groundtruth_text, file=print_to)
        sample = recognizer.sample(example)
        recognized_text = data.pretty_print(sample, example)
        print("Recognized:", recognized_text, file=print_to)


def show_config(config):
    def _normalize(conf):
        if isinstance(conf, (int, str, float)):
            return conf
        if isinstance(conf, dict):
            result = {}
            for key, value in conf.items():
                normalized = _normalize(value)
                if normalized is not None:
                    result[key] = normalized
                else:
                    result[key] = str(value)
            return result
        if isinstance(conf, list):
            result = []
            for value in conf:
                normalized = _normalize(value)
                if normalized is not None:
                    result.append(normalized)
                else:
                    result.append(str(value))
            return result
        return None
    yaml.dump(_normalize(config), sys.stdout,
              default_flow_style=False)


def show_data(config):
    data = Data(**config['data'])
    stream = data.get_stream("train")
    batch = next(stream.get_epoch_iterator(as_dict=True))
    import IPython; IPython.embed()


def test(config, **kwargs):
    raise NotImplementedError()
