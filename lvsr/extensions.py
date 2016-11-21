"""Nice small extensions that maybe will it make to Blocks at some point."""
from __future__ import print_function
import subprocess
import pkgutil
import math
import logging
import copy
import re

import theano
from theano.gof.graph import io_toposort
from theano.scan_module.scan_op import Scan

from blocks.extensions import TrainingExtension, SimpleExtension,\
    FinishAfter
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.utils import shared_floatx_zeros
from blocks.select import Selector
from blocks.utils import shared_like

logger = logging.getLogger(__name__)


class CGStatistics(SimpleExtension):

    def __init__(self, **kwargs):
        kwargs.setdefault('before_first_epoch', True)
        kwargs.setdefault('on_resumption', True)
        super(CGStatistics, self).__init__(**kwargs)

    def do(self, *args, **kwargs):
        logger.info("Computation graph statistics:")
        cost_cg = ComputationGraph(self.main_loop.algorithm.cost)
        updates_cg = ComputationGraph(
            [u[1] for u in self.main_loop.algorithm.updates
             if isinstance(u[1], theano.Variable)])
        cost_nodes = io_toposort(cost_cg.inputs, cost_cg.outputs)
        updates_nodes = io_toposort(updates_cg.inputs, updates_cg.outputs)

        cost_scan_nodes = [
            node for node in cost_nodes
            if isinstance(node.op, Scan)]
        updates_scan_nodes = [
            node for node in updates_nodes
            if isinstance(node.op, Scan)]
        final_scan_nodes = [
            node for node in self.main_loop.algorithm._function.maker.fgraph.apply_nodes
            if isinstance(node.op, Scan)]

        logger.info("SCAN NODES IN THE COST GRAPH:")
        for n in cost_scan_nodes:
            logger.info(n.op.name)
        logger.info("SCAN NODES IN THE UPDATES GRAPH:")
        for n in updates_scan_nodes:
            logger.info(n.op.name)
        logger.info("SCAN NODES IN THE FINAL GRAPH:")
        for n in final_scan_nodes:
            logger.info(n.op.name)


class CodeVersion(SimpleExtension):

    def __init__(self, packages, **kwargs):
        self.packages = packages
        kwargs.setdefault('before_training', True)
        super(CodeVersion, self).__init__(**kwargs)

    def do(self, *args, **kwargs):
        package_paths = {name: loader.path
                         for loader, name, _ in pkgutil.iter_modules()
                         # skipping .eggs imported with zipimporter
                         if isinstance(loader, pkgutil.ImpImporter)}
        for package in self.packages:
            path = package_paths[package]
            last_commit_record = "_{}_last_commit".format(package)
            git_diff_record = "_{}_git_diff".format(package)
            self.main_loop.log.status[last_commit_record] = (
                subprocess.check_output("git --no-pager log -1",
                                        cwd=path, shell=True))
            self.main_loop.log.status[git_diff_record] = (
                subprocess.check_output("git diff",
                                        cwd=path, shell=True))


class IPDB(SimpleExtension):

    def do(self, *args, **kwargs):
        import ipdb; ipdb.set_trace()


class AdaptiveClipping(TrainingExtension):

    def __init__(self, gradient_norm, clipping_rule,
                 initial_threshold, burnin_period=100, decay_rate=0.99):
        self.gradient_norm = gradient_norm
        self.clipping_rule = clipping_rule
        self.initial_threshold = initial_threshold
        self.burnin_period = burnin_period
        self.decay_rate = decay_rate

        self.mean_gradient_norm = self.mean_gradient_norm2 = .0
        self._gradient_norm_receiver = shared_like(self.gradient_norm)

    def before_training(self):
        self.main_loop.algorithm.add_updates([
            (self._gradient_norm_receiver, self.gradient_norm)])

    def after_batch(self, batch):
        gradient_norm = math.log(self._gradient_norm_receiver.get_value())
        self.mean_gradient_norm = (self.decay_rate * self.mean_gradient_norm
                                   + (1 - self.decay_rate) * gradient_norm)
        self.mean_gradient_norm2 = (self.decay_rate * self.mean_gradient_norm2
                                    + (1 - self.decay_rate) * gradient_norm ** 2)
        self.std_gradient_norm = (
            (self.mean_gradient_norm2 - self.mean_gradient_norm ** 2) ** .5)
        threshold = math.exp(self.mean_gradient_norm + 1 * self.std_gradient_norm)
        confidence = (min(
            self.burnin_period, self.main_loop.status['iterations_done']) /
            float(self.burnin_period))
        threshold = (confidence * threshold +
                     (1 - confidence) * self.initial_threshold)
        threshold = min(threshold, 5 * self.initial_threshold)
        self.clipping_rule.threshold.set_value(threshold)


class GeneratePredictions(SimpleExtension):

    def __init__(self, extra_generation_steps,
                 compute_targets, compute_policy,
                 force_generate_groundtruth,
                 catching_up_coof, catching_up_freq,
                **kwargs):
        self.extra_generation_steps = extra_generation_steps
        self.compute_targets = compute_targets
        self.compute_policy = compute_policy
        self.force_generate_groundtruth = force_generate_groundtruth
        self.catching_up_coof = catching_up_coof
        self.catching_up_freq = catching_up_freq
        kwargs.setdefault('before_training', True)
        kwargs.setdefault('before_batch', True)
        kwargs.setdefault('after_batch', True)
        super(GeneratePredictions, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        if which_callback == 'before_training':
            logger.info("Compiling prediction generator...")
            recognizer, = self.main_loop.model.get_top_bricks()
            self.trained_recognizer = recognizer
            self.recognizer = copy.deepcopy(recognizer)

            # A bit of defensive programming, because why not :)
            assert self.recognizer.generator.readout.compute_targets
            assert self.recognizer.generator.readout.compute_policy
            assert self.recognizer.generator.readout.solve_bellman
            assert self.recognizer.generator.readout.epsilon == 0.0

            groundtruth = self.recognizer.labels
            groundtruth_mask = self.recognizer.labels_mask
            generated = self.recognizer.get_generate_graph(
                n_steps=self.recognizer.labels.shape[0]
                    + self.extra_generation_steps,
                return_initial_states=True,
                use_softmax_t=True)
            generation_method = self.recognizer.generator.generate

            if not self.force_generate_groundtruth:
                prediction = generated.pop('samples')
                prediction_mask = self.recognizer.mask_for_prediction(prediction)
            else:
                prediction = groundtruth.copy()
                prediction_mask = groundtruth_mask.copy()
            prediction.name = 'predicted_labels'
            prediction_mask.name = 'predicted_mask'

            cg = ComputationGraph(generated.values())
            attended, = VariableFilter(
                applications=[generation_method], name='attended')(cg)
            attended_mask, = VariableFilter(
                applications=[generation_method], name='attended_mask')(cg)
            generated = {key: value[:-1] for key, value in generated.items()}
            costs = self.recognizer.generator.readout.costs(
                prediction=prediction, prediction_mask=prediction_mask,
                groundtruth=groundtruth, groundtruth_mask=groundtruth_mask,
                attended=attended, attended_mask=attended_mask,
                **generated)
            cost_cg = ComputationGraph(costs)
            value_targets, = VariableFilter(name='value_targets')(cost_cg)
            value_targets.name = 'value_targets'
            probs, = VariableFilter(name='probs')(cost_cg)
            probs.name = 'probs'
            rewards, = VariableFilter(name='rewards')(cost_cg)

            variables_to_compute = [prediction, prediction_mask]
            if self.compute_targets:
                logger.debug("Also compute the targets")
                variables_to_compute += [value_targets]
            if self.compute_policy:
                variables_to_compute += [probs]
            self.extended_cg = ComputationGraph(variables_to_compute)
            self._generate = self.extended_cg.get_theano_function()
            logger.info("Prediction generator compiled")

            params = Selector(self.recognizer).get_parameters()
            trained_params = Selector(self.trained_recognizer).get_parameters()
            if self.catching_up_freq:
                def get_coof(name):
                    if isinstance(self.catching_up_coof, float):
                        return self.catching_up_coof
                    elif isinstance(self.catching_up_coof, list):
                        result = None
                        for pattern, coof in self.catching_up_coof:
                            if re.match(pattern, name):
                                result = coof
                        return result
                    else:
                        raise ValueError
                updates = []
                for name in params:
                    coof = get_coof(name)
                    logging.debug("Catching up coefficient for {} is {}".format(
                        name, coof))
                    updates.append((params[name], params[name] * (1 - coof)
                                                  + trained_params[name] * coof))
                # This is needed when parameters are shared between brick
                # and occur more than once in the list of updates.
                updates = dict(updates).items()
                self._catch_up = theano.function([], [], updates=updates)
        elif which_callback == 'before_batch':
            batch, = args
            generated = self._generate(
                *[batch[variable.name] for variable in self.extended_cg.inputs])
            for variable, value in zip(self.extended_cg.outputs, generated):
                batch[variable.name] = value
        elif which_callback == 'after_batch':
            if (self.catching_up_freq
                    and self.main_loop.status['iterations_done'] % self.catching_up_freq == 0):
                self._catch_up()
        else:
            raise ValueError("can't be called on " + which_callback)



class Patience(FinishAfter):
    """Stop after improvements have ceased for a given period.

    Parameters
    ----------
    notification_names : list of str
        The names of the log record to look for which indicate that a new
        best performer has been found.  Note that the value of this
        record is not inspected.
    min_iterations : int, optional
        The minimum number of iterations to perform. Exactly one of
        `iterations` or `epochs` must be not `None` (default).
    min_epochs : int, optional
        The minimum number of epochs to perform. Exactly one of
        `iterations` or `epochs` must be not `None` (default).
    patience_factor : float
        The factor by which to expand the number of iterations to do after
        after an improvement.
    patience_log_record : str, optional
        The name under which to record the number of iterations we
        are currently willing to wait for a new best performer.
        Defaults to `notification_name + '_patience_epochs'` or
        `notification_name + '_patience_iterations'`, depending
        which measure is being used.

    Notes
    -----
    By default, runs after each epoch. This can be manipulated via
    keyword arguments (see :class:`blocks.extensions.SimpleExtension`).

    """
    def __init__(self, notification_names, min_iterations=None,
                 min_epochs=None, patience_factor=1.5,
                 patience_log_record=None, **kwargs):
        if (min_epochs is None) == (min_iterations is None):
            raise ValueError("Need exactly one of epochs or iterations "
                             "to be specified")
        self.notification_names = notification_names
        self.min_iterations = min_iterations
        self.min_epochs = min_epochs
        self.patience_factor = patience_factor
        kwargs.setdefault('after_epoch', True)
        kwargs.setdefault('before_first_epoch', True)
        self.last_best_iter = self.last_best_epoch = 0
        if patience_log_record is None:
            self.patience_log_record = ('patience' +
                                        ('_epochs' if self.min_epochs is not None
                                         else '_iterations'))
        else:
            self.patience_log_record = patience_log_record
        super(Patience, self).__init__(**kwargs)

    def update_best(self):
        # Here mainly so we can easily subclass different criteria.
        matched = False
        for not_name in self.notification_names:
            if not_name in self.main_loop.log.current_row:
                matched = True
                break

        if matched:
            self.last_best_iter = self.main_loop.log.status['iterations_done']
            self.last_best_epoch = self.main_loop.log.status['epochs_done']

    def do(self, which_callback, *args):
        self.update_best()
        if self.min_epochs is not None:
            to_do = max(self.min_epochs,
                        int(self.patience_factor * self.last_best_epoch+0.5))
            self.main_loop.log.status[self.patience_log_record] = to_do
            if to_do <= self.main_loop.log.status['epochs_done']:
                super(Patience, self).do(which_callback, *args)
        else:
            to_do = max(self.min_iterations,
                        int(self.patience_factor * self.last_best_iter+0.5))
            self.main_loop.log.status[self.patience_log_record] = to_do
            if to_do <= self.main_loop.log.status['iterations_done']:
                super(Patience, self).do(which_callback, *args)
