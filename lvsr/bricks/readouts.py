import logging
import theano
from theano.gradient import disconnected_grad
from theano import tensor

from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.bricks import Linear, NDimensionalSoftmax
from blocks.bricks.base import application
from blocks.roles import OUTPUT, add_role, WEIGHT
from blocks.utils import dict_subset, shared_floatx_nans
from blocks_extras.bricks.sequence_generator2 import SoftmaxReadout, MergeReadout

logger = logging.getLogger(__name__)


class ReinforceReadout(SoftmaxReadout):

    def __init__(self, reward_brick, entropy=None, **kwargs):
        super(ReinforceReadout, self).__init__(**kwargs)
        self.reward_brick = reward_brick
        self.entropy_coof = entropy

        self.value_prediction = Linear(output_dim=1, name='value_prediction')

        self.children += [
            reward_brick, self.value_prediction]

        self.costs.inputs += ['attended', 'attended_mask']

    def _push_allocation_config(self):
        super(ReinforceReadout, self)._push_allocation_config()
        self.value_prediction.input_dim = self.get_dim('states')

    @application
    def costs(self, application_call, prediction, prediction_mask,
              groundtruth, groundtruth_mask,
              **inputs):
        states = disconnected_grad(inputs['states'])

        merged = self.merge(**dict_subset(inputs, self.merge_names))
        # Compute log-probabilities for the predicted tokens
        log_probs = -self.all_scores(prediction, merged) * prediction_mask
        # Compute per-token rewards
        rewards = self.reward_brick.apply(prediction, prediction_mask,
                                          groundtruth, groundtruth_mask).sum(axis=-1)
        # Encourage entropy by adding negated log-probs to the rewards
        application_call.add_auxiliary_variable(log_probs, name='log_probs')
        if self.entropy_coof:
            rewards += self.entropy_coof * disconnected_grad(-log_probs)

        future_rewards = rewards[::-1].cumsum(axis=0)[::-1]

        baselines = self.value_prediction.apply(states)[:, :, 0]
        application_call.add_auxiliary_variable(
            baselines, name='baselines')
        # Compute baseline error
        centered_future_rewards = future_rewards - baselines
        baseline_errors = (
            (centered_future_rewards *
            disconnected_grad(prediction_mask)) ** 2).sum(axis=0)
        application_call.add_auxiliary_variable(
            baseline_errors, name='baseline_errors')

        # The gradient of this will be the REINFORCE 1-sample
        # gradient estimate
        costs = (disconnected_grad(centered_future_rewards)
                 * log_probs
                 * prediction_mask).sum(axis=0)

        # Add auxiliary variables for intermediate steps of the computation
        application_call.add_auxiliary_variable(
            rewards, name='rewards')
        application_call.add_auxiliary_variable(
            log_probs.copy(), name='prediction_log_probs')

        return costs


class CriticReadout(MergeReadout):

    def __init__(self, num_tokens,
                 value_softmax=False, same_value_for_wrong=False,
                 groundtruth_word_bonus=False, dueling_outputs=False, **kwargs):
        self.value_softmax = value_softmax
        self.same_value_for_wrong = same_value_for_wrong
        self.groundtruth_word_bonus = groundtruth_word_bonus
        self.dueling_outputs = dueling_outputs
        super(CriticReadout, self).__init__(post_merge_dim=num_tokens, **kwargs)
        self.costs.inputs = ([
            'prediction', 'prediction_mask',
            'groundtruth', 'groundtruth_mask']
            + self.input_names)

    def _allocate(self):
        w = shared_floatx_nans((self.get_dim('states'),), name='add_weights')
        add_role(w, WEIGHT)
        self.parameters.append(w)

    def _initialize(self):
        self.weights_init.initialize(self.parameters[0], self.rng)

    # For compatibility with Blocks-extras
    def sample(self):
        raise NotImplementedError()

    # For compatibility with Blocks-extras
    def scores(self):
        pass

    @application
    def costs(self, prediction, prediction_mask,
              groundtruth, groundtruth_mask, **inputs):
        outputs = self.all_outputs(groundtruth, groundtruth_mask, **inputs)
        # It does not matter what we return here, as long as it contains
        # the values in the computation graph.
        return outputs.sum()

    @application
    def all_outputs(self, application_call, groundtruth, groundtruth_mask, **inputs):
        outputs = self.merge(**dict_subset(inputs, self.merge_names))
        indices = tensor.repeat(
            tensor.arange(groundtruth.shape[1]), groundtruth.shape[0])
        if self.value_softmax:
            logger.debug('Applying value softmax')
            outputs = (tensor.addbroadcast(outputs[:, :, :1], 2)
                       + self.softmax.apply(outputs[:, :, 1:], extra_ndim=1))
        if self.same_value_for_wrong:
            logger.debug('Same value for apriori wrong actions')
            wrong_output = outputs[:, :, 0]
            outputs = outputs[:, :, 1:]
            wrong_mask = tensor.ones_like(outputs[0])
            wrong_mask = tensor.set_subtensor(
                wrong_mask[indices, groundtruth.T.flatten()], 0)
            outputs = (outputs * (1 - wrong_mask)
                        + wrong_output[:, :, None] * wrong_mask)
            application_call.add_auxiliary_variable(wrong_mask, name='wrong_mask')
        if self.groundtruth_word_bonus:
            logger.debug('Bonus for grondtruth words')
            wrong_mask = tensor.ones_like(outputs[0])
            wrong_mask = tensor.set_subtensor(
                wrong_mask[indices, groundtruth.T.flatten()], 0)
            w, = self.parameters
            bonuses = inputs['states'].dot(w)
            outputs += bonuses[:, :, None] * (1 - wrong_mask)[None, :, :]
        if self.dueling_outputs:
            logger.debug('Dueling outputs a-la dueling networks')
            base_output = outputs[:, :, [0]]
            dueling_outputs = outputs[:, :, 1:]
            outputs = base_output + dueling_outputs - dueling_outputs.mean(axis=2, keepdims=True)
        return outputs

    @application
    def outputs(self, groundtruth, groundtruth_mask, **inputs):
        # Copy-pasted from all_outputs, because Theano does not support ellipsis
        outputs = self.merge(**dict_subset(inputs, self.merge_names))
        indices = tensor.repeat(
            tensor.arange(groundtruth.shape[1]), groundtruth.shape[0])
        if self.value_softmax:
            logger.debug('Applying value softmax')
            outputs = (tensor.addbroadcast(outputs[:, :1], 1)
                       + self.softmax.apply(outputs[:, 1:]))
        if self.same_value_for_wrong:
            logger.debug('Same value for apriori wrong actions')
            wrong_output = outputs[:, 0]
            outputs = outputs[:, 1:]
            wrong_mask = tensor.ones_like(outputs)
            wrong_mask = tensor.set_subtensor(
                wrong_mask[indices, groundtruth.T.flatten()], 0)
            outputs = (outputs * (1 - wrong_mask)
                        + wrong_output[:, None] * wrong_mask)
        if self.groundtruth_word_bonus:
            logger.debug('Bonus for grondtruth words')
            wrong_mask = tensor.ones_like(outputs)
            wrong_mask = tensor.set_subtensor(
                wrong_mask[indices, groundtruth.T.flatten()], 0)
            w, = self.parameters
            bonuses = inputs['states'].dot(w)
            outputs = outputs + bonuses[:, None] * (1 - wrong_mask)
        if self.dueling_outputs:
            logger.debug('Dueling outputs a-la dueling networks')
            base_output = outputs[:, [0]]
            dueling_outputs = outputs[:, 1:]
            outputs = base_output + dueling_outputs - dueling_outputs.mean(axis=1, keepdims=True)
        return outputs


class ActorCriticReadout(SoftmaxReadout):
    """Actor-critic

    Params
    ------
    bos_token : int
        The token used to pad critic input. Critic needs to do
        at least one extra step compared to the actor in order
        to get the first glimpse of the ground-truth sequence
        before predicting the actual values.

    """
    def __init__(self, reward_brick,
                compute_targets, solve_bellman,
                freeze_actor, freeze_critic, critic_uses_actor_states,
                critic_uses_groundtruth,
                critic=None, critic_burnin_steps=None,
                critic_policy_t=None,
                entropy_reward_coof=None, cross_entropy_reward_coof=None,
                trpo_coef=None,
                discount=None,
                value_penalty=None, value_penalty_type=None,
                accumulate_outputs=False, use_value_biases=None,
                actor_grad_estimate=None,
                bos_token=None,
                **kwargs):
        super(ActorCriticReadout, self).__init__(**kwargs)
        self.reward_brick = reward_brick
        self.critic = critic
        self.freeze_actor = freeze_actor
        self.freeze_critic = freeze_critic
        self.critic_uses_actor_states = critic_uses_actor_states
        self.critic_uses_groundtruth = (
            critic_uses_groundtruth if critic_uses_groundtruth is not None else True)
        self.critic_burnin_steps = (
            critic_burnin_steps if critic_burnin_steps is not None else 0)
        self.value_summand = Linear(output_dim=1, name='summand')
        self.softmax_t = 1.
        self.critic_policy_t = (
            critic_policy_t if critic_policy_t is not None else 1.0)
        self.epsilon = 0.
        self.discount = (
            discount if discount is not None else 1.)
        self.entropy_reward_coof = (
            entropy_reward_coof if entropy_reward_coof is not None else 0.)
        self.cross_entropy_reward_coof = (
            cross_entropy_reward_coof if cross_entropy_reward_coof is not None else 0.)
        self.trpo_coef = (
            trpo_coef if trpo_coef is not None else 0.)
        self.value_penalty = value_penalty
        self.value_penalty_type = (
            value_penalty_type if value_penalty_type is not None else "L2")
        self.compute_targets = compute_targets
        self.solve_bellman = solve_bellman
        self.accumulate_outputs = accumulate_outputs
        self.use_value_biases = (
            use_value_biases if use_value_biases is not None else True)
        self.actor_grad_estimate = (
            actor_grad_estimate if actor_grad_estimate else 'all_actions')
        self.bos_token = bos_token
        self.softmax = NDimensionalSoftmax()
        self.children += [reward_brick, self.value_summand, self.softmax]
        if self.critic:
            self.children.append(self.critic)
        self.costs.inputs += ['attended', 'attended_mask']

    def _push_allocation_config(self):
        super(ActorCriticReadout, self)._push_allocation_config()
        self.value_summand.input_dim = self.get_dim('attended')

    @application
    def scores(self, **inputs):
        merged = self.merge(**dict_subset(inputs, self.merge_names))
        return self.softmax.log_probabilities(
            merged * self.softmax_t, extra_ndim=merged.ndim - 2)

    @application
    def costs(self, application_call, prediction, prediction_mask,
              groundtruth, groundtruth_mask,
              **inputs):
        def _prediction_subtensor(data):
            if data.ndim != 3:
                raise ValueError
            flat_data = data.reshape((
                data.shape[0] * data.shape[1],
                data.shape[2]))
            flat_data = flat_data[
                    tensor.arange(flat_data.shape[0]), prediction.flatten()]
            return flat_data.reshape((
                prediction.shape[0], prediction.shape[1]))

        attended = disconnected_grad(inputs.pop('attended'))
        attended_mask = disconnected_grad(inputs.pop('attended_mask'))

        # Compute the rewards
        rewards = self.reward_brick.apply(
            prediction, prediction_mask,
            groundtruth, groundtruth_mask)[:, :, 0]
        future_rewards = rewards[::-1].cumsum(axis=0)[::-1]

        # Compute the critic outputs
        if self.critic:
            padding = tensor.repeat(
                tensor.fill(prediction[0:1], self.bos_token), 1, axis=0)
            mask_padding = tensor.repeat(
                tensor.fill(prediction_mask[0:1], 1.), 1, axis=0)
            padded_prediction = tensor.concatenate([padding, prediction])
            padded_prediction_mask = tensor.concatenate([mask_padding, prediction_mask])
            if self.critic_uses_groundtruth:
                critic_context = groundtruth
                critic_context_mask = groundtruth_mask
            else:
                critic_context = tensor.zeros_like(groundtruth[0:1])
                critic_context_mask = tensor.zeros_like(groundtruth_mask[0:1])
            critic_kwargs = dict(
                prediction=padded_prediction, prediction_mask=padded_prediction_mask,
                groundtruth=critic_context, groundtruth_mask=critic_context_mask,
                inputs=critic_context, inputs_mask=critic_context_mask)

            if self.critic_uses_actor_states:
                extra_inputs = disconnected_grad(inputs['states'])
                # We don't need the very last hidden state of the actor
                # in extra_inputs. We have to add something instead for the shapes
                # to match. It doesn't matter at all, what exactly we add.
                critic_kwargs['extra_inputs'] = tensor.concatenate(
                    [extra_inputs, tensor.zeros_like(extra_inputs[0:1])])
            critic_cg = ComputationGraph(self.critic.costs(**critic_kwargs))
            outputs, = VariableFilter(
                applications=[self.critic.generator.readout.all_outputs],
                roles=[OUTPUT])(critic_cg)
            # The first subtensor should be discarded, because it was outputted
            # for the padding. In addition to that Q-values from the first
            # 'critic_burnin_steps' will be ignored, see later in the code.
            outputs = outputs[1:]
        else:
            outputs = self.merge(**dict_subset(inputs, self.merge_names))
        prediction_outputs = _prediction_subtensor(outputs)

        # Compute Q adjustments
        adjustments = outputs
        prediction_adjustments = prediction_outputs
        if self.accumulate_outputs:
            prediction_adjustments = prediction_outputs.cumsum(axis=0)
            adjustments = tensor.inc_subtensor(
                adjustments[1:],  prediction_adjustments[:-1][:, :, None])

        # Compute shared additive biases for all Q values
        if self.use_value_biases:
            value_biases = (
                self.value_summand.apply(attended)[:, :, 0]
                * attended_mask).sum(axis=0)
        else:
            value_biases = tensor.zeros_like(adjustments[0, :, 0])
        values = adjustments + value_biases[None, :, None]
        prediction_values = prediction_adjustments + value_biases[None, :]

        rolled_prediction_mask = tensor.roll(prediction_mask, -1, axis=0)
        rolled_prediction_mask = tensor.set_subtensor(
            rolled_prediction_mask[-1], 0)

        # Compute probabilities
        logs = self.scores(use_epsilon=False, **inputs)
        probs = tensor.exp(logs)
        if self.trpo_coef:
            logger.debug("Using TRPO coefficient of {}".format(self.trpo_coef))
            old_probs = tensor.tensor3('probs')
        else:
            old_probs = tensor.zeros_like(probs)
        prediction_logs = _prediction_subtensor(logs)

        # Compute value targets
        value_targets = (disconnected_grad(probs) * values).sum(axis=-1)
        value_targets = tensor.roll(value_targets, -1, axis=0)
        value_targets = (self.discount * value_targets * rolled_prediction_mask
                         + rewards)
        value_targets = value_targets.astype(theano.config.floatX)

        total_costs = 0

        # Compute critic cost
        if not self.compute_targets:
            logger.debug("Using given targets")
            value_targets = tensor.matrix('value_targets')
        if self.solve_bellman == 'no':
            logger.debug("Not solving Bellman, just predicting the rewards")
            value_targets = rewards.copy(name='value_targets')
        elif self.solve_bellman == 'without_dp':
            future_rewards = rewards[::-1].cumsum(axis=0)[::-1]
            logger.debug("Solving Bellman, but without DP")
            value_targets = future_rewards
        elif self.solve_bellman is not True:
            raise ValueError()
        critic_costs_per_char = ((prediction_values - value_targets) ** 2) * prediction_mask
        critic_costs = critic_costs_per_char[self.critic_burnin_steps:].sum(axis=0)
        if not self.freeze_critic:
            total_costs += critic_costs

        # Compute critic Monte-Carlo cost
        critic_monte_carlo_costs = (
            (((prediction_values - future_rewards) ** 2) * prediction_mask)
            [self.critic_burnin_steps:].sum(axis=0))

        # Value penalty
        if self.value_penalty:
            logger.debug("Use value penalty")
            if self.value_penalty_type == 'L2':
                value_deviations = (values - values.mean(axis=-1, keepdims=True)) ** 2
            elif self.value_penalty_type == 'L1':
                value_deviations = abs(values - values.mean(axis=-1, keepdims=True))
            else:
                raise ValueError("unknown value penalty type {}".format(self.value_penalty_type))
            if not self.freeze_critic:
                total_costs += (
                    self.value_penalty *
                    (value_deviations.sum(axis=-1) * prediction_mask)
                    [self.critic_burnin_steps:].sum(axis=0))

        # Compute actor cost
        if self.critic:
            # The actor cost will be minimized, that's why values
            # must be negated.
            est_name = self.actor_grad_estimate
            if est_name == 'all_actions':
                disadvantages = disconnected_grad(
                    values.max(axis=-1)[:, :, None] - values)
                actor_costs = ((probs * disadvantages).sum(axis=-1)
                               * prediction_mask)
                actor_costs = actor_costs[self.critic_burnin_steps:]
            elif est_name.startswith('1_action'):
                # Here we do not provide a target for the first step for
                # the reason we lack an estimate of the value of the initial state.
                # This is how our critic works.
                # Hopefully the network won't unlearn
                # to produce a BOS first.
                future_reward_estimate = (future_rewards
                                          if est_name.endswith('unbiased')
                                          else prediction_values)
                weights = -disconnected_grad(
                    future_reward_estimate[1:] + rewards[:-1] - prediction_values[:-1])
                actor_costs = ((prediction_logs[1:] * weights) * prediction_mask[1:])
                actor_costs = actor_costs[self.critic_burnin_steps + 1:]
            else:
                raise ValueError
            actor_costs = actor_costs.sum(axis=0)

            actor_entropies = (probs * -logs).sum(axis=-1) * prediction_mask
            actor_entropies = actor_entropies[self.critic_burnin_steps:].sum(axis=0)
            old_actor_cross_entropies = (old_probs * -logs).sum(axis=-1) * prediction_mask
            old_actor_cross_entropies = old_actor_cross_entropies[self.critic_burnin_steps:].sum(axis=0)
            critic_policy = disconnected_grad(
                self.softmax.apply(self.critic_policy_t * values, extra_ndim=1))
            critic_cross_entropies = (
                (critic_policy * -logs).sum(axis=-1)
                 * prediction_mask)
            critic_cross_entropies = critic_cross_entropies[self.critic_burnin_steps:].sum(axis=0)
            actor_costs_with_penalties = (
                actor_costs
                - self.entropy_reward_coof * actor_entropies
                # But really, should it be minus here, below?
                - self.cross_entropy_reward_coof * critic_cross_entropies
                + self.trpo_coef * old_actor_cross_entropies)
            if not self.freeze_actor:
                total_costs += actor_costs_with_penalties
            else:
                total_costs += disconnected_grad(actor_costs_with_penalties)

        # Add auxiliary variables for intermediate steps of the computation
        application_call.add_auxiliary_variable(
            rewards, name='rewards')
        application_call.add_auxiliary_variable(
            value_biases, name='value_biases')
        application_call.add_auxiliary_variable(
            values.copy(), name='values')
        application_call.add_auxiliary_variable(
            outputs.copy(), name='outputs')
        application_call.add_auxiliary_variable(
            prediction_values, name='prediction_values')
        application_call.add_auxiliary_variable(
            prediction_outputs, name='prediction_outputs')
        application_call.add_auxiliary_variable(
            value_targets.copy(), name='value_targets')
        application_call.add_auxiliary_variable(
            probs.copy(), name='probs')
        application_call.add_auxiliary_variable(
            prediction_logs, name='prediction_log_probs')

        # Compute some statistics for debugging
        last_character_mask = prediction_mask - rolled_prediction_mask
        last_character_costs = (critic_costs_per_char * last_character_mask).sum(axis=0)
        mean2_output = (
            ((prediction_outputs ** 2) * prediction_mask).sum()
            / prediction_mask.sum()) ** 0.5
        max_output = abs(prediction_outputs * prediction_mask).max()
        expected_reward = (probs[0] * values[0]).sum(axis=-1)
        application_call.add_auxiliary_variable(
            last_character_costs, name='last_character_costs')
        application_call.add_auxiliary_variable(
            critic_costs.mean(), name='mean_critic_cost')
        application_call.add_auxiliary_variable(
            critic_monte_carlo_costs.mean(), name='mean_critic_monte_carlo_cost')
        if self.critic:
            application_call.add_auxiliary_variable(
                actor_costs.mean(), name='mean_actor_cost')
            application_call.add_auxiliary_variable(
                actor_entropies.mean(), name='mean_actor_entropy')
        application_call.add_auxiliary_variable(
            expected_reward.mean(), name='mean_expected_reward')
        application_call.add_auxiliary_variable(
            mean2_output, name='mean2_output')
        application_call.add_auxiliary_variable(
            max_output, name='max_output')

        return total_costs
