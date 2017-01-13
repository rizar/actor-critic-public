from matplotlib import pyplot
import pandas
import re
from StringIO import StringIO

from blocks.serialization import load


if 'logs' not in locals():
    logs = {}
    dfs = {}


def load_model(path):
    log = load(open(path), name='log')
    df = pandas.DataFrame.from_dict(log, orient='index')
    name = path[:-4] if path[-3:] == 'tar' else path
    logs[name] = log
    dfs[name] = df
    print path
    print log.status['iterations_done'], ' iterations done'
    if 'best_valid_per' in log.status:
        print 'best_valid_per', log.status['best_valid_per']
    if 'best_valid_train_cost' in log.status:
        print 'best_valid_train_cost', log.status['best_valid_train_cost']
    if 'best_valid_mean_total_reward' in log.status:
        print 'best_valid_mean_total_reward', log.status['best_valid_mean_total_reward']
    if 'mean_total_reward' in dfs[name]:
        print 'mean_total_reward:', dfs[name].mean_total_reward[-10:].mean()


def compare_log_likelihood(models, s=slice(None)):
    pyplot.figure(figsize=(10, 5))
    legend = []
    for m in models:
        dfs[m].train_cost.astype('float32').dropna().loc[s].plot()
        dfs[m].valid_train_cost.astype('float32').dropna().loc[s].plot(ls='--')
        legend += ['train_' + m]
        legend += ['valid_' + m]
    pyplot.legend(legend)


def compare_actor_critic_costs(models, s=slice(None)):
    pyplot.figure(figsize=(10, 5))
    legend = []
    for m in models:
        dfs[m].readout_costs_mean_critic_cost.astype('float32').dropna().loc[s].plot()
        legend += [m]
    pyplot.legend(legend)


def compare_rewards_and_errors(
        models,
        s=slice(None),
        curves=('train', 'valid', 'mixed')):
    pyplot.figure(figsize=(15, 10))
    legend = []
    for m in models:
        if 'train' in curves:
            dfs[m].mean_total_reward.astype('float32').dropna().loc[s].plot()
            legend += ['train_' + m]
        if 'valid' in curves and 'valid_mean_total_reward' in dfs[m]:
            dfs[m].valid_mean_total_reward.astype('float32').dropna().loc[s].plot(ls='--')
            legend += ['valid_' + m]
        if 'mixed' in curves and 'mixed_valid_mean_total_reward' in dfs[m]:
            dfs[m].mixed_valid_mean_total_reward.astype('float32').dropna().loc[s].plot(ls='-.')
            legend += ['mixed_' + m]
    pyplot.legend(legend, loc='best')
    pyplot.xlabel("Iterations")
    pyplot.ylabel("Reward")
    pyplot.title('Reward')
    pyplot.show()

    pyplot.figure(figsize=(15, 5))
    for m in models:
        dfs[m].readout_costs_mean_critic_cost.astype('float32').dropna().loc[s].plot()
    pyplot.legend(models, loc='best')
    pyplot.xlabel("Iterations")
    pyplot.ylabel("TD Square Error")
    pyplot.title('Critic error')
    pyplot.show()

    pyplot.figure(figsize=(15, 5))
    for m in models:
        dfs[m].readout_costs_mean_actor_cost.astype('float32').dropna().loc[s].plot()
    pyplot.legend(models, loc='best')
    pyplot.xlabel("Iterations")
    pyplot.ylabel("Actor suboptimality")
    pyplot.title('Actor error')
    pyplot.show()


def compare_critic_cost(models, s=slice(None)):
    for m in models:
        dfs[m].readout_costs_mean_critic_cost.astype('float32').dropna().loc[s].plot()
    pyplot.legend(models, loc='best')
    pyplot.xlabel("Iterations")
    pyplot.ylabel("TD Square Error")
    pyplot.title('Critic error')
    pyplot.show()


def compare_score(models, s=slice(None), figsize=None, curves=('train', 'valid'),
                  select=max):
    pyplot.figure(figsize=(10, 5) if not figsize else figsize,
                  dpi=180)
    legend = []
    for m in models:
        if 'train' in curves and 'train_per' in dfs[m]:
            train_score = dfs[m].train_per.astype('float32').dropna().loc[s]
            print "best train score for {}: {}".format(m, select(train_score[1:]))
            train_score.plot()
            legend += ['train_' + m]
        if 'valid' in curves:
            valid_score = dfs[m].valid_per.astype('float32').dropna().loc[s]
            print "best valid score for {}: {}".format(m, select(valid_score[1:]))
            valid_score.plot(ls='--')
            legend += ['valid_' + m]
    pyplot.legend(legend, loc='best')


def compare_gradient_norms(models):
    legend = []
    for m in models:
        dfs[m].total_gradient_norm.astype('float32').dropna().plot(use_index=False)
        legend += [m]
    pyplot.legend(legend)
    pyplot.title('Gradient norm')


def compare_max_adjustments(models):
    legend = []
    for m in models:
        dfs[m].readout_costs_max_adjustment.astype('float32').dropna().plot()
        legend += [m]
    pyplot.legend(legend)
    pyplot.title('Max adjustment')


def compare_weight_entropy(models):
    legend = []
    for m in models:
        dfs[m].average_weights_entropy_per_label.astype('float32').dropna().plot()
        legend += [m]
    pyplot.legend(legend)
    pyplot.title('Weight entropy')
    pyplot.show()


def compare_entropies(models, s=slice(None)):
    legend = []
    for m in models:
        dfs[m].readout_costs_mean_actor_entropy.astype('float32').dropna().loc[s].plot()
        legend += [m]
    pyplot.legend(legend)
    pyplot.title('Entropy')


def compare_mean2_output(models, s=slice(None)):
    legend = []
    for m in models:
        dfs[m].readout_costs_mean2_output.astype('float32').dropna().loc[s].plot()
        legend += [m]
    pyplot.legend(legend, loc='best')
    pyplot.title('mean2_output')


def compare_critic_monte_carlo_cost(models, s=slice(None)):
    legend = []
    for m in models:
        dfs[m].readout_costs_mean_critic_monte_carlo_cost.astype('float32').dropna().loc[s].plot()
        legend += [m]
    pyplot.legend(legend, loc='best')
    pyplot.title('critic_monte_carlo_cost')


def tex_escape(text):
    conv = {
        '%': r'\%',
        '$': r'\$',
        '#': r'\#',
        '_': r'\_',
        '~': r'\textasciitilde{}',
        '&apos;': r"'",
        '&quot;': r'"',
        #'&': r'\&',
        #'^': r'\^',
        '<s>': r' BOS ',
        '</s>': r' EOS ',
    }
    regex = re.compile('|'.join(re.escape(unicode(key))
                       for key in sorted(conv.keys(), key=lambda item: - len(item))))
    return regex.sub(lambda match: conv[match.group()], text)


class Analyzer(object):

    def __init__(self, log_name, num2word, word2num, words=True):
        self.num2word = num2word
        self.word2num = word2num
        if words:
            self.eos = '</s>'
            self.sep = ' '
            self.split = str.split
        else:
            self.eos = '$'
            self.sep = ''
            self.split = lambda x: x

        self.log = logs[log_name]
        self.train_cost = [self.log[t].get('train_cost') for t in range(0, self.log.status['iterations_done'] + 1)]
        self.rewards = [self.log[t].get('readout_costs_rewards') for t in range(0, self.log.status['iterations_done'] + 1)]
        self.mean_reward = [self.log[t].get('mean_reward') for t in range(0, self.log.status['iterations_done'] + 1)]
        self.critic_cost = [self.log[t].get('readout_costs_mean_critic_cost') for t in range(0, self.log.status['iterations_done'] + 1)]
        self.actor_cost = [self.log[t].get('readout_costs_mean_actor_cost') for t in range(0, self.log.status['iterations_done'] + 1)]

        self.inputs = [self.log[t].get('average_inputs') for t in range(0, self.log.status['iterations_done'] + 1)]
        self.predictions = [self.log[t].get('average_predictions') for t in range(0, self.log.status['iterations_done'] + 1)]
        self.prediction_masks = [self.log[t].get('readout_costs_prediction_mask') for t in range(0, self.log.status['iterations_done'] + 1)]
        self.groundtruth = [self.log[t].get('average_groundtruth') for t in range(0, self.log.status['iterations_done'] + 1)]

        self.value_biases = [self.log[t].get('readout_costs_value_biases') for t in range(0, self.log.status['iterations_done'] + 1)]
        self.values = [self.log[t].get('readout_costs_values') for t in range(0, self.log.status['iterations_done'] + 1)]
        self.probs = [self.log[t].get('readout_costs_probs') for t in range(0, self.log.status['iterations_done'] + 1)]
        self.outputs = [self.log[t].get('readout_costs_outputs') for t in range(0, self.log.status['iterations_done'] + 1)]

        self.prediction_values = [self.log[t].get('readout_costs_prediction_values') for t in range(0, self.log.status['iterations_done'] + 1)]
        self.prediction_outputs = [self.log[t].get('readout_costs_prediction_outputs') for t in range(0, self.log.status['iterations_done'] + 1)]

        self.value_targets = [self.log[t].get('readout_costs_value_targets') for t in range(0, self.log.status['iterations_done'] + 1)]


    def print_critic_suggestions(self, it, i,
                                 just_from_groundtruth=False,
                                 p_threshold=0.0, num_words=5, crop_at=None,
                                 output='values'):
        result = StringIO()

        prediction_words = self.split(self.predictions[it][i])
        prediction_mask = self.prediction_masks[it][:, i]
        groundtruth_words = self.split(self.groundtruth[it][i])

        prediction_words = prediction_words[:int(prediction_mask.sum())]
        if crop_at and len(prediction_words) > crop_at:
            prediction_words = prediction_words[:crop_at]
        groundtruth_words = groundtruth_words[:groundtruth_words.index(self.eos) + 1]


        print >>result, r"$\begin{array}{cc}"
        print >>result, r"\mathbf{{Groundtruth}} & \textrm{{ {} }}\\".format(tex_escape(self.sep.join(groundtruth_words)))
        print >>result, r"\mathbf{{Prediction}} & \textrm{{ {} }}".format(tex_escape(self.sep.join(prediction_words)))
        print >>result, r"\end{array}$"
        print >>result

        groundtruth_nums = set([self.word2num[word] for word in self.split(self.groundtruth[it][i])])

        print >>result, "$\\begin{array}{ccccc}"
        print >>result, r"\textrm{Word} & \textrm{Reward} & \textrm{Actor prob}. & \textrm{Q} & \textrm{Best Q} \\"
        for step in range(len(prediction_words)):
            step_values = self.values[it][step, i]
            step_probs = self.probs[it][step, i]
            step_expected_value = (step_values * step_probs).sum()
            if output == 'values':
                pass
            elif output == 'advantages':
                step_values = step_values - step_expected_value
            elif output == 'gradients':
                step_values = step_probs * (step_values - step_expected_value)
            else:
                raise ValueError
            actions = enumerate(step_values)
            if just_from_groundtruth:
                actions = [(n, q) for n, q in actions if n in groundtruth_nums]
            else:
                actions = [(n, q) for (n, q) in actions if step_probs[n] > p_threshold]
            best = list(sorted(actions, key=lambda (i, o): -o))[:num_words]
            print >>result, r"\textrm{{ {} }} & {:.3f} & {:.6f} & {:.3f} &".format(
                tex_escape(prediction_words[step]),
                self.rewards[it][step, i],
                step_probs[self.word2num[prediction_words[step]]],
                step_values[self.word2num[prediction_words[step]]],
            )
            print >>result, "\,".join([
                r"\textrm{{ {} }}({:.3f}, {:.6f})".format(
                    tex_escape(self.num2word[c]), q, step_probs[c])
                for c, q in best]),
            print >>result, '\\\\'
            if prediction_words[step] == self.eos:
                break
        print >>result, "\\end{array}$"
        result.seek(0)
        return result.read()
