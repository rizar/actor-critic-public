import math
import numpy
from collections import defaultdict

COPY = 0
INSERTION = 1
DELETION = 2
SUBSTITUTION = 3

INFINITY = 10 ** 9


def _edit_distance_matrix(y, y_hat, special_tokens=None):
    """Returns the matrix of edit distances.

    Parameters
    ----------

    Returns
    -------
    dist : numpy.ndarray
        dist[i, j] is the edit distance between the first
    action : numpy.ndarray
        action[i, j] is the action applied to y_hat[j - 1]  in a chain of
        optimal actions transducing y_hat[:j] into y[:i].
        i characters of y and the first j characters of y_hat.
    special_tokens : set
        Tokens, for which deletion is free.
    """
    if not special_tokens:
        special_tokens = set()
    dist = numpy.zeros((len(y) + 1, len(y_hat) + 1), dtype='int64')
    insertion_cost = numpy.ones(len(y))
    deletion_cost = numpy.ones(len(y_hat))
    for i in range(len(y)):
        if y[i] in special_tokens:
            insertion_cost[i] = 0
    for j in range(len(y_hat)):
        if y_hat[j] in special_tokens:
            deletion_cost[j] = 0
    dist[1:, 0] = insertion_cost.cumsum()
    dist[0, 1:] = deletion_cost.cumsum()

    for i in xrange(1, len(y) + 1):
        for j in xrange(1, len(y_hat) + 1):
            if y[i - 1] != y_hat[j - 1]:
                cost = 1
            else:
                cost = 0
            insertion_dist = dist[i - 1][j] + insertion_cost[i - 1]
            deletion_dist = dist[i][j - 1] + deletion_cost[j - 1]
            substitution_dist = dist[i - 1][j - 1] + 1 if cost else INFINITY
            copy_dist = dist[i - 1][j - 1] if not cost else INFINITY
            best = min(insertion_dist, deletion_dist,
                       substitution_dist, copy_dist)

            dist[i][j] = best

    return dist


def _bleu(y, y_hat, n=4):
    """ BLEU score between the reference sequence y
    and y_hat for each partial sequence ranging
    from the first input token to the last

    Parameters
    ----------
    y : vector
        The reference matrix with dimensions of number
        of words (rows) by batch size (columns)
    y_hat : vector
        The predicted matrix with same dimensions
    n : integer
        highest n-gram order in the Bleu sense
        (e.g Bleu-4)

    Returns
    -------
    results : vector (len y_hat)
        Bleu scores for each partial sequence
        y_hat_1..T from T = 1 to len(y_hat)
    """
    bleu_scores = numpy.zeros((len(y_hat), n))

    # count reference ngrams
    ref_counts = defaultdict(int)
    for k in xrange(1, n+1):
        for i in xrange(len(y) - k + 1):
            ref_counts[tuple(y[i:i + k])] += 1

    # for each partial sequence, 1) compute addition to # of correct
    # 2) apply brevity penalty
    # ngrams, magic stability numbers from pycocoeval
    ref_len = len(y)
    pred_counts = defaultdict(int)
    correct = numpy.zeros(4)
    for i in xrange(1, len(y_hat) + 1):
        for k in xrange(i, max(-1, i - n), -1):
            # print i, k
            ngram = tuple(y_hat[k-1:i])
            # UNK token hack. Must work for both indices
            # and words. Very ugly, I know.
            if 0 in ngram or 'UNK' in ngram:
                continue
            pred_counts[ngram] += 1
            if pred_counts[ngram] <= ref_counts.get(ngram, 0):
                correct[len(ngram)-1] += 1

        # compute partial bleu score
        bleu = 1.
        for j in xrange(n):
            possible = max(0, i - j)
            bleu *= float(correct[j] + 1.) / (possible + 1.)
            bleu_scores[i - 1, j] = bleu ** (1./(j+1))

        # brevity penalty
        if i < ref_len:
            ratio = (i + 1e-15)/(ref_len + 1e-9)
            bleu_scores[i - 1, :] *= math.exp(1 - 1/ratio)

    return bleu_scores.astype('float32'), correct, pred_counts, ref_counts


def edit_distance(y, y_hat):
    """Edit distance between two sequences.

    Parameters
    ----------
    y : str
        The groundtruth.
    y_hat : str
        The recognition candidate.

   the minimum number of symbol edits (i.e. insertions,
    deletions or substitutions) required to change one
    word into the other.

    """
    return _edit_distance_matrix(y, y_hat)[-1, -1]


def wer(y, y_hat):
    return edit_distance(y, y_hat) / float(len(y))


def reward_matrix(y, y_hat, alphabet, eos_label):
    dist, _,  = _edit_distance_matrix(y, y_hat)
    y_alphabet_indices = [alphabet.index(c) for c in y]
    if y_alphabet_indices[-1] != eos_label:
        raise ValueError("Last character of the groundtruth must be EOS")

    # Optimistic edit distance for every y_hat prefix
    optim_dist = dist.min(axis=0)
    pess_reward = -optim_dist

    # Optimistic edit distance for every y_hat prefix plus a character
    optim_dist_char = numpy.tile(
        optim_dist[:, None], [1, len(alphabet)]) + 1
    pess_char_reward = numpy.tile(
        pess_reward[:, None], [1, len(alphabet)]) - 1
    for i in range(len(y)):
        for j in range(len(y_hat) + 1):
            c = y_alphabet_indices[i]
            cand_dist = dist[i, j]
            if cand_dist < optim_dist_char[j, c]:
                optim_dist_char[j, c] = cand_dist
                pess_char_reward[j, c] = -cand_dist
    for j in range(len(y_hat) + 1):
        # Here we rely on y[-1] being eos_label
        pess_char_reward[j, eos_label] = -dist[len(y) - 1, j]
    return pess_char_reward

def gain_matrix(y, y_hat, alphabet=None, given_reward_matrix=None,
                eos_label=None):
    y_hat_indices = [alphabet.index(c) for c in y_hat]
    reward = (given_reward_matrix.copy() if given_reward_matrix is not None
              else reward_matrix(y, y_hat, alphabet, eos_label))
    reward[1:] -= reward[:-1][numpy.arange(len(y_hat)), y_hat_indices][:, None]
    return reward
