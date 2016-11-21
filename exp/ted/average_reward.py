import sys
from lvsr.error_rate import _bleu
from lvsr.bricks import BleuReward
from lvsr.datasets.mt import H5PyMTDataset
import numpy
import theano
from theano import tensor

brick = BleuReward(1, 2, True)
prediction = tensor.lmatrix('p')
groundtruth = tensor.lmatrix('g')
reward = brick.apply(prediction, tensor.ones_like(prediction, dtype='float32'),
                     groundtruth, tensor.ones_like(groundtruth, dtype='float32'))
reward = reward.sum()
f = theano.function([prediction, groundtruth], [reward])

ted = H5PyMTDataset(
    'targets',
    file_or_path='/data/lisatmp4/bahdanau/data/TED/de-en/ted.h5',
    which_sets=('train',))
num2word = ted.num2word
word2num = ted.word2num

total_bleu = 0.0
total_reward = 0.0
num_examples = 0

for g, p in zip(open(sys.argv[1]), open(sys.argv[2])):
    g_words = g.strip().split()
    p_words = p.strip().split()
    g_indices = numpy.array([1] + [word2num.get(w, 0) for w in g_words] + [2])[:, None]
    p_indices = numpy.array([1] + [word2num.get(w, 0) for w in p_words] + [2])[:, None]

    bleu = _bleu(g_words, p_words)[0]
    assert len(bleu) == len(p_words)
    bleu = (bleu[-1, 3] if len(bleu) else 0.) * (len(g_words) + 2)
    total_bleu += bleu

    r = f(p_indices, g_indices)[0]
    total_reward += r

    num_examples += 1
    # print bleu, r

print total_bleu / num_examples, total_reward / num_examples
