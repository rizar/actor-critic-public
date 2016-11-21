#!/usr/bin/env python

import h5py
import numpy
import argparse
import cPickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Add vocabulary to a data source")
    parser.add_argument('dataset')
    parser.add_argument('source')
    parser.add_argument('vocabulary')
    args = parser.parse_args()

    vocabulary = cPickle.load(open(args.vocabulary))

    with h5py.File(args.dataset, mode='a') as f:
        vocabulary_arr = numpy.fromiter(
            vocabulary.iteritems(),
            dtype=[
                ('key','S{}'.format(max(len(k) for k in vocabulary.keys()))),
                ('val','int32')])
        f[args.source + '_vocab'] = vocabulary_arr
