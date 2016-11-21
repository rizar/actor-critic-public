#!/usr/bin/env python

import h5py
import random
import argparse

from shutil import copyfile


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Shuffle a subset of a Fuel style "
                                     "HDF5 file.")
    parser.add_argument('-s', dest='subset',
                        help="Subset to shuffle")
    parser.add_argument('-S', dest='seed',
                        default=101,
                        help="Seed to use")
    parser.add_argument('input', help='Input file')
    parser.add_argument('output', help='Output file')
    args = parser.parse_args()

    print "Reading {0} and writing to {1}".format(args.input, args.output)
    print "Copying file...",
    copyfile(args.input, args.output)
    print "done"

    random.seed(args.seed)

    with h5py.File(args.input, mode='r') as fin:
        split = fin.attrs['split']
        sources = [i[1] for i in split if i[0] == args.subset]
        print 'Permuting the following sources:', sources
        intervals = [(i[2], i[3]) for i in split if i[0] == args.subset]
        # every value of the intervals list should be identical
        assert intervals.count(intervals[0]) == len(intervals)
        interval = intervals[0]
        indices = range(*interval)
        random.shuffle(indices)
        with h5py.File(args.output, mode='r+') as fout:
            for source in sources:
                for i in range(*interval):
                    fout[source][i] = fin[source][indices[i]]
