#!/usr/bin/env python

import h5py
import numpy
import argparse
import cPickle

from fuel.datasets.hdf5 import H5PYDataset

def pack(f, name, dataset_pathes):
    datasets = [cPickle.load(open(path)) for path in dataset_pathes]
    data = sum(datasets, [])
    dtype = h5py.special_dtype(vlen=numpy.dtype('int32'))
    table = f.create_dataset(name, (len(data),), dtype=dtype)
    for i, example in enumerate(data):
        table[i] = example
    return numpy.array([len(d) for d in datasets])


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Pack data to HDF5")
    parser.add_argument('-s', dest='sources', nargs='*', help="Source datasets")
    parser.add_argument('-t', dest='targets', nargs='*', help="Target datasets")
    parser.add_argument('-n', dest='names', nargs='*', help="Dataset names")
    parser.add_argument('-i', dest='add_ids',
                        action='store_true', default=False,
                        help="Add integer IDs")
    parser.add_argument('dest', help="Destination")
    args = parser.parse_args()

    assert len(args.sources) == len(args.targets)
    assert len(args.sources) == len(args.names)
    with h5py.File(args.dest, mode='w') as f:
        lengths = pack(f, "sources", args.sources)
        assert numpy.all(lengths == pack(f, "targets", args.targets))

        offsets = [0] + list(lengths.cumsum())
        total_len = offsets[-1]
        if args.add_ids:
            id_table = f.create_dataset('ids',
                                        data=numpy.arange(total_len,
                                                          dtype='int32'))

            split_dict = {
                args.names[i]:
                    {'sources': (offsets[i], offsets[i + 1]),
                     'targets': (offsets[i], offsets[i + 1]),
                     'ids': (offsets[i], offsets[i + 1])}
                for i in range(len(args.names))}
        else:
            split_dict = {
                args.names[i]:
                    {'sources': (offsets[i], offsets[i + 1]),
                     'targets': (offsets[i], offsets[i + 1])}
                for i in range(len(args.names))}

        f.attrs['split'] = H5PYDataset.create_split_array(split_dict)
