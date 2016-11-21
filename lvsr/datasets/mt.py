from fuel.datasets.hdf5 import H5PYDataset


class H5PyMTDataset(H5PYDataset):
    def __init__(self, target_source, **kwargs):
        super(H5PyMTDataset, self).__init__(**kwargs)
        self.open()

        self._token_map_cache = {}

        self.word2num = self.token_map(target_source)
        self.num2word = {num: word for word, num in self.word2num.items()}
        self.num_labels = len(self.num2word)
        self.bos_label = self.word2num['<s>']
        self.eos_label = self.word2num['</s>']

    def token_map(self, source):
        if not source in self._token_map_cache:
            self._token_map_cache[source] = dict(self._file_handle[source + '_vocab'])
        return self._token_map_cache[source]

    def decode(self, labels, keep_all=False):
        return [self.num2word[label] for label in labels
                if (label != self.eos_label or keep_all)
                and (label != self.bos_label or keep_all)]

    def pretty_print(self, labels, example):
        labels = self.decode(labels)
        return ' '.join(labels)

    def monospace_print(self, labels):
        labels = self.decode(labels, keep_all=True)
        return ' '.join(labels)
