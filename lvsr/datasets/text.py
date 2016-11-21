"""Wraps Fuel H5PYDataset."""
import glob
from fuel import datasets

all_chars = ([chr(ord('a') + i) for i in range(26)] +
             [chr(ord('0') + i) for i in range(10)] +
             [',', '.', '!', '?', '#'] +
             [' ', '^', '$'])
code2char = dict(enumerate(all_chars))
char2code = {v: k for k, v in code2char.items()}

def _lower(s):
    return s.lower()


class TextDataset(datasets.TextFile):
    def __init__(self, file_or_path, which_sets, sources, target_source, **kwargs):
        if target_source != 'text':
            raise ValueError
        if sources != ('text',):
            raise ValueError
        # Ignore which_sets and file_or_path
        char2code_without_spec_tokens = dict(char2code)
        del char2code_without_spec_tokens['$']
        del char2code_without_spec_tokens['^']
        file_or_path = glob.glob(file_or_path)
        if not file_or_path:
            raise ValueError
        super(TextDataset, self).__init__(
            files=file_or_path,
            dictionary=char2code_without_spec_tokens,
            level='character', preprocess=_lower,
            # Bos and Eos token are added at
            # a higher level
            bos_token=None, eos_token=None, unk_token='#',
            **kwargs)
        self.sources = sources
        self.target_source = target_source

        self.num2char = dict(enumerate(all_chars))
        self.char2num = {v: k for k, v in code2char.items()}
        self.num_labels = len(self.num2char)
        self.eos_label = self.char2num['$']
        self.bos_label = self.char2num.get('^')

    def token_map(self, source):
        if source != 'text':
            raise ValueError
        return self.char2num

    def dim(self, source):
        raise ValueError

    def decode(self, labels, keep_all=False):
        return [self.num2char[label] for label in labels
                if keep_all or label not in [self.eos_label, self.bos_label]]

    def pretty_print(self, labels, example):
        return "".join(self.decode(labels))

    def monospace_print(self, labels):
        return "".join(self.decode(labels, keep_all=True))
