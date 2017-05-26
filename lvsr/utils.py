import sys
import logging
from contextlib import contextmanager


def global_push_initialization_config(brick, initialization_config,
                                      filter_type=object):
    #TODO: this needs proper selectors! NOW!
    if not brick.initialization_config_pushed:
        raise Exception("Please push_initializatio_config first to prevent it "
                        "form overriding the changes made by "
                        "global_push_initialization_config")
    if isinstance(brick, filter_type):
        for k,v in initialization_config.items():
            if hasattr(brick, k):
                setattr(brick, k, v)
    for c in brick.children:
        global_push_initialization_config(
            c, initialization_config, filter_type)


def rename(var, name):
    var.name = name
    return var


class Fork(object):
    def __init__(self, file1, file2):
        self.file1 = file1
        self.file2 = file2

    def write(self, data):
        self.file1.write(data)
        self.file2.write(data)

    def flush(self):
        self.file1.flush()
        self.file2.flush()


@contextmanager
def replace_logging_stream(file_):
    root = logging.getLogger()
    if len(root.handlers) != 1:
        raise ValueError("Don't know what to do with many handlers")
    if not isinstance(root.handlers[0], logging.StreamHandler):
        raise ValueError
    stream = root.handlers[0].stream
    root.handlers[0].stream = file_
    try:
        yield
    finally:
        root.handlers[0].stream = stream

@contextmanager
def replace_standard_stream(stream_name, file_):
    stream = getattr(sys, stream_name)
    setattr(sys, stream_name, file_)
    try:
        yield
    finally:
        setattr(sys, stream_name, stream)
