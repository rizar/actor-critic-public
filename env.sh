#!/usr/bin/env bash

# The directory where the script is
export LVSR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# For PyFST
if [ -n "$KALDI_ROOT" ];
then
    export LD_LIBRARY_PATH=$KALDI_ROOT/tools/openfst/lib:$LD_LIBRARY_PATH
fi

export BLOCKS_CONFIG=$LVSR/config/blocks.yaml
export THEANORC=$LVSR/config/theano.rc:$HOME/.theanorc

export PYTHONPATH=$LVSR:$PYTHONPATH
export PATH=$LVSR/bin:$PATH
