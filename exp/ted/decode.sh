#!/usr/bin/env bash
set -ex

if [ `hostname -d` == 'helios' ]
then
    export LVSR_EXP_TED="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    source $LVSR_EXP_TED/../../env.sh
    export FUEL_DATA_PATH=$HOME/data
    export THEANO_FLAGS='device=gpu,floatX=float32,optimizer=fast_run,lib.cnmem=0.9,reoptimize_unpickled_function=True'
fi

CONFIG=$1
DISCOUNTS=$2
PREFIX=$3
DIRS=${@:4}

function decode {
    $LVSR/bin/run.py search --part $1 ../${PREFIX}_best.tar  $LVSR/exp/ted/configs/${CONFIG}.yaml monitoring.search.beam_size 1 >$1".bs1"
    $LVSR/bin/run.py search --part $1 ../${PREFIX}_best.tar  $LVSR/exp/ted/configs/${CONFIG}.yaml monitoring.search.beam_size 10 >$1".bs10"
    for ds in $DISCOUNTS
    do
        suffix=`echo $ds | sed s/\\.//`
        $LVSR/bin/run.py search --part $1 ../${PREFIX}_best.tar  $LVSR/exp/ted/configs/${CONFIG}.yaml monitoring.search.char_discount $ds monitoring.search.beam_size 10 >$1".bs10.cd$suffix"
    done
}

for DIR in $DIRS
do    
    cd $DIR/${PREFIX}_decoded

    decode dev
    decode test

    cd ../../
done

