export LVSR_BIN="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source $LVSR_BIN/../env.sh
export FUEL_DATA_PATH=~/data
export THEANO_FLAGS='device=gpu,floatX=float32,optimizer=fast_run,lib.cnmem=0.9,reoptimize_unpickled_function=True'
# On cluster the temporary directory often overflows
export BLOCKS_TEMPDIR=$HOME

TIMEOUT=$1
GPU=$2
ARGS=${@:3}

GPU_FLAGS=''
if [ $GPU == k80 ]
then
    GPU_FLAGS="--pbsFlags='-lfeature=k80'"
elif [ $GPU == k20 ] 
then
    GPU_FLAGS="--pbsFlags='-lfeature=k20'"
elif [ $GPU == any ]
then
    GPU_FLAGS=""
else
    echo "Unknown GPU type: $GPU"
    exit 1
fi

# send SIGTERM after TIMEOUT seconds
timeout -s 15 $TIMEOUT $LVSR/bin/run.py $ARGS
# check the exit code if timeout
if [ $?  == 124 ]
then
    # reserve 12 hours of GPU time,
    # override dangerous settings
    smart-dispatch -q gpu_1 -t 6:00:00 $GPU_FLAGS launch bash -x $LVSR/bin/run_cluster.sh\
        $TIMEOUT $GPU $ARGS --params="''" --start-stage="''"
fi
