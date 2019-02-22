if [[ -z $KALDI_ROOT ]]
then
   echo "KALDI_ROOT path has not been set. Maybe run install.sh again. Exiting"
   exit 1
fi

if [ -n "${KALDI_ROOT+1}" ]
then
    echo "KALDI_ROOT path has not been set. Maybe run install.sh again. Exiting"
    exit 1
fi

# Setting paths to useful tools
export PATH=$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/src/onlinebin:$KALDI_ROOT/src/nnet2bin:$KALDI_ROOT/src/nnet3bin:$KALDI_ROOT/src/chainbin:$KALDI_ROOT/src/ivectorbin:$KALDI_ROOT/src/online2bin:$KALDI_ROOT/egs/wsj/s5/local:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$PWD:$PATH

# Variable needed for proper data sorting
export LC_ALL=C

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$KALDI_ROOT/tools/openfst/lib
