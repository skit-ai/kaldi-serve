#!/bin/sh
. ./scripts/path.sh

online2-wav-nnet3-latgen-faster --acoustic-scale=1.0 --max_active=7000 --beam=13.0 --lattice-beam=6.0 --online=true --frame-subsampling-factor=3 --config=$2 $3 $4 "ark:echo utterance-id1 utterance-id1|" "scp:echo utterance-id1 $1|" "ark:|lattice-best-path --word-symbol-table=$5 ark:- ark,t:- |$KALDI_ROOT/egs/vernacularai/s1/utils/int2sym.pl -f 2- $5"

rm $1
