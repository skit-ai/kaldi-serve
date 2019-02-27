#!/bin/sh
. ./scripts/path.sh

online2-wav-gmm-latgen-faster --acoustic-scale=0.0714 --adaptation-delay=0.5 --beam=13.0 --config=$2 $3 "ark:echo utterance-id1 utterance-id1|" "scp:echo utterance-id1 $1|" "ark:|lattice-best-path --word-symbol-table=$4 ark:- ark,t:- |$KALDI_ROOT/egs/vernacularai/s1/utils/int2sym.pl -f 2- $4"
rm $1
