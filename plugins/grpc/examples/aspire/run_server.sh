#!/bin/bash

cxx="g++" # g++ needs to be >= 8.0
kaldi_root="/opt/kaldi"

# Safety mechanism (possible running this script with modified arguments)
. utils/parse_options.sh || exit 1
[[ $# -ge 1 ]] && {
    echo "Wrong arguments!"
    exit 1
}

. utils/setup_aspire_chain_model.sh --kaldi-root $kaldi_root || exit 1;
cd ../../

if ! [ -x build/kaldi_serve_app ]; then
    make -j KALDI_ROOT=$kaldi_root CXX=$cxx || exit 1;
fi

./build/kaldi_serve_app examples/aspire/model_spec.toml
