#!/bin/bash

kaldi_root="/opt/kaldi"

# Safety mechanism (possible running this script with modified arguments)
. utils/parse_options.sh || exit 1
[[ $# -ge 1 ]] && {
    echo "Wrong arguments!"
    exit 1
}

aspire_model_path="exp/tdnn_7b_chain_online"
example_root=$(pwd)

escape_path () {
    echo $(echo $1 | sed -e 's/\.//g' | sed -e 's/\/\//\//g' | sed -e 's/\//\\\//g')
}

if ! [ -d model/ ]; then
    
    cd $kaldi_root/egs/aspire/s5

    if ! [ -f aspire_chain_model.tar.bz2 ]; then
        echo "downloading ASPIRE Chain Model..."
        wget https://kaldi-asr.org/models/1/0001_aspire_chain_model_with_hclg.tar.bz2 -O aspire_chain_model.tar.bz2
    fi

    if ! [ -d exp/ ]; then
        tar -xvf aspire_chain_model.tar.bz2
    fi

    if ! [ -f $aspire_model_path/conf/online.conf ]; then
        echo "generating files needed for online decoding"
        . steps/online/nnet3/prepare_online_decoding.sh \
                --mfcc-config conf/mfcc_hires.conf data/lang_chain \
                exp/nnet3/extractor exp/chain/tdnn_7b exp/tdnn_7b_chain_online
    fi

    if ! [ -d model/ ]; then
        echo "copying essential files from aspire recipe"
        mkdir model/
        cp $aspire_model_path/final.mdl model/
        cp $aspire_model_path/graph_pp/HCLG.fst model/
        cp $aspire_model_path/graph_pp/words.txt model/
        cp $aspire_model_path/graph_pp/phones/word_boundary.int model/
        cp -r $aspire_model_path/conf model/conf
        cp -r $aspire_model_path/ivector_extractor model/ivector_extractor

        escaped_model_path="$(escape_path "$kaldi_root/egs/aspire/s5/$aspire_model_path/")"

        sed -i "s/$escaped_model_path//g" "model/conf/online.conf"
        sed -i "s/$escaped_model_path//g" "model/conf/ivector_extractor.conf"
    fi

    mv model $example_root/model

    cd $example_root
fi
