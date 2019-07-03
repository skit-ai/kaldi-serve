#!/bin/bash

rm build/kaldi_serve*
rm protos/kaldi_serve.pb*
rm protos/kaldi_serve.grpc*

make -j8