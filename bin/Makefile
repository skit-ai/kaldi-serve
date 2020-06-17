KALDI_ROOT=/opt/kaldi
CUDA_TK_ROOT=/usr/local/cuda

HOST_SYSTEM = $(shell uname | cut -f 1 -d_)
SYSTEM ?= $(HOST_SYSTEM)
CXX = g++
CXXFLAGS += -std=c++11 -O3 -DHAVE_CUDA -DKALDI_DOUBLEPRECISION=0 \
	-Wno-sign-compare -Wno-unused-local-typedefs \
	-Wno-unused-variable -Winit-self -O3

LDFLAGS += -L/usr/local/lib -Wl,--as-needed -ldl \
	'-Wl,-rpath,$$ORIGIN/../lib' -L${KALDI_ROOT}/src/lib -L${KALDI_ROOT}/tools/openfst/lib

KALDISERVE_INCLUDES = -I../include -I${KALDI_ROOT}/src -I${KALDI_ROOT}/tools/openfst/include -I${CUDA_TK_ROOT}/include
KALDISERVE_LIBS = -rdynamic -lm -lpthread -ldl -lkaldi-decoder -lkaldi-lat -lkaldi-fstext \
	-lkaldi-hmm -lkaldi-feat -lkaldi-transform -lkaldi-gmm -lkaldi-tree -lkaldi-util \
	-lkaldi-base -lkaldi-online2 -lkaldi-ivector -lfst -lkaldi-matrix -lkaldi-cudamatrix -lkaldi-nnet3 \
	-lkaldi-rnnlm -lkaldi-chain -lkaldi-cudadecoder -lkaldi-cudafeat -lkaldiserve \
	-static-libstdc++

all: build/batched-gpu-decoder

build/batched-gpu-decoder: batched-gpu-decoder.cc
	$(CXX) batched-gpu-decoder.cc $(CXXFLAGS) $(KALDISERVE_INCLUDES) $(LDFLAGS) $(KALDISERVE_LIBS) -o $@

clean:
	rm -f ./build/*