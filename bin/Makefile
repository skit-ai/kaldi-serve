KALDI_ROOT=/opt/kaldi

HOST_SYSTEM = $(shell uname | cut -f 1 -d_)
SYSTEM ?= $(HOST_SYSTEM)
CXX = g++
CXXFLAGS += -std=c++11 -O3

LDFLAGS += -L/usr/local/lib -Wl,--as-needed -ldl

CUDA_INCLUDES = 
CUDA_LIBS = 

KALDISERVE_INCLUDES = -I../include -I${KALDI_ROOT}/src -I${KALDI_ROOT}/tools/openfst/include
KALDISERVE_LIBS = -rdynamic -lkaldiserve -static-libstdc++

all: batched-gpu-decoder

batched-gpu-decoder: batched-gpu-decoder.cc
	$(CXX) $(CXXFLAGS) $(KALDISERVE_INCLUDES) $(LDFLAGS) $(KALDISERVE_LIBS) -o $@ batched-gpu-decoder.cc

clean:
	rm -f batched-gpu-decoder