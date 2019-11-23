FROM vernacularai/kaldi:2019-11-23 as builder

# gRPC Pre-requisites - https://github.com/grpc/grpc/blob/master/BUILDING.md
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    build-essential \
    autoconf \
    libtool \
    pkg-config \
    libgflags-dev \
    libgtest-dev \
    clang \
    libc++-dev \
    libboost-all-dev \
    curl \
    vim

# Install gRPC
RUN cd /home/ && \
    git clone -b $(curl -L https://grpc.io/release) https://github.com/grpc/grpc && \
    cd /home/grpc/ && \
    git submodule update --init && \
    make && \
    make install

# Install Protobuf v3
RUN cd /home/grpc/third_party/protobuf && make install

WORKDIR /home/app

COPY . .

ENV KALDI_ROOT="/home/kaldi" \
    LD_LIBRARY_PATH="/home/kaldi/tools/openfst/lib:/home/kaldi/src/lib"

RUN make

FROM debian:jessie-slim
WORKDIR /home/app

COPY --from=builder /home/app/build build
COPY --from=builder /home/app/resources resources

ENV LD_LIBRARY_PATH="/usr/local/lib:/home/kaldi/tools/openfst/lib:/home/kaldi/src/lib"

COPY --from=builder /usr/local/lib/libgrpc++.so.1 /usr/local/lib/libgrpc++.so.1
COPY --from=builder /usr/local/lib/libgrpc++_reflection.so.1 /usr/local/lib/libgrpc++_reflection.so.1
COPY --from=builder /usr/lib/x86_64-linux-gnu/libboost_system.so.1.62.0 /usr/local/lib/libboost_system.so.1.62.0
COPY --from=builder /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.62.0 /usr/local/lib/libboost_filesystem.so.1.62.0
COPY --from=builder /home/kaldi/src/lib/libkaldi-decoder.so /home/kaldi/src/lib/libkaldi-decoder.so
COPY --from=builder /home/kaldi/src/lib/libkaldi-fstext.so /home/kaldi/src/lib/libkaldi-fstext.so
COPY --from=builder /home/kaldi/src/lib/libkaldi-hmm.so /home/kaldi/src/lib/libkaldi-hmm.so
COPY --from=builder /home/kaldi/src/lib/libkaldi-feat.so /home/kaldi/src/lib/libkaldi-feat.so
COPY --from=builder /home/kaldi/src/lib/libkaldi-util.so /home/kaldi/src/lib/libkaldi-util.so
COPY --from=builder /home/kaldi/src/lib/libkaldi-matrix.so /home/kaldi/src/lib/libkaldi-matrix.so
COPY --from=builder /home/kaldi/src/lib/libkaldi-base.so /home/kaldi/src/lib/libkaldi-base.so
COPY --from=builder /home/kaldi/src/lib/libkaldi-nnet3.so /home/kaldi/src/lib/libkaldi-nnet3.so
COPY --from=builder /home/kaldi/src/lib/libkaldi-online2.so /home/kaldi/src/lib/libkaldi-online2.so
COPY --from=builder /home/kaldi/src/lib/libkaldi-cudamatrix.so /home/kaldi/src/lib/libkaldi-cudamatrix.so
COPY --from=builder /home/kaldi/src/lib/libkaldi-ivector.so /home/kaldi/src/lib/libkaldi-ivector.so
COPY --from=builder /home/kaldi/tools/openfst/lib/libfst.so.10 /home/kaldi/tools/openfst/lib/libfst.so.10
COPY --from=builder /usr/local/lib/libgrpc.so.7 /usr/local/lib/libgrpc.so.7
COPY --from=builder /usr/local/lib/libgpr.so.7 /usr/local/lib/libgpr.so.7
COPY --from=builder /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /usr/local/lib/libstdc++.so.6
COPY --from=builder /home/kaldi/src/lib/libkaldi-lat.so /home/kaldi/src/lib/libkaldi-lat.so
COPY --from=builder /home/kaldi/src/lib/libkaldi-tree.so /home/kaldi/src/lib/libkaldi-tree.so
COPY --from=builder /home/kaldi/src/lib/libkaldi-transform.so /home/kaldi/src/lib/libkaldi-transform.so
COPY --from=builder /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.so /opt/intel/mkl/lib/intel64/libmkl_intel_lp64.so
COPY --from=builder /opt/intel/mkl/lib/intel64/libmkl_core.so /opt/intel/mkl/lib/intel64/libmkl_core.so
COPY --from=builder /opt/intel/mkl/lib/intel64/libmkl_sequential.so /opt/intel/mkl/lib/intel64/libmkl_sequential.so
COPY --from=builder /home/kaldi/src/lib/libkaldi-chain.so /home/kaldi/src/lib/libkaldi-chain.so
COPY --from=builder /home/kaldi/src/lib/libkaldi-nnet2.so /home/kaldi/src/lib/libkaldi-nnet2.so
COPY --from=builder /home/kaldi/src/lib/libkaldi-gmm.so /home/kaldi/src/lib/libkaldi-gmm.so

CMD [ "./build/kaldi_serve_app" ]
