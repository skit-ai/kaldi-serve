FROM gcr.io/vernacular-voice-services/asr/kaldi:latest as builder

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

FROM alpine:3.10.1
RUN apk --no-cache add ca-certificates
WORKDIR /home/app

COPY --from=builder /home/app/build build
COPY --from=builder /home/app/resources resources

ENV LD_LIBRARY_PATH="/usr/local/lib:/home/kaldi/tools/openfst/lib:/home/kaldi/src/lib"

COPY --from=builder /usr/local/lib/libgrpc++.so.1 /usr/local/lib/libgrpc++.so.1
COPY --from=builder /usr/local/lib/libgrpc++_reflection.so.1 /usr/local/lib/libgrpc++_reflection.so.1
COPY --from=builder /usr/local/lib/libgrpc++_reflection.so.1 /usr/local/lib/libgrpc++_reflection.so.1
COPY --from=builder /lib/x86_64-linux-gnu/libpthread.so.0 /lib/x86_64-linux-gnu/libpthread.so.0
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
COPY --from=builder /lib/x86_64-linux-gnu/libm.so.6 /lib/x86_64-linux-gnu/libm.so.6
COPY --from=builder /lib/x86_64-linux-gnu/libgcc_s.so.1 /lib/x86_64-linux-gnu/libgcc_s.so.1
COPY --from=builder /lib/x86_64-linux-gnu/libc.so.6 /lib/x86_64-linux-gnu/libc.so.6
COPY --from=builder /lib/x86_64-linux-gnu/libdl.so.2 /lib/x86_64-linux-gnu/libdl.so.2
COPY --from=builder /lib/x86_64-linux-gnu/librt.so.1 /lib/x86_64-linux-gnu/librt.so.1
COPY --from=builder /lib/x86_64-linux-gnu/libz.so.1 /lib/x86_64-linux-gnu/libz.so.1
COPY --from=builder /usr/local/lib/libgrpc.so.7 /usr/local/lib/libgrpc.so.7
COPY --from=builder /usr/local/lib/libgpr.so.7 /usr/local/lib/libgpr.so.7
COPY --from=builder /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /usr/lib/x86_64-linux-gnu/libstdc++.so.6
COPY --from=builder /home/kaldi/src/lib/libkaldi-lat.so /home/kaldi/src/lib/libkaldi-lat.so
COPY --from=builder /home/kaldi/src/lib/libkaldi-tree.so /home/kaldi/src/lib/libkaldi-tree.so
COPY --from=builder /home/kaldi/src/lib/libkaldi-transform.so /home/kaldi/src/lib/libkaldi-transform.so
COPY --from=builder /usr/lib/libcblas.so.3 /usr/lib/libcblas.so.3
COPY --from=builder /usr/lib/liblapack_atlas.so.3 /usr/lib/liblapack_atlas.so.3
COPY --from=builder /home/kaldi/src/lib/libkaldi-gmm.so /home/kaldi/src/lib/libkaldi-gmm.so
COPY --from=builder /usr/lib/libatlas.so.3 /usr/lib/libatlas.so.3
COPY --from=builder /usr/lib/x86_64-linux-gnu/libgfortran.so.3 /usr/lib/x86_64-linux-gnu/libgfortran.so.3
COPY --from=builder /usr/lib/libf77blas.so.3 /usr/lib/libf77blas.so.3
COPY --from=builder /usr/lib/x86_64-linux-gnu/libquadmath.so.0 /usr/lib/x86_64-linux-gnu/libquadmath.so.0
COPY --from=builder /lib64/ld-linux-x86-64.so.2 /lib64/ld-linux-x86-64.so.2

CMD [ "./build/kaldi_serve_app" ]
