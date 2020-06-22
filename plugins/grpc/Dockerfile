# Stage 1: dev/build
FROM vernacularai/kaldi-serve:latest as builder

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
RUN cd /root/ && \
    git clone -b v1.28.1 https://github.com/grpc/grpc && \
    cd /root/grpc/ && \
    git submodule update --init && \
    make -j$(nproc) && \
    make install

# Install Protobuf v3
RUN cd /root/grpc/third_party/protobuf && make install

WORKDIR /root/kaldi-serve
COPY . .

WORKDIR /root/kaldi-serve/plugins/grpc
ENV LD_LIBRARY_PATH="/opt/kaldi/tools/openfst/lib:/opt/kaldi/src/lib"
RUN make clean && \
    make KALDI_ROOT="/opt/kaldi" KALDISERVE_INCLUDE="/usr/include" -j$(nproc)

RUN bash -c "mkdir /so-files/; cp /opt/intel/mkl/lib/intel64/lib*.so /so-files/"

# Stage 2: prod
FROM debian:jessie-slim
WORKDIR /home/app

COPY --from=builder /root/kaldi-serve/plugins/grpc/build/kaldi_serve_app .

# LIBS
COPY --from=builder /usr/lib/x86_64-linux-gnu/libssl.so* /usr/local/lib/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libcrypto.so* /usr/local/lib/

# CPP LIBS
COPY --from=builder /usr/lib/x86_64-linux-gnu/libstdc++.so* /usr/local/lib/

# BOOST LIBS
COPY --from=builder /usr/lib/x86_64-linux-gnu/libboost_system.so* /usr/local/lib/
COPY --from=builder /usr/lib/x86_64-linux-gnu/libboost_filesystem.so* /usr/local/lib/

# GRPC LIBS
COPY --from=builder /usr/local/lib/libgrpc++.so* /usr/local/lib/
COPY --from=builder /usr/local/lib/libgrpc++_reflection.so* /usr/local/lib/
COPY --from=builder /usr/local/lib/libgrpc.so* /usr/local/lib/
COPY --from=builder /usr/local/lib/libgpr.so* /usr/local/lib/
COPY --from=builder /usr/local/lib/libupb.so* /usr/local/lib/

# INTEL MKL
COPY --from=builder /so-files /opt/intel/mkl/lib/intel64

# KALDI LIBS
COPY --from=builder /opt/kaldi/tools/openfst/lib/libfst.so.10 /opt/kaldi/tools/openfst/lib/
COPY --from=builder /opt/kaldi/src/lib/libkaldi-*.so /opt/kaldi/src/lib/

# KALDISERVE LIB
COPY --from=builder /usr/local/lib/libkaldiserve.so* /usr/local/lib/

ENV LD_LIBRARY_PATH="/usr/local/lib:/opt/kaldi/tools/openfst/lib:/opt/kaldi/src/lib"

CMD [ "./kaldi_serve_app" ]
