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

# Remove gRPC source code
RUN rm -rf /home/grpc/

WORKDIR /home/app

COPY . .

ENV KALDI_ROOT="/home/kaldi" \
    LD_LIBRARY_PATH="/home/kaldi/tools/openfst/lib:/home/kaldi/src/lib"

RUN make

FROM ubuntu:bionic
WORKDIR /home/app
COPY --from=builder /home/app/build build

CMD [ "./build/kaldi_serve_app" ]
