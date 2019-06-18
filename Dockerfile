FROM gcr.io/vernacular-voice-services/asr/kaldi:latest

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

RUN mkdir /home/app
WORKDIR /home/app

COPY . /home/app

ENV KALDI_ROOT="/home/kaldi"
ENV LD_LIBRARY_PATH="/home/kaldi/tools/openfst/lib:/home/kaldi/src/lib"

RUN make

ENV REDIS_HOST="localhost"
ENV REDIS_VIRTUAL_PORT=1
ENV MODELS_PATH="/vol/data/models"

CMD [ "./build/kaldi_serve_app" ]