FROM kaldiasr/kaldi:latest

# build latest cmake
WORKDIR /root

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
        libssl-dev \
        cmake

RUN wget https://github.com/Kitware/CMake/releases/download/v3.17.3/cmake-3.17.3.tar.gz && \
    tar -xvf cmake-3.17.3.tar.gz

WORKDIR /root/cmake-3.17.3

# using an older cmake to build a newer cmake (>=3.13)
RUN cmake . && \
    make -j$(nproc) && \
    make install

# install c++ std & boost libs
RUN apt-get update && \
    apt-get install -y \
        g++ \
        make \
        automake \
        libc++-dev \
        libboost-all-dev

WORKDIR /root/kaldi-serve
COPY . .

# build libkaldiserve.so
RUN cd build/ && \
    cmake .. -DBUILD_SHARED_LIBS=ON -DBUILD_PYTHON_MODULE=OFF && \
    make -j$(nproc) VERBOSE=1 && \
    cd /root/kaldi-serve

# KALDISERVE HEADERS & LIB
RUN cp build/src/libkaldiserve.so* /usr/local/lib/
RUN cp -r include/kaldiserve /usr/include/

WORKDIR /root

# cleanup
RUN rm -rf kaldi-serve cmake-*