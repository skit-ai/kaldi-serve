# Kaldi-Serve

![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/Vernacular-ai/kaldi-serve?style=flat-square) ![GitHub](https://img.shields.io/github/license/Vernacular-ai/kaldi-serve?style=flat-square)

A plug-and-play abstraction over [Kaldi](https://kaldi-asr.org/) ASR toolkit, designed for ease of deployment and optimal runtime performance.

**Key Features**:

- Real-time streaming (uni & bi-directional) audio recognition.
- Thread-safe concurrent Decoder queue for server environments.
- RNNLM lattice rescoring.
- N-best alternatives with AM/LM costs, word-level timings and confidence scores.
- Easy extensibility for custom applications.

## Installation

### Dependencies

Make sure you have the following dependencies installed on your system before beginning the build process:

* g++ compiler (>=4.7) that supports C++11 std
* [CMake](https://cmake.org/install/) (>=3.13)
* [Kaldi](https://kaldi-asr.org/)
* [Boost C++](https://www.boost.org/) libraries

### Build from Source

Let's build the shared library:

```bash
cd build/
cmake ..
make -j${nproc}
```

You will find the the built shared library in `build/src/` to use for linking against custom applications.

#### Python bindings

We also provide python bindings for the library. You can find the build instructions [here](./python).

### Docker Image

#### Using pre-built images

You can also pull a pre-built docker image from our [Docker Hub repository](https://hub.docker.com/repository/docker/vernacularai/kaldi-serve):

```bash
docker pull vernacularai/kaldi-serve:latest
docker run -it -v /path/to/my/app:/home/app vernacularai/kaldi-serve:latest
```

You will find our headers in `/usr/include/kaldiserve` and the shared library `libkaldiserve.so` in `/usr/local/lib`.

#### Building the image

You can build the docker image using the [Dockerfile](./Dockerfile) provided.

```bash
docker build -t kaldi-serve:lib .
```

## Getting Started

<!-- [**Documentation**]()

Please check out the docs for a reference of how to use the library. -->

### Usage

You can include the [headers](./include) and link the shared library you get after the build process, against your application and start using it.

### Plugins

It's also worth noting that there are a few [plugins](./plugins) we actively maintain and will keep adding to, that use the library:
- [gRPC Server](./plugins/grpc)

## License

This project is licensed under the Apache License version 2.0. Please see [LICENSE](./LICENSE) for more details.