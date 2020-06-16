# Kaldi-Serve

![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/Vernacular-ai/kaldi-serve?style=flat-square) ![GitHub](https://img.shields.io/github/license/Vernacular-ai/kaldi-serve?style=flat-square)

A plug-and-play abstraction over [Kaldi](https://kaldi-asr.org/) ASR toolkit, designed for ease of deployment and optimal runtime performance.

**Key Features**:

- Real-time streaming (uni & bi-directional) recognition.
- Batched audio recognition on GPU.
- Thread-safe concurrent Decoder queue for server environments.
- RNNLM lattice rescoring.
- N-best alternatives with AM/LM costs, word-level timings and confidence scores.
- Easily extensible for custom applications.

## Installation

### Build from Source

Make sure you have the following dependencies installed on your system before beginning the build process:

* g++ compiler (>=4.7) that supports C++11 std
* [CMake](https://cmake.org/install/) (>=3.12)
* [Kaldi](https://kaldi-asr.org/)
* [Boost C++](https://www.boost.org/) libraries

Let's build the shared library:

```bash
cd build/
cmake ..
make -j${nproc}
```

You will find the the built `.so` file in `build/src/` to use for linking against custom applications.

#### Python bindings

We also provide python bindings for the library. You can find the build instructions [here](./python).

<!-- ### Docker Image

#### Using pre-built images

You can also pull a pre-built docker image from Docker Hub and run with docker (for CPU runtime):

```bash
docker run -it vernacularai/kaldi-serve:master
```

For GPU runtime, you will need nvidia-docker installed:

```bash
docker run --gpus all -it vernacularai/kaldi-serve:gpu-latest
```

#### Building the image

You can build the docker image using the [Dockerfile](./Dockerfile) provided.

```bash
docker build -t kaldi-serve:${DOCKER_TAG} .
``` -->

## Getting Started

<!-- [**Documentation**]()

Please check out the docs for a reference of how to use the library. -->

It's also worth noting that there are a few [plugins](./plugins) and [binaries](./bin) we maintain that use the library and can be referenced as examples:
- gRPC Server
- Batched decoding binaries (CPU & GPU)

So, you can include the [headers](./include) and link the shared [library](./src) aginst your application and start using it.

## License

This project is licensed under the Apache License version 2.0. Please see [LICENSE](./LICENSE) for more details.