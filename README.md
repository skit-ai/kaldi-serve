# Kaldi-Serve

![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/Vernacular-ai/kaldi-serve?style=flat-square) ![GitHub](https://img.shields.io/github/license/Vernacular-ai/kaldi-serve?style=flat-square)

A plug-and-play abstraction over [Kaldi](https://kaldi-asr.org/) ASR toolkit, designed for ease of deployment and optimal runtime performance.

**Key Features**:

- Supports real-time streaming (uni & bi-directional) and batch audio recognition.
- Thread-safe concurrent queue to handle multiple audio streams.
- RNNLM lattice rescoring.
- N-best alternatives with LM and AM costs.
- Word level timings and confidence scores.

## Getting Started

### Setup

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

### Python bindings

We also provide python bindings for the library. You can find the build instructions [here](./python).

## Usage

There are a few [plugins](./plugins) and [binaries](./bin) using the library that we maintain:
- gRPC Server
- Batched decoding binaries (CPU & GPU)

Alternately, you can include our [headers](./include) and link the [library](./src) aginst your application and start using it.