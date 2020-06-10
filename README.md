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

Make sure you have C++14 std and Boost C++ libraries installed on your
system. [Kaldi](https://kaldi-asr.org/) also needs to be present and built. Let's build the shared library:

```bash
cd build/
cmake .. -DKALDI_ROOT=/path/to/local/repo/for/kaldi/
make -j${nproc}
```

You will find the headers in `include/` and the built `.so` file in `build/src/` to use for linking against custom applications.

#### Python binding

We also provide a [python binding](./python) of the library, which needs pybind11 to be present and built, or alternately you can pass the `-DBUILD_PYBIND11` flag and cmake will take care of it. You can build the bindings by passing `-DBUILD_PYTHON_MODULE` flag to the cmake command [above](###setup).

## Usage

There are a few [plugins](./plugins) and [binaries](./bin) using the library that we maintain:
- gRPC Server
- Batch decoding binary (CPU & GPU)

Alternately, you can include our [headers](./include) and link the [library](./src) aginst your application at compile time and start using it.