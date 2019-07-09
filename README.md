# Kaldi-Serve

[gRPC](https://grpc.io/) server component for [Kaldi](https://kaldi-asr.org/)
based ASR.

**Key Features**:

- Multithreaded gRPC server.
- Supports streaming recognition.
- Thread-safe concurrent queue to process each audio stream separately.

## Getting Started

### Setup

Make sure you have gRPC, protobuf installed on your system. Kaldi also needs to
be present and built. Let's build the server:

```bash
make KALDI_ROOT=/path/to/local/repo/for/kaldi/ -j8
```

Run `make clean` to clear old build files.

### Running the Server

```bash
# Make sure to have kaldi and openfst library available using LD_LIBRARY_PATH or something
# e.g. env LD_LIBRARY_PATH=../../asr/kaldi/tools/openfst/lib/:../../asr/kaldi/src/lib/ ./build/kaldi_serve_app

# Alternatively, you can also put all the required .so files in the ./lib/ directory since
# that is added to the binary's rpath.

$ ./build/kaldi_serve_app --help

Kaldi gRPC server
Usage: ./build/kaldi_serve_app [OPTIONS]

Options:
  -h,--help                   Print this help message and exit
  -m,--model-dir TEXT:DIR REQUIRED
                              Model root directory. This is a temporary API for testing.
  -n,--num-decoders INT:NUMBER
                              Number of decoders to initialize in the concurrent queue.
```

### Python

Python client for the server is present in [./python](./python) directory.

### Running Tests

We use [Catch2](https://github.com/catchorg/Catch2) for unit testing.

```bash
make check -j8
```
