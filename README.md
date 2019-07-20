# Kaldi-Serve

[gRPC](https://grpc.io/) server component for [Kaldi](https://kaldi-asr.org/)
based ASR.

**Key Features**:

- Multithreaded gRPC server.
- Supports streaming recognition.
- Thread-safe concurrent queue to process each audio stream separately.

## Getting Started

### Setup

Make sure you have gRPC, protobuf and Boost C++ libraries installed on your system. Kaldi also needs to
be present and built. Let's build the server:

```bash
~$ make KALDI_ROOT=/path/to/local/repo/for/kaldi/ -j8
```

Run `make clean` to clear old build files.

### Running the gRPC C++ Server

For running the server, you need to first write model config in a toml which
tells the program which models to load and where to look for. Structure of
`model_spec_toml` file is specified in a sample in
[resources](./resources/model-spec.toml).

```bash
# Make sure to have kaldi and openfst library available using LD_LIBRARY_PATH or something
# e.g. env LD_LIBRARY_PATH=../../asr/kaldi/tools/openfst/lib/:../../asr/kaldi/src/lib/ ./build/kaldi_serve_app

# Alternatively, you can also put all the required .so files in the ./lib/ directory since
# that is added to the binary's rpath.

~$ ./build/kaldi_serve_app --help

Kaldi gRPC server
Usage: ./build/kaldi_serve_app [OPTIONS] model_spec_toml

Positionals:
  model_spec_toml TEXT:FILE REQUIRED
                              Path to toml specifying models to load.

Options:
  -h,--help                   Print this help message and exit
```

### Python Client

Python client for the server is present in [./python](./python) directory.

### Load Testing

We perform load testing using [ghz](https://ghz.sh/) which is a gRPC benchmarking and load testing tool. You will need to download it's binary into your PATH and load test the application using the following command template:

```bash
~$ ghz \
--insecure \
--proto ./protos/kaldi_serve.proto \
--call kaldi_serve.KaldiServe.StreamingRecognize \
-n [NUM REQUESTS] -c [CONCURRENT REQUESTS] \
--cpus [NUM CORES] \
-d "[{\"audio\": {\"content\": \"$chunk1\"}, \"config\": {\"max_alternatives\": [N_BEST], \"language_code\": \"[LANGUUAGE]\", \"model\": \"[MODEL]\"}}, ...more chunks]" \
0.0.0.0:5016
```
