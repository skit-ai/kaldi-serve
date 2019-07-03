# Kaldi-Serve

[gRPC](https://grpc.io/) Synchronous Streaming Server component for [Kaldi](https://kaldi-asr.org/) based ASR.

**Key Features**:

- Multithreaded gRPC server (default)
- Supports audio streaming i.e. decoding happens in the backgroud while recording
- Thread-safe concurrent queue to process each audio stream separately

## Getting Started

### Setup

Make sure you have gRPC, protobuf and kaldi installed on your system. Let's build the server:

```bash
make -j8
```

Building again with a clean directory structure:

```bash
./build.sh
```

### Running the Server

```bash
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

### Development

During development, you can test the gRPC C++ server using a gRPC python client that supports streaming as well:

```bash
cd python_stream/
python stream_client.py
```

In case there has been some change in the `.proto` file, run the following command beforehand:

```bash
pip install grpcio-tools # grpcio utility tools needed for next step

cd python_stream/kaldi/
python -m grpc_tools.protoc -I../../protos --python_out=. \
    --grpc_python_out=. ../../protos/kaldi_serve.proto
```

### Running Tests

We use [Catch2](https://github.com/catchorg/Catch2) for unit testing.

```bash
make check -j8
```
