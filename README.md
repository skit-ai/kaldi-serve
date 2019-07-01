# Kaldi-Serve

[gRPC](https://grpc.io/) Synchronous Streaming Server component for [Kaldi](https://kaldi-asr.org/) based ASR.

## Getting Started

### Setup

Make sure you have gRPC, protobuf and kaldi installed on your system. Let's build the server:

```bash
make -j8
```

### Running the Server

```bash
./build/kaldi_serve_app [MODEL PATH]
```

### Running Tests

```bash
cd test/
make -j8
./test_server.sh
```
