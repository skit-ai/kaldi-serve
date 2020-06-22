# Kaldi-Serve gRPC Plugin

[gRPC](https://grpc.io/) server & client components for [Kaldi](https://kaldi-asr.org/) based ASR.

## Installation

### Dependencies

Make sure you have the following dependencies installed on your system before beginning the build process:

* g++ compiler (>=4.7) that supports C++11 std
* [CMake](https://cmake.org/install/) (>=3.13)
* [Kaldi](https://kaldi-asr.org/)
* [gRPC](https://github.com/grpc/grpc)
* [Boost C++](https://www.boost.org/) libraries

### Build from source

Make sure you have built the kaldiserve shared library and placed in `/usr/local/lib`. Let's build the server application using the kaldiserve library:

```bash
make KALDI_ROOT="/path/to/local/repo/for/kaldi/" -j${nproc}
```

Run `make clean` to clear old build files.

### Docker Image

#### Using pre-built images

You can also pull a pre-built docker image from our [Docker Hub repository](https://hub.docker.com/repository/docker/vernacularai/kaldi-serve):

```bash
docker pull vernacularai/kaldi-serve:latest-grpc
docker run -it -p 5016:5016 -v /models:/home/app/models vernacularai/kaldi-serve:latest-grpc resources/model-spec.toml
```

You will find the built server application `kaldi_serve_app` in `/home/app`.

#### Building the image

You can build the docker image using the [Dockerfile](./Dockerfile) provided.

```bash
cd ../../
docker build -t kaldi-serve:grpc -f plugins/grpc/Dockerfile .
```

You will get a stripped down production ready image from running the above command as we use multi-stage docker builds. In case you need a **development** image, build the image as follows:

```bash
docker build --target builder -t kaldi-serve:grpc-dev -f plugins/grpc/Dockerfile .
```

## Getting Started

### Server

For running the server, you need to first specify model config in a toml which
tells the program which models to load, where to look for etc. Structure of
`model_spec_toml` file is specified in a sample in [resources](../../resources/model-spec.toml).

```bash
./kaldi_serve_app --help

Kaldi gRPC server
Usage: ./kaldi_serve_app [OPTIONS] model_spec_toml

Positionals:
  model_spec_toml TEXT:FILE REQUIRED
                              Path to toml specifying models to load

Options:
  -h,--help                   Print this help message and exit
  -v,--version                Show program version and exit
  -d,--debug                  Enable debug request logging
```

Please also see our [Aspire example](./examples/aspire) on how to get a server up and running with your models.

#### Python Client

A [Python gRPC client](./client) is also provided with a few example scripts (client SDK needs to be installed via [poetry](https://github.com/python-poetry/poetry)). For simple microphone testing, you can do something like the following (make sure the server is running on the same machine on the specified port, default: 5016):

```bash
cd client/
poetry run python scripts/example_client.py mic --n-secs=5 --model=general --lang=hi
```

The output should look something like the following:
```bash
{
  "results": [
    {
      "alternatives": [
        {
          "transcript": "हेलो दुनिया",
          "confidence": 0.95897794,
          "am_score": -374.5963,
          "lm_score": 131.33058
        },
        {
          "transcript": "हैलो दुनिया",
          "confidence": 0.95882875,
          "am_score": -372.76187,
          "lm_score": 131.84035
        }
      ]
    }
  ]
}
```