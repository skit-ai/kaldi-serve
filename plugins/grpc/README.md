# Kaldi-Serve gRPC Plugin

[gRPC](https://grpc.io/) server component for [Kaldi](https://kaldi-asr.org/)
based ASR.

## Getting Started

### Setup

Make sure you have built the kaldiserve library. You will also need Kaldi, gRPC, protobuf and Boost C++ libraries installed on your system. Let's build the server:

```bash
make KALDI_ROOT=/path/to/local/repo/for/kaldi/ -j${nproc}
```

Run `make clean` to clear old build files.

### Running the server

For running the server, you need to first specify model config in a toml which
tells the program which models to load, where to look for etc. Structure of
`model_spec_toml` file is specified in a sample in [resources](../../resources/model-spec.toml).

```bash
# Make sure to have kaldi and openfst library available using LD_LIBRARY_PATH or something
# e.g. env LD_LIBRARY_PATH=../../asr/kaldi/tools/openfst/lib/:../../asr/kaldi/src/lib/ ./build/kaldi_serve_app

# Alternatively, you can also put all the required .so files in the ./lib/ directory since
# that is added to the binary's rpath.

./build/kaldi_serve_app --help

Kaldi gRPC server
Usage: ./build/kaldi_serve_app [OPTIONS] model_spec_toml

Positionals:
  model_spec_toml TEXT:FILE REQUIRED
                              Path to toml specifying models to load

Options:
  -h,--help                   Print this help message and exit
  -v,--version                Show program version and exit
```

### Clients

For simple microphone testing, you can do something like the following (needs
[evans](https://github.com/ktr0731/evans) installed):

```bash
audio_bytes=$(arecord -f S16_LE -d 5 -r 8000 -c 1 | base64 -w0) # Recording 5 seconds of audio
echo "{\"audio\": {\"content\": \"$audio_bytes\"}, \"config\": {\"max_alternatives\": 2, \"model\": \"general\", \"language_code\": \"hi\"} }" | evans --package kaldi_serve --service KaldiServe ./protos/kaldi_serve.proto  --call Recognize --port 5016 | jq
```

The output structure looks like the following:
```
{
  "results": [
    {
      "alternatives": [
        {
          "transcript": "हेलो दुनिया",
          "confidence": 0.95897794,
          "amScore": -374.5963,
          "lmScore": 131.33058
        },
        {
          "transcript": "हैलो दुनिया",
          "confidence": 0.95882875,
          "amScore": -372.76187,
          "lmScore": 131.84035
        }
      ]
    }
  ]
}
```

A Python client is also present in [client](./client) directory with a few
example scripts.

### Load testing

We perform load testing using [ghz](https://ghz.sh/) which is a gRPC
benchmarking and load testing tool. You can use the following command template:

```bash
ghz \
--insecure \
--proto ./protos/kaldi_serve.proto \
--call kaldi_serve.KaldiServe.StreamingRecognize \
-n [NUM REQUESTS] -c [CONCURRENT REQUESTS] \
--cpus [NUM CORES] \
-d "[{\"audio\": {\"content\": \"$chunk1\"}, \"config\": {\"max_alternatives\": [N_BEST], \"language_code\": \"[LANGUUAGE]\", \"model\": \"[MODEL]\"}}, ...more chunks]" \
0.0.0.0:5016
```
