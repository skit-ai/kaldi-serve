# Kaldi-Serve Python Binding

Python binding for the `kaldiserve` C++ library.

## Installation

### Build from source

You will need [pybind11](https://github.com/pybind/pybind11) to be present and built, or alternately you can pass the `-DBUILD_PYBIND11=ON` flag and cmake will take care of it. You can build the bindings by passing `-DBUILD_PYTHON_MODULE=ON -DPYTHON_EXECUTABLE=${which python}` options to the main cmake command:

```bash
# build the python bindings (starting from current dir)
cd ../build
cmake .. -DBUILD_PYBIND11=ON -DBUILD_PYTHON_MODULE=ON -DPYTHON_EXECUTABLE=${which python}
make -j${nproc}

# copy over the built shared library to the python package
cp python/kaldiserve_pybind*.so ../python/kaldiserve/

# build the python package
cd ../build/python
pip install . -U
```

Now you can import `kaldiserve` into your python project.

### Docker Image

#### Using pre-built images

You can pull pre-built docker images (we currently support python version 3.6) from our [Docker Hub repository](https://hub.docker.com/repository/docker/vernacularai/kaldi-serve):

```bash
docker pull vernacularai/kaldi-serve:latest-py3.6
docker run -it vernacularai/kaldi-serve:latest-py3.6
```

You will find Python 3.6 pre-installed with `kaldiserve` python package.

#### Building the image

You can also build the docker image using the [Dockerfile](./Dockerfile) provided (make sure to do it from the root dir):

```bash
cd ../
docker build -t kaldi-serve:py${PY_VERSION} -f python/Dockerfile .
```

## Getting Started

### Transcription Interface

```python
from io import BytesIO
from kaldiserve import ChainModel, Decoder, parse_model_specs, start_decoding

# model specification
model_spec = parse_model_specs("../resources/model-spec.toml")[0]
# chain model contains all large const components that can be shared across decoders on multiple threads
model = ChainModel(model_spec)

# initialize a decoder that keeps a const reference to the model
decoder = Decoder(model)

audio_files = ["sample1.wav", "sample2.wav"]

for audio_file in audio_files:
    # read audio file as bytes
    with open(audio_file, "rb") as f:
        audio_bytes = BytesIO(f.read()).getvalue()

    with start_decoding(decoder):
        # decode the audio
        decoder.decode_wav_audio(audio_bytes)
        # get the transcripts (10 alternatives)
        alts = decoder.get_decoded_results(10)
    
    print(alts)
```

### Sample Scripts

You will need `kaldiserve` python package and some other [dependencies](./scripts/requirements.txt) to be installed:

```bash
pip install -r scripts/requirements.txt
```

There are some sample [scripts](./scripts) provided that can be referenced as examples:
1. [Transcribe](./scripts/transcribe.py) - transcribes a single audio file
2. [Batch Transcribe](./scripts/batch_transcribe.py) - transcribes a batch of audio files via multi-threading

## Known Issues

1. If you face `INTEL MKL ERROR` when instantiating `ChainModel|DecoderFactory|DecoderQueue`, try the following:

```bash
export LD_PRELOAD="${MKL_ROOT}/lib/intel64/libmkl_rt.so"
```

2. If you face `*** Error in python: double free or corruption (!prev): ...`, try the following:

```bash
apt-get install libtcmalloc-minimal4
export LD_PRELOAD="/usr/lib/libtcmalloc_minimal.so.4"
```