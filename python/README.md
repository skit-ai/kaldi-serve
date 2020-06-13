## Kaldi-Serve Python Binding


### Known Issues

1. If you face `INTEL MKL ERROR` when instantiating `ChainModel|DecoderFactory|DecoderQueue`, try the following:

```bash
export LD_PRELOAD="${MKL_ROOT}/lib/intel64/libmkl_rt.so"
```

2. If you face `*** Error in python: double free or corruption (!prev): ...`, try the following:

```bash
apt-get install libtcmalloc-minimal4
export LD_PRELOAD="/usr/lib/libtcmalloc_minimal.so.4"
```