## Kaldi-Serve Python Binding


### Known Issues

If you face an `MKL_ERROR` when initializing `kaldiserve.ChainModel`, try doing the following:

```bash
export LD_PRELOAD=${MKL_ROOT}/mkl/lib/intel64/libmkl_rt.so
```