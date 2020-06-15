# Python Binding

Python binding for `kaldiserve` C++ library.

## Build

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