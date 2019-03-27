import os
from distutils.core import setup, Extension

KALDI_ROOT = os.environ.get('KALDI_ROOT', '')

tdnn_decode = Extension(
   'tdnn_decode',
   language = "c++",
   include_dirs = [
      f'{KALDI_ROOT}/src/', f'{KALDI_ROOT}/tools/openfst/include/'
   ],
   library_dirs = [f'{KALDI_ROOT}/src/lib', f'{KALDI_ROOT}/tools/openfst/lib'],
   sources=['tdnn_decode.cc', 'tdnn_wrapper.cc'],
   extra_compile_args = ['-std=c++11', '-DKALDI_DOUBLEPRECISION=0', '-Wno-sign-compare',
      '-Wno-unused-local-typedefs', '-Wno-unused-variable', '-Winit-self'
   ],
   extra_link_args = ['-rdynamic', '-lm', '-lpthread', '-ldl',
      '-lkaldi-decoder', '-lkaldi-lat', '-lkaldi-fstext', '-lkaldi-hmm', '-lkaldi-feat',
      '-lkaldi-transform', '-lkaldi-gmm', '-lkaldi-tree', '-lkaldi-util', '-lkaldi-matrix',
      '-lkaldi-base', '-lkaldi-nnet3', '-lkaldi-online2', '-lkaldi-cudamatrix',
      '-lkaldi-ivector', '-lfst'
   ],
)

setup(
   name = 'kaldi_extension',
   version = '1.0',
   ext_modules = [tdnn_decode]
)