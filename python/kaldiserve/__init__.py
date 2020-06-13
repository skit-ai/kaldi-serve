from kaldiserve.kaldiserve_pybind import ModelSpec, Word, Alternative                       # types
from kaldiserve.kaldiserve_pybind import _ModelSpecList, _WordList, _AlternativeList        # type list aliases
from kaldiserve.kaldiserve_pybind import ChainModel                                         # models
from kaldiserve.kaldiserve_pybind import Decoder, DecoderQueue, DecoderFactory              # decoders
from kaldiserve.kaldiserve_pybind import parse_model_specs                                  # utils

from contextlib import contextmanager


@contextmanager
def acquire_decoder(dq: DecoderQueue):
    decoder = dq.acquire()
    try:
        yield decoder
    finally:
        dq.release(decoder)