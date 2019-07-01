import os
import grpc

from .kaldi_serve_pb2 import RecognizeRequest
from .kaldi_serve_pb2_grpc import KaldiServeStub

# Reference: https://github.com/googleapis/google-cloud-python/blob/3ba1ae73070769854a1f7371305c13752c0374ba/speech/google/cloud/speech_v1/gapic/speech_client.py

def audio_stream_gen(audio_chunks, config, uuid):
    for chunk in audio_chunks:
        req = RecognizeRequest(config=config, audio=chunk, uuid=uuid)
        yield req

class KaldiClient(object):
    """Service that implements Kaldi API."""

    def __init__(
        self,
        transport=None,
        channel=None,
        client_info=None,
    ):
        KALDI_SERVE_HOST = os.environ.get('KALDI_SERVE_HOST', "grpc://0.0.0.0:5017")
        if KALDI_SERVE_HOST.startswith("grpc://"):
            KALDI_SERVE_HOST = KALDI_SERVE_HOST[len("grpc://"):]
        print(KALDI_SERVE_HOST)
        self._channel = grpc.insecure_channel(KALDI_SERVE_HOST)
        self._client = KaldiServeStub(self._channel)

    def recognize(
        self,
        config,
        audio_chunks,
        uuid,
        timeout=None,
        retry=None,
    ):
        request_gen = audio_stream_gen(audio_chunks, config, uuid)
        return self._client.Recognize(request_gen, timeout=timeout)
