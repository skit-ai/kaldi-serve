import grpc

from kaldi_serve.kaldi_serve_pb2 import RecognitionConfig, RecognizeRequest
from kaldi_serve.kaldi_serve_pb2_grpc import KaldiServeStub


class KaldiServeClient(object):
    """
    Service that implements Kaldi API.

    Reference: https://github.com/googleapis/google-cloud-python/blob/3ba1ae73070769854a1f7371305c13752c0374ba/speech/google/cloud/speech_v1/gapic/speech_client.py
    """

    def __init__(self, kaldi_serve_url="0.0.0.0:5016"):
        self.channel = grpc.insecure_channel(kaldi_serve_url)
        self._client = KaldiServeStub(self.channel)

    def recognize(self, config: RecognitionConfig, audio, uuid: str, timeout=None):
        request = RecognizeRequest(config=config, audio=audio, uuid=uuid)
        return self._client.Recognize(request, timeout=timeout)

    def streaming_recognize(self, config: RecognitionConfig, audio_chunks_gen, uuid: str, timeout=None):
        request_gen = (RecognizeRequest(config=config, audio=chunk, uuid=uuid) for chunk in audio_chunks_gen)
        return self._client.StreamingRecognize(request_gen, timeout=timeout)

    def streaming_recognize_raw(self, audio_params_gen, uuid: str, timeout=None):
        request_gen = (RecognizeRequest(config=config, audio=chunk, uuid=uuid) for config, chunk in audio_params_gen)
        return self._client.StreamingRecognize(request_gen, timeout=timeout)

    def bidi_streaming_recognize(self, config: RecognitionConfig, audio_chunks_gen, uuid: str, timeout=None):
        request_gen = (RecognizeRequest(config=config, audio=chunk, uuid=uuid) for chunk in audio_chunks_gen)
        return self._client.BidiStreamingRecognize(request_gen, timeout=timeout)

    def bidi_streaming_recognize_raw(self, audio_params_gen, uuid: str, timeout=None):
        request_gen = (RecognizeRequest(config=config, audio=chunk, uuid=uuid) for config, chunk in audio_params_gen)
        return self._client.BidiStreamingRecognize(request_gen, timeout=timeout)