import threading
import time

import pytest
import yaml

from kaldi_serve import KaldiServeClient, RecognitionAudio, RecognitionConfig
from kaldi_serve.utils import chunks_from_file


def read_items(file_path: str):
    """
    Read transcription specs for testing
    """

    with open(file_path) as fp:
        return yaml.safe_load(fp)


def pytest_collect_file(parent, path):
    if path.ext == ".yaml" and path.basename.startswith("test"):
        return TranscriptionSpecFile(path, parent)


class TranscriptionSpecFile(pytest.File):
    def collect(self):
        client = KaldiServeClient()
        for i, item in enumerate(read_items(self.fspath)):
            yield TranscriptionItem(f"item-{i}", self, item, client)


def dreamer(source_gen, sleep_time: int):
    for item in source_gen:
        yield item
        time.sleep(sleep_time)


class TranscriptionItem(pytest.Item):
    """
    Each item tells which files to read and throw at the server in parallel.
    Also tells the expected transcriptions for each.
    """

    def __init__(self, name, parent, item, client):
        super().__init__(name, parent)
        self.audios = [
            (chunks_from_file(audio_spec["file"]), audio_spec["transcription"])
            for audio_spec in item
        ]
        self.client = client
        self.results = [None for _ in item]

    def decode_audio(self, index: int):
        # NOTE: These are only assumptions for now so test failures might not
        #       necessarily mean error in model/server.
        config = RecognitionConfig(
            sample_rate_hertz=8000,
            encoding=RecognitionConfig.AudioEncoding.LINEAR16,
            language_code="hi",
            max_alternatives=10,
            model="general"
        )

        audio = dreamer((RecognitionAudio(content=chunk) for chunk in self.audios[index][0]), 1)
        self.results[index] = self.client.streaming_recognize(config, audio, uuid="")

    def runtest(self):
        threads = []
        for i in range(len(self.audios)):
            threads.append(threading.Thread(target=self.decode_audio, args=(i, )))

        for thread in threads:
            thread.start()

        for i, thread in enumerate(threads):
            thread.join()
            assert self.results[i].results[0].alternatives[0].transcript == self.audios[i][1]

    def reportinfo(self):
        return self.fspath, 0, self.name
