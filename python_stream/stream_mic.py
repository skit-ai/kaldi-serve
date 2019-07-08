"""
Basic microphone based ASR server tester

Usage:
  stream_mic.py [--secs=<secs>]

Options:
  --secs=<secs>      Number of seconds to records, ideally there should be a VAD here. [default: 3]
"""

import io
import wave

import pyaudio
from docopt import docopt

from kaldi import KaldiClient, RecognitionAudio, RecognitionConfig

SR = 8000
CHUNK_SIZE = 4000
CHANNELS = 1


def chunk_generator(secs: int):
    """
    Generate wave audio chunks worth `secs` seconds.
    """

    p = pyaudio.PyAudio()
    sample_format = pyaudio.paInt16

    def _make_wave_chunk(data: bytes) -> bytes:
        """
        Convert this chunk to wave chunk (with the initial 44 bytes header). The
        right way probably is to not send headers at all and let the server side's
        chunk handler maintain state, taking data from metadata.
        """

        out = io.BytesIO()
        wf = wave.open(out, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(SR)
        wf.writeframes(data)
        wf.close()
        return out.getvalue()

    stream = p.open(format=sample_format,
                    channels=CHANNELS,
                    rate=SR,
                    frames_per_buffer=CHUNK_SIZE,
                    input=True)

    for i in range(0, int(SR / CHUNK_SIZE * secs)):
        yield _make_wave_chunk(stream.read(CHUNK_SIZE))

    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    args = docopt(__doc__)

    client = KaldiClient()
    audio = (RecognitionAudio(content=chunk) for chunk in chunk_generator(int(args["--secs"])))

    config = RecognitionConfig(
        sample_rate_hertz=SR,
        encoding=RecognitionConfig.AudioEncoding.LINEAR16,
        language_code="hi",
        max_alternatives=10,
        model=None
    )

    response = client.recognize(config, audio, uuid="")

    for result in response.results:
        for alt in result.alternatives:
            print(alt.transcript, alt.confidence)
