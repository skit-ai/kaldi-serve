"""
Script for testing out ASR server.

Usage:
  example_client.py mic [--n-secs=<n-secs>] [--model=<model>] [--lang=<lang>]
  example_client.py <file>... [--model=<model>] [--lang=<lang>]

Options:
  --n-secs=<n-secs>     Number of seconds to records, ideally there should be a VAD here. [default: 3]
  --model=<model>       Name of the model to hit [default: general]
  --lang=<lang>         Language code of the model [default: hi]
"""

import threading
import traceback
from pprint import pprint
from typing import List

from docopt import docopt

from kaldi_serve import KaldiServeClient, RecognitionAudio, RecognitionConfig
from kaldi_serve.utils import (chunks_from_file, chunks_from_mic,
                               raw_bytes_to_wav)

SR = 8000
CHANNELS = 1


def parse_response(response):
    output = []

    for res in response.results:
        output.append([
            {"transcript": alt.transcript, "confidence": alt.confidence}
            for alt in res.alternatives
        ])
    return output


def transcribe_chunks(client, audio_chunks, model: str, language_code: str):
    """
    Transcribe the given audio chunks
    """

    response = {}
    encoding = RecognitionConfig.AudioEncoding.LINEAR16

    audio = (RecognitionAudio(content=chunk) for chunk in audio_chunks)
    config = RecognitionConfig(
        sample_rate_hertz=SR,
        encoding=encoding,
        language_code=language_code,
        max_alternatives=10,
        model=model,
    )

    try:
        response = client.streaming_recognize(config, audio, uuid="")
    except Exception as e:
        traceback.print_exc()
        print(f'error: {str(e)}')

    pprint(parse_response(response))


def decode_files(client, audio_paths: List[str], model: str, language_code: str):
    """
    Decode files using threaded requests
    """

    chunked_audios = [chunks_from_file(x, chunk_size=1) for x in audio_paths]

    threads = [
        threading.Thread(target=transcribe_chunks, args=(client, chunks, model, language_code))
        for chunks in chunked_audios
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    args = docopt(__doc__)
    client = KaldiServeClient()
    model = args["--model"]
    language_code = args["--lang"]

    if args["mic"]:
        transcribe_chunks(client, chunks_from_mic(int(args["--n-secs"]), SR, 1), model, language_code)
    else:
        decode_files(client, args["<file>"], model, language_code)
