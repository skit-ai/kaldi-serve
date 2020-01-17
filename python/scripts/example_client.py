"""
Script for testing out ASR server.

Usage:
  example_client.py mic [--n-secs=<n-secs>] [--model=<model>] [--lang=<lang>] [--raw] [--pcm] [--word-level]
  example_client.py <file>... [--model=<model>] [--lang=<lang>] [--raw] [--pcm] [--word-level]

Options:
  --n-secs=<n-secs>     Number of seconds to records, ideally there should be a VAD here. [default: 3]
  --model=<model>       Name of the model to hit [default: general]
  --lang=<lang>         Language code of the model [default: hi]
  --raw                 Flag that specifies whether to stream raw audio bytes to server.
  --pcm                 Flag for sending raw pcm bytes
  --word-level          Whether to get word level features from server.
"""

import random
import threading
import traceback
from pprint import pprint
from typing import List

from docopt import docopt
from pydub import AudioSegment

from kaldi_serve import KaldiServeClient, RecognitionAudio, RecognitionConfig
from kaldi_serve.utils import (chunks_from_file, chunks_from_mic,
                               raw_bytes_to_wav)

SR = 8000
CHANNELS = 1


def parse_response(response):
    output = []

    for res in response.results:
        output.append([
            {
                "transcript": alt.transcript,
                "confidence": alt.confidence,
                "am_score": alt.am_score,
                "lm_score": alt.lm_score,
                "words": [
                    {
                        "start_time": word.start_time,
                        "end_time": word.end_time,
                        "word": word.word,
                        "confidence": word.confidence
                    }
                    for word in alt.words
                ]
            }
            for alt in res.alternatives
        ])
    return output


def transcribe_chunks_streaming(client, audio_chunks, model: str, language_code: str, raw: bool=False, word_level: bool=False):
    """
    Transcribe the given audio chunks
    """

    response = {}
    encoding = RecognitionConfig.AudioEncoding.LINEAR16

    try:
        if raw:
            config = lambda chunk_len: RecognitionConfig(
                sample_rate_hertz=SR,
                encoding=encoding,
                language_code=language_code,
                max_alternatives=10,
                model=model,
                raw=True,
                word_level=word_level,
                data_bytes=chunk_len
            )
            audio_params = [(config(len(chunk)), RecognitionAudio(content=chunk)) for chunk in audio_chunks]
            response = client.streaming_recognize_raw(audio_params, uuid="")
        else:
            audio = (RecognitionAudio(content=chunk) for chunk in audio_chunks)
            config = RecognitionConfig(
                sample_rate_hertz=SR,
                encoding=encoding,
                language_code=language_code,
                max_alternatives=10,
                model=model,
                word_level=word_level
            )
            response = client.streaming_recognize(config, audio, uuid="")
    except Exception as e:
        traceback.print_exc()
        print(f'error: {str(e)}')

    pprint(parse_response(response))

def transcribe_chunks_bidi_streaming(client, audio_chunks, model: str, language_code: str, word_level: bool=False):
    """
    Transcribe the given audio chunks
    """

    response = {}
    encoding = RecognitionConfig.AudioEncoding.LINEAR16

    try:
        config = lambda chunk_len: RecognitionConfig(
            sample_rate_hertz=SR,
            encoding=encoding,
            language_code=language_code,
            max_alternatives=10,
            model=model,
            raw=True,
            word_level=word_level,
            data_bytes=chunk_len
        )

        def audio_params_gen(audio_chunks_gen):
            for chunk in audio_chunks_gen:
                yield config(len(chunk)), RecognitionAudio(content=chunk)

        response_gen = client.bidi_streaming_recognize_raw(audio_params_gen(audio_chunks), uuid="")
    except Exception as e:
        traceback.print_exc()
        print(f'error: {str(e)}')

    for response in response_gen:
        pprint(parse_response(response))


def decode_files(client, audio_paths: List[str], model: str, language_code: str, raw: bool=False, pcm: bool=False, word_level: bool=False):
    """
    Decode files using threaded requests
    """
    chunked_audios = [chunks_from_file(x, chunk_size=random.randint(1, 3), raw=raw, pcm=pcm) for x in audio_paths]

    threads = [
        threading.Thread(target=transcribe_chunks_streaming, args=(client, chunks, model, language_code, raw, word_level))
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
    raw = args['--raw']
    pcm = args['--pcm']
    word_level = args["--word-level"]

    if args["mic"]:
        transcribe_chunks_bidi_streaming(client, chunks_from_mic(int(args["--n-secs"]), SR, 1), model, language_code, word_level)
    else:
        decode_files(client, args["<file>"], model, language_code, raw, pcm, word_level)
