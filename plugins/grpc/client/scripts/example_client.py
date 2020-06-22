"""
Script for testing out ASR server.

Usage:
  example_client.py mic [--n-secs=<n-secs>] [--model=<model>] [--lang=<lang>] [--sample-rate=<sample-rate>] [--max-alternatives=<max-alternatives>] [--stream] [--raw] [--pcm] [--word-level]
  example_client.py <file>... [--model=<model>] [--lang=<lang>] [--sample-rate=<sample-rate>] [--max-alternatives=<max-alternatives>] [--stream] [--raw] [--pcm] [--word-level]

Options:
  --n-secs=<n-secs>                         Number of seconds to record the audio for before making a request. [default: 5]
  --model=<model>                           Name of the model to hit. [default: general]
  --lang=<lang>                             Language code of the model. [default: en]
  --sample-rate=<sample-rate>               Sampling rate to use for the audio. [default: 8000]
  --max-alternatives=<max-alternatives>     Number of maximum alternatives to query from the server. [default: 10]
  --raw                                     Flag that specifies whether to stream raw audio bytes to server.
  --pcm                                     Flag that specifies whether to send raw pcm bytes.
  --word-level                              Flag to enable word level features from server.
"""
import time
import random
import threading
import traceback
from pprint import pprint
from typing import List

from docopt import docopt
from pydub import AudioSegment

from kaldi_serve import KaldiServeClient, RecognitionAudio, RecognitionConfig
from kaldi_serve.utils import (
    chunks_from_file,
    chunks_from_mic,
    raw_bytes_to_wav
)

ENCODING = RecognitionConfig.AudioEncoding.LINEAR16


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


def transcribe_chunks_streaming(client, audio_chunks, model: str, language_code: str,
                                sample_rate=8000, max_alternatives=10, raw: bool=False,
                                word_level: bool=False, chunk_size: float=0.5):
    """
    Transcribe the given audio chunks
    """

    response = {}

    try:
        if raw:
            config = lambda chunk_len: RecognitionConfig(
                sample_rate_hertz=sample_rate,
                encoding=ENCODING,
                language_code=language_code,
                max_alternatives=max_alternatives,
                model=model,
                raw=True,
                word_level=word_level,
                data_bytes=chunk_len
            )

            start = [None]
            def audio_params_gen(audio_chunks, start):
                for chunk in audio_chunks[:-1]:
                    yield config(len(chunk)), RecognitionAudio(content=chunk)
                    time.sleep(chunk_size)
                start[0] = time.time()
                yield config(len(audio_chunks[-1])), RecognitionAudio(content=audio_chunks[-1])

            response = client.streaming_recognize_raw(audio_params_gen(audio_chunks, start), uuid=str(random.randint(1000, 100000)))
            end = time.time()
            print(f"{((end - start[0])*1000):.2f}ms")
        else:
            audio = (RecognitionAudio(content=chunk) for chunk in audio_chunks)
            config = RecognitionConfig(
                sample_rate_hertz=sample_rate,
                encoding=ENCODING,
                language_code=language_code,
                max_alternatives=max_alternatives,
                model=model,
                word_level=word_level
            )
            response = client.streaming_recognize(config, audio, uuid=str(random.randint(1000, 100000)))
    except Exception as e:
        traceback.print_exc()
        print(f'error: {str(e)}')

    pprint(parse_response(response))

def transcribe_chunks_bidi_streaming(client, audio_chunks, model: str, language_code: str,
                                     sample_rate=8000, max_alternatives=10, raw: bool=False,
                                     word_level: bool=False):
    """
    Transcribe the given audio chunks
    """
    response = {}

    try:
        if raw:
            config = lambda chunk_len: RecognitionConfig(
                sample_rate_hertz=sample_rate,
                encoding=ENCODING,
                language_code=language_code,
                max_alternatives=max_alternatives,
                model=model,
                raw=True,
                data_bytes=chunk_len,
                word_level=word_level,
            )

            def audio_params_gen(audio_chunks):
                for chunk in audio_chunks:
                    yield config(len(chunk)), RecognitionAudio(content=chunk)

            response_gen = client.bidi_streaming_recognize_raw(audio_params_gen(audio_chunks), uuid=str(random.randint(1000, 100000)))
        else:
            config = RecognitionConfig(
                sample_rate_hertz=sample_rate,
                encoding=ENCODING,
                language_code=language_code,
                max_alternatives=max_alternatives,
                model=model,
                word_level=word_level
            )

            def audio_chunks_gen(audio_chunks):
                for chunk in audio_chunks:
                    yield RecognitionAudio(content=chunk)

            response_gen = client.bidi_streaming_recognize(config, audio_chunks_gen(audio_chunks), uuid=str(random.randint(1000, 100000)))
    except Exception as e:
        traceback.print_exc()
        print(f'error: {str(e)}')

    for response in response_gen:
        pprint(parse_response(response))


def decode_files(client, audio_paths: List[str], model: str, language_code: str,
                 sample_rate=8000, max_alternatives=10, raw: bool=False,
                 pcm: bool=False, word_level: bool=False, chunk_size: float=0.5):
    """
    Decode files using threaded requests
    """
    chunked_audios = [chunks_from_file(x, sample_rate=sample_rate, chunk_size=chunk_size, raw=raw, pcm=pcm) for x in audio_paths]

    threads = [
        threading.Thread(
            target=transcribe_chunks_streaming,
            args=(client, chunks, model, language_code,
                  sample_rate, max_alternatives, raw,
                  word_level, chunk_size)
        )
        for chunks in chunked_audios
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    args = docopt(__doc__)
    client = KaldiServeClient()

    # args
    model = args["--model"]
    language_code = args["--lang"]
    sample_rate = int(args["--sample-rate"])
    max_alternatives = int(args["--max-alternatives"])

    # flags
    raw = args['--raw']
    pcm = args['--pcm']
    word_level = args["--word-level"]

    if args["mic"]:
        transcribe_chunks_bidi_streaming(client, chunks_from_mic(int(args["--n-secs"]), sample_rate, 1),
                                         model, language_code, sample_rate, max_alternatives,
                                         raw or pcm, word_level)
    else:
        decode_files(client, args["<file>"], model, language_code,
                     sample_rate, max_alternatives, raw, pcm, word_level)
