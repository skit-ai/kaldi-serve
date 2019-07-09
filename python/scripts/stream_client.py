"""
Client for testing out streaming ASR server

Usage:
  stream_client.py <file>...
"""

import random
import threading
import time
import traceback
from pprint import pprint
from typing import List

from docopt import docopt

from kaldi_serve import KaldiClient, RecognitionAudio, RecognitionConfig
from kaldi_serve.utils import get_chunks_from_file

CLIENT = None
TIMES = []


def transcribe_chunks(audio_chunks, language_code='hi', **kwargs):
    """Transcribe the given audio chunks"""
    print(f'no. of audio chunks: {len(audio_chunks)}')
    global CLIENT
    if not CLIENT:
        CLIENT = KaldiClient()
    response = {}

    encoding = RecognitionConfig.AudioEncoding.LINEAR16

    audio = [RecognitionAudio(content=chunk) for chunk in audio_chunks]
    config = RecognitionConfig(
        sample_rate_hertz=kwargs.get('sampling_rate', 8000),
        encoding=encoding,
        language_code=language_code,
        max_alternatives=10,
        model=kwargs.get('model', None),
    )

    try:
        start_time = time.clock()
        response = CLIENT.recognize(config, audio, uuid=kwargs.get('uuid', ''), timeout=80)
        elapsed = time.clock() - start_time
        TIMES.append(elapsed)
    except Exception as e:
        status_code = 500
        traceback.print_exc()
        print(f'error: {str(e)}')

    pprint(transcript_dict(response)['transcript'])


def transcript_dict(response):
    # Initial values of transcript, confidence and alternatives
    transcript = '_unknown_'
    confidence = 0.0
    alternatives = [[]]

    # Parsing the results of the ASR
    if response and hasattr(response, 'results'):
        for result in response.results:
            # The first alternative is the most likely one for this portion.
            if hasattr(result, 'alternatives') and result.alternatives:
                transcript = result.alternatives[0].transcript.lower()
                confidence = result.alternatives[0].confidence
        alternatives = parse_response(response)

    # Building the transcription dict
    return {
        "alternatives": alternatives,
        "transcript": transcript,
        "confidence": confidence
    }


def _parse_result(res):
    return [{
        "transcript": alt.transcript,
        "confidence": alt.confidence
    } for alt in res.alternatives]


def parse_response(response):
    """
    Parse response from GSpeech client and return a dictionary
    NOTE: We are not parsing word information from the alternatives
    """
    return [_parse_result(res) for res in response.results]


def main(audio_paths: List[str]):
    audio_paths = audio_paths * 200
    chunked_audios = [get_chunks_from_file(x, chunk_size=random.randint(1, 3)) for x in audio_paths]

    threads = [
        threading.Thread(target=transcribe_chunks, args=(audio_chunks, ))
        for audio_chunks in chunked_audios
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    avg_time = sum(TIMES) / len(TIMES)
    print((len(TIMES) / len(audio_paths)) * 100)
    print(avg_time)
    print(min(TIMES))
    print(max(TIMES))


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args["<file>"])
