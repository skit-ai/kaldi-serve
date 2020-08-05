"""
Script for transcribing audios in batches using Kaldi-Serve ASR server.

Usage:
  batch_decode.py <audio_paths_file> [--model=<model>] [--lang=<lang>] [--sample-rate=<sample-rate>] [--max-alternatives=<max-alternatives>] [--num-proc=<num-proc>] [--output-json=<output-json>] [--raw] [--transcripts-only]

Options:
  --model=<model>                           Name of the model to hit. [default: general]
  --lang=<lang>                             Language code of the model. [default: en]
  --sample-rate=<sample-rate>               Sampling rate to use for the audio. [default: 8000]
  --max-alternatives=<max-alternatives>     Number of maximum alternatives to query from the server. [default: 10]
  --num-proc=<num-proc>                     Number of parallel processes. [default: 8]
  --output-json=<output-json>               Output json file path for decoded transcriptions. [default: transcripts.json]
  --raw                                     Flag that specifies whether to stream raw audio bytes to server.
  --transcripts-only                        Flag that specifies whether or now to keep decoder metadata for transcripts.
"""

import json
import random
import traceback

from typing import List
from docopt import docopt
from tqdm import tqdm
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

from kaldi_serve import KaldiServeClient, RecognitionAudio, RecognitionConfig
from kaldi_serve.utils import byte_stream_from_file

ENCODING = RecognitionConfig.AudioEncoding.LINEAR16

client = KaldiServeClient()

def run_multiprocessing(func, tasks, num_processes=None):
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(func, tasks), total=len(tasks)))
    return results

def run_multithreading(func, tasks, num_workers=None):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(func, tasks), total=len(tasks)))
    return results

def parse_response(response):
    output = []

    for res in response.results:
        output.append([
            {
                "transcript": alt.transcript,
                "confidence": alt.confidence,
                "am_score": alt.am_score,
                "lm_score": alt.lm_score
            }
            for alt in res.alternatives
        ])
    return output


def transcribe_audio(audio_stream, model: str, language_code: str, sample_rate=8000, max_alternatives=10, raw: bool=False):
    """
    Transcribe the given audio chunks
    """
    global client

    try:
        audio = RecognitionAudio(content=audio_stream)
        
        config = RecognitionConfig(
            sample_rate_hertz=sample_rate,
            encoding=ENCODING,
            language_code=language_code,
            max_alternatives=max_alternatives,
            model=model,
            raw=raw,
            data_bytes=len(audio_stream)
        )

        response = client.recognize(config, audio, uuid=str(random.randint(1000, 100000)), timeout=1000)
    except Exception as e:
        print(f"error: {str(e)}")
        return []

    return parse_response(response)

def stream_and_transcribe(audio_path: str, model: str, language_code: str, sample_rate=8000, max_alternatives=10, raw: bool=False):
    try:
        audio_stream = byte_stream_from_file(audio_path, sample_rate, raw)
        result = transcribe_audio(audio_stream, model, language_code, sample_rate, max_alternatives, raw)
        return result
    except Exception as e:
        print('Error while handling {}'.format(audio_path))
        print(e)
        return None

def stream_and_transcribe_wrapper(args):
    return stream_and_transcribe(*args)

def decode_files(audio_paths: List[str], model: str, language_code: str,
                 sample_rate=8000, max_alternatives=10, raw: bool=False,
                 num_proc: int=8):
    """
    Decode files using parallel requests
    """
    args = [
        (path, model, language_code, sample_rate, max_alternatives, raw)
        for path in audio_paths
    ]

    results = run_multithreading(stream_and_transcribe_wrapper, args)

    results_dict = {path: response for path, response in list(zip(audio_paths, results)) if response is not None}
    return results_dict


if __name__ == "__main__":
    args = docopt(__doc__)

    # args
    model = args["--model"]
    language_code = args["--lang"]
    sample_rate = int(args["--sample-rate"])
    max_alternatives = int(args["--max-alternatives"])
    raw = args["--raw"]

    num_proc = int(args["--num-proc"])
    output_json = args["--output-json"]

    transcripts_only = args["--transcripts-only"]

    audio_paths_file = args["<audio_paths_file>"]
    with open(audio_paths_file, "r", encoding="utf-8") as f:
        audio_paths = f.read().split("\n")

    audio_paths = list(filter(lambda x: x.endswith(".wav"), audio_paths))
    results_dict = decode_files(audio_paths, model, language_code, sample_rate, max_alternatives, num_proc=num_proc, raw=raw)
    
    if transcripts_only:
        for audio_file, transcripts in results_dict.items():
            transcripts = [[alt["transcript"] if isinstance(alt, dict) else alt[0]["transcript"] for alt in segment] for segment in transcripts]
            results_dict[audio_file] = transcripts

    with open(output_json, "w", encoding="utf-8") as f:
        f.write(json.dumps(results_dict))

