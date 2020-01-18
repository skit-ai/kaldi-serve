"""
Script for transcribing audios in batches using Kaldi-Serve ASR server.

Usage:
  batch_decode.py <audio_paths_file> [--model=<model>] [--lang=<lang>] [--num-proc=<num-proc>] [--output-json=<output-json>] [--transcripts-only]

Options:
  --model=<model>               Name of the model to hit [default: general]
  --lang=<lang>                 Language code of the model [default: hi]
  --num-proc=<num-proc>         Number of parallel processes [default: 8]
  --output-json=<output-json>   Output json file path for decoded transcriptions [default: transcripts.json]
  --transcripts-only            Flag that specifies whether or now to keep decoder metadata for transcripts.
"""

import json
import traceback

from typing import List
from docopt import docopt

from multiprocessing import Pool

from kaldi_serve import KaldiServeClient, RecognitionAudio, RecognitionConfig
from kaldi_serve.utils import byte_stream_from_file

SR = 8000
CHANNELS = 1

client = KaldiServeClient()

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
    if len(output) == 1:
        return output[0]
    return output


def transcribe_audio(audio_stream, model: str, language_code: str):
    """
    Transcribe the given audio chunks
    """
    global client

    encoding = RecognitionConfig.AudioEncoding.LINEAR16

    try:
        audio = RecognitionAudio(content=audio_stream)
        
        config = RecognitionConfig(
            sample_rate_hertz=SR,
            encoding=encoding,
            language_code=language_code,
            max_alternatives=10,
            model=model,
        )
        
        response = client.recognize(config, audio, uuid="", timeout=1000)
    except Exception as e:
        print(f"error: {str(e)}")
        return []

    return parse_response(response)


def decode_audios(audio_paths: List[str], model: str, language_code: str, num_proc: int=8, segment_long_utt: bool=False, raw: bool=False):
    """
    Decode files using threaded requests
    """
    audio_streams = [byte_stream_from_file(x) for x in audio_paths]

    args = [
        (stream, model, language_code, raw)
        for stream in audio_streams
    ]

    with Pool(num_proc) as pool:
        results = pool.starmap(transcribe_audio, args)

    results_dict = {path: response for path, response in list(zip(audio_paths, results))}
    return results_dict


if __name__ == "__main__":
    args = docopt(__doc__)

    model = args["--model"]
    language_code = args["--lang"]
    num_proc = int(args["--num-proc"])
    output_json = args["--output-json"]

    transcripts_only = args["--transcripts-only"]

    audio_paths_file = args["<audio_paths_file>"]
    with open(audio_paths_file, "r", encoding="utf-8") as f:
        audio_paths = f.read().split("\n")

    audio_paths = list(filter(lambda x: x.endswith(".wav"), audio_paths))

    results_dict = decode_audios(audio_paths, model, language_code, num_proc=num_proc)
    
    if transcripts_only:
        for audio_file, transcripts in results_dict.items():
            transcripts = [[alt["transcript"] if isinstance(alt, dict) else alt[0]["transcript"] for alt in segment] for segment in transcripts]
            results_dict[audio_file] = transcripts

    with open(output_json, "w", encoding="utf-8") as f:
        f.write(json.dumps(results_dict))

