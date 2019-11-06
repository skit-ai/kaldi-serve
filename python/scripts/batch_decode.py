"""
Script for transcribing audios in batches using Kaldi-Serve ASR server.

Usage:
  batch_decode.py <audio_paths_file> [--model=<model>] [--lang=<lang>] [--num-proc=<num-proc>] [--output-json=<output-json>] [--segment-long-utts] [--raw]

Options:
  --model=<model>               Name of the model to hit [default: general]
  --lang=<lang>                 Language code of the model [default: hi]
  --num-proc=<num-proc>         Number of parallel processes [default: 8]
  --output-json=<output-json>   Output json file path for decoded transcriptions [default: transcripts.json]
  --segment-long-utts           Flag that specifies whether to segment long audios with some overlap.
  --raw                         Flag that specifies whether to stream raw audio bytes to server.
"""

import json
import traceback

from typing import List
from docopt import docopt

from multiprocessing import Pool

from kaldi_serve import KaldiServeClient, RecognitionAudio, RecognitionConfig
from kaldi_serve.utils import (
    non_silent_segments_from_file,
    chunks_from_audio_segment,
    chunks_from_file,
    raw_bytes_to_wav
)

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
    return output


def transcribe_chunks(audio_chunks, model: str, language_code: str, raw: bool=False):
    """
    Transcribe the given audio chunks
    """
    global client

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
            )
            response = client.streaming_recognize(config, audio, uuid="")
    except Exception as e:
        traceback.print_exc()
        print(f'error: {str(e)}')
        return None

    return parse_response(response)


def decode_audios(audio_paths: List[str], model: str, language_code: str, num_proc: int=8, segment_long_utt: bool=False, raw: bool=False):
    """
    Decode files using threaded requests
    """
    if segment_long_utt:
        segmented_audios = [non_silent_segments_from_file(audio_file, segment_length=10) for audio_file in audio_paths]
        chunked_segmented_audios = [[chunks_from_audio_segment(seg, chunk_size=1, raw=raw) for seg in segments] for segments in segmented_audios]

        args = [
            [(segment_chunks, model, language_code, raw) for segment_chunks in segmented_audio]
            for segmented_audio in chunked_segmented_audios
        ]

        results = []
        for audio_segs in args:
            with Pool(num_proc) as pool:
                results.extend(pool.starmap(transcribe_chunks, audio_segs))
    else:
        chunked_audios = [chunks_from_file(x, chunk_size=1, raw=raw) for x in audio_paths]

        args = [
            (chunks, model, language_code, raw)
            for chunks in chunked_audios
        ]

        with Pool(num_proc) as pool:
            results = pool.starmap(transcribe_chunks, args)

    results_dict = {path: response for path, response in list(zip(audio_paths, results))}
    return results_dict


if __name__ == "__main__":
    args = docopt(__doc__)

    model = args["--model"]
    language_code = args["--lang"]
    num_proc = int(args['--num-proc'])
    output_json = args['--output-json']

    segment_long_utt = args['--segment-long-utts']
    raw = args['--raw']

    audio_paths_file = args["<audio_paths_file>"]
    with open(audio_paths_file, 'r', encoding='utf-8') as f:
        audio_paths = f.read().split('\n')

    audio_paths = list(filter(lambda x: x.endswith(".wav"), audio_paths))

    results_dict = decode_audios(audio_paths, model, language_code, num_proc=num_proc, segment_long_utt=segment_long_utt, raw=raw)
    with open(output_json, 'w', encoding='utf-8') as f:
        f.write(json.dumps(results_dict))

