"""
Batch Audio Transcription script using kalidserve.

Usage: batch_transcribe.py <model-spec-toml> <audio-paths-file>
"""
import time
import threading

from io import BytesIO
from typing import List, Text
from docopt import docopt

import kaldiserve as ks


def transcribe(decoder: ks.Decoder, wav_stream: bytes) -> List[ks.Alternative]:
    with ks.start_decoding(decoder):
        # decode the audio
        decoder.decode_wav_audio(wav_stream)
        # get the transcripts
        alts = decoder.get_decoded_results(10, False, False)
    return alts


def decode_thread(decoder_queue: ks.DecoderQueue, audio_file: Text, n: int):
    # read audio bytes
    with open(audio_file, "rb") as f:
        audio_bytes = BytesIO(f.read()).getvalue()

    start = time.time()
    with ks.acquire_decoder(decoder_queue) as decoder:
        end = time.time()
        print(f"{audio_file}: decoder acquired in {(end - start):.4f}s")
        # transcribe audio
        start = time.time()
        alts = transcribe(decoder, audio_bytes)
        end = time.time()
        print(f"{audio_file}: decoded audio in {(end - start):.4f}s")

    print(f"{audio_file}: Alternatives\n{alts}")


if __name__ == "__main__":
    args = docopt(__doc__)

    model_spec_toml = args["<model-spec-toml>"]
    audio_paths_file = args["<audio-paths-file>"]

    # parse model spec
    model_spec = ks.parse_model_specs(model_spec_toml)[0]
    # create decoder queue
    decoder_queue = ks.DecoderQueue(model_spec)

    # read audio paths
    with open(audio_paths_file, "r", encoding="utf-8") as f:
        audio_paths = f.read().split("\n")

    audio_paths = list(filter(lambda x: x.endswith(".wav"), audio_paths))

    # multithreaded decoding
    threads = [
        threading.Thread(target=decode_thread, args=(decoder_queue, audio_path, i + 1,))
        for i, audio_path in enumerate(audio_paths)
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()