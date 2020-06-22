"""
Audio Transcription script using kalidserve.

Usage: transcribe.py <model-spec-toml> <audio-file>
"""
import time

from io import BytesIO
from typing import List, Text
from docopt import docopt

import kaldiserve as ks


def transcribe(decoder: ks.Decoder, wav_stream: bytes) -> List[ks.Alternative]:
    with ks.start_decoding(decoder):
        # decode the audio
        decoder.decode_wav_audio(wav_stream)
        # get the transcripts
        alts = decoder.get_decoded_results(10)
    return alts


if __name__ == "__main__":
    args = docopt(__doc__)

    model_spec_toml = args["<model-spec-toml>"]
    audio_file = args["<audio-file>"]

    # parse model spec
    model_spec = ks.parse_model_specs(model_spec_toml)[0]
    # create chain model
    model = ks.ChainModel(model_spec)
    # create decoder instance
    decoder = ks.Decoder(model)

    # read audio bytes
    with open(audio_file, "rb") as f:
        audio_bytes = BytesIO(f.read()).getvalue()

    # transcribe audio
    start = time.time()
    alts = transcribe(decoder, audio_bytes)
    end = time.time()
    print(f"{audio_file}: decoded audio in {(end - start):.4f}s")

    print(f"{audio_file}: Alternatives\n{alts}")