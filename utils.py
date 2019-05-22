import os
import shutil

from pydub.silence import detect_nonsilent
from pydub import AudioSegment

def get_chunks(filename):
    complete_audio = AudioSegment.from_file(filename)

    # Normalize the audio
    complete_audio = complete_audio.apply_gain(-complete_audio.max_dBFS)

    # Setting the silence threshold. Right now I am just subtracting 5dB from the average dB of the audio. Need to tinker with this
    avg_db = complete_audio.dBFS
    min_silence_len = 500
    silence_thresh = avg_db - 5


    seek_step = 1

    # Get ranges which are "non silent" according to pydub
    not_silence_ranges = detect_nonsilent(complete_audio, min_silence_len, silence_thresh, seek_step)

    # If empty, it is fully silent. Dont trust pydub. Send it to ASR anyway
    if not not_silence_ranges:
        return [complete_audio]

    # Dont need this right now. Logic is, append 500 ms of silence to non silent audio at the beginning and end
    keep_silence = 500
    chunks = []

    current_start = 0
    current_end = 0
    current_index = 0

    # Force the "last" non silent range to extend till the end of the audio
    not_silence_ranges[-1][1] = len(complete_audio)

    chunks = []
    while(current_end < len(complete_audio)):
        # Breaking into 30 sec chunks. Increase/decrease based on results
        while((current_index < len(not_silence_ranges)) and (not_silence_ranges[current_index][1] - current_start) <= 30000):
            current_end = not_silence_ranges[current_index][1]
            current_index += 1
        chunks.append(complete_audio[current_start:current_end])
        current_start = current_end
    return chunks


def copy_models():
    if not os.path.exists("/home/app/models"):
        try:
            models_path = os.environ['MODELS_PATH']
            shutil.copytree(models_path, "/home/app/models")
        except OSError as e:
            print('models not copied. Error: %s' % e)