"""
Utility functions for working with ASR, audio, devices etc.
"""

import io
import wave

import pyaudio

from pydub import AudioSegment
from pydub.silence import detect_nonsilent


def raw_bytes_to_wav(data: bytes, frame_rate: int, channels: int, sample_width: int) -> bytes:
    """
    Convert raw PCM bytes to wav bytes (with the initial 44 bytes header)
    """

    out = io.BytesIO()
    wf = wave.open(out, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(sample_width)
    wf.setframerate(frame_rate)
    wf.writeframes(data)
    wf.close()
    return out.getvalue()


def chunks_from_mic(secs: int, frame_rate: int, channels: int):
    """
    Generate wave audio chunks from microphone worth `secs` seconds.
    """

    p = pyaudio.PyAudio()
    sample_format = pyaudio.paInt16

    # 8k samples ~ 1sec of audio
    chunk_size = 8000

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=frame_rate,
                    frames_per_buffer=chunk_size,
                    input=True)

    sample_width = p.get_sample_size(sample_format)

    print('recording...')
    for _ in range(0, int(frame_rate / chunk_size * secs)):
        # The right way probably is to not send headers at all and let the
        # server side's chunk handler maintain state, taking data from
        # metadata.
        yield raw_bytes_to_wav(stream.read(chunk_size), frame_rate, channels, sample_width)

    stream.stop_stream()
    stream.close()
    p.terminate()


def chunks_from_file(filename: str, chunk_size=1, raw=False, pcm=False):
    """
    Return wav chunks of given size (in seconds) from the file.
    """

    # TODO: Should remove assumptions about audio properties from here
    audio = AudioSegment.from_file(filename, format="s16le" if pcm else "wav", frame_rate=8000, channels=1, sample_width=2)
    return chunks_from_audio_segment(audio, chunk_size=chunk_size, raw=True if pcm else raw)

def chunks_from_audio_segment(audio: str, chunk_size=1, raw=False):
    """
    Return wav chunks of given size (in seconds) from the audio segment.
    """
    if audio.duration_seconds <= chunk_size:
        if raw:
            return [audio.raw_data]
        else:
            audio_stream = io.BytesIO()
            audio.export(audio_stream, format="wav")
            return [audio_stream.getvalue()]

    chunks = []
    for i in range(0, len(audio), int(chunk_size * 1000)):
        chunk = audio[i: i + chunk_size * 1000]
        if raw:
            chunks.append(chunk.raw_data)
        else:
            chunk_stream = io.BytesIO()
            chunk.export(chunk_stream, format="wav")
            chunks.append(chunk_stream.getvalue())

    return chunks

def non_silent_segments_from_file(filename: str, segment_length=10):
    """
    Return wav chunks of given size (in seconds) from the file.
    """

    # TODO: Should remove assumptions about audio properties from here
    audio = AudioSegment.from_file(filename, format="wav", frame_rate=8000, channels=1, sample_width=2)

    # Normalize the audio
    audio = audio.apply_gain(-audio.max_dBFS)

    # Setting the silence threshold. Right now I am just subtracting 5dB from the average dB of the audio. Need to tinker with this
    avg_db = audio.dBFS
    min_silence_len = 250
    silence_thresh = avg_db - 30
    seek_step = 1

    # Get ranges which are "non silent" according to pydub
    not_silence_ranges = detect_nonsilent(audio, min_silence_len, silence_thresh, seek_step)

    segments = []

    # If empty, it is fully silent. Dont trust pydub. Send it to ASR anyway
    if not not_silence_ranges:
        for i in range(0, len(audio), segment_length * 1000):
            segments.append(audio[i: i + 1000])
    else:
        segment = AudioSegment.silent(duration=500, frame_rate=8000)
        for x in not_silence_ranges:
            segment += audio[x[0]: x[1]] + AudioSegment.silent(duration=500, frame_rate=8000)
        for i in range(0, len(segment), segment_length * 1000):
            segments.append(segment[i: i + (segment_length * 1000)])

    return segments
