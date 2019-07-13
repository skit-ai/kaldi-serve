"""
Utility functions for working with ASR, audio, devices etc.
"""

import io
import wave

import pyaudio
from pydub import AudioSegment


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

    # This is in samples not seconds
    chunk_size = 4000

    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=frame_rate,
                    frames_per_buffer=chunk_size,
                    input=True)

    sample_width = p.get_sample_size(sample_format)

    for _ in range(0, int(frame_rate / chunk_size * secs)):
        # The right way probably is to not send headers at all and let the
        # server side's chunk handler maintain state, taking data from
        # metadata.
        yield raw_bytes_to_wav(stream.read(chunk_size), frame_rate, channels, sample_width)

    stream.stop_stream()
    stream.close()
    p.terminate()


def chunks_from_file(filename: str, chunk_size=1):
    """
    Return wav chunks of given size (in seconds) from the file.
    """

    # TODO: Should remove assumptions about audio properties from here
    audio = AudioSegment.from_file(filename, format="wav", frame_rate=8000, channels=1, sample_width=2)

    if audio.duration_seconds <= chunk_size:
        audio_stream = io.BytesIO()
        audio.export(audio_stream, format="wav")
        return [audio_stream.getvalue()]

    chunks = []
    for i in range(0, len(audio), int(chunk_size * 1000)):
        chunk = audio[i: i + chunk_size * 1000]
        chunk_stream = io.BytesIO()
        chunk.export(chunk_stream, format="wav")
        chunks.append(chunk_stream.getvalue())

    return chunks
