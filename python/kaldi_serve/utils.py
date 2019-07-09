"""
Utility functions for working with ASR, audio, devices etc.
"""

from io import BytesIO

from pydub import AudioSegment


def get_chunks_from_file(filename: str, chunk_size=1):
    """
    Return wav chunks of given size (in seconds) from the file.
    """

    # TODO: Should remove assumptions about audio properties from here
    audio = AudioSegment.from_file(filename, format="wav", frame_rate=8000, channels=1, sample_width=2)

    if audio.duration_seconds == chunk_size:
        audio_stream = BytesIO()
        audio.export(audio_stream, format="wav")
        return [audio_stream.getvalue()]

    chunks = []
    for i in range(0, len(audio), int(chunk_size * 1000)):
        chunk = audio[i: i + chunk_size * 1000]
        chunk_stream = BytesIO()
        chunk.export(chunk_stream, format="wav")
        chunks.append(chunk_stream.getvalue())

    return chunks
