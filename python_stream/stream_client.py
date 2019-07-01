import traceback
from io import BytesIO

from pprint import pprint

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from kaldi import KaldiClient, RecognitionAudio, RecognitionConfig

def get_chunks(filename):
    """
    taken from `http-server` branch of `kaldi-serve`
    """
    complete_audio = AudioSegment.from_file(filename)
    print(f'audio len: {len(complete_audio)}')
    print(f'channels: {complete_audio.channels}')
    print(f'frame rate: {complete_audio.frame_rate}')

    # Normalize the audio
    # complete_audio = complete_audio.apply_gain(-complete_audio.max_dBFS)

    # # Setting the silence threshold. Right now I am just subtracting 5dB from the average dB of the audio. Need to tinker with this
    # avg_db = complete_audio.dBFS
    # min_silence_len = 500
    # silence_thresh = avg_db - 5
    chunk_len = 3 # audio seconds per chunk

    # seek_step = 1

    # # Get ranges which are "non silent" according to pydub
    # not_silence_ranges = detect_nonsilent(complete_audio, min_silence_len, silence_thresh, seek_step)

    # If empty, it is fully silent. Dont trust pydub. Send it to ASR anyway
    # if not not_silence_ranges:
    #     audio_stream = BytesIO()
    #     complete_audio.export(audio_stream, format='wav')
    #     return [audio_stream.getvalue()]

    # # Force the "last" non silent range to extend till the end of the audio
    # not_silence_ranges[-1][1] = len(complete_audio)

    # non_silence = complete_audio[not_silence_ranges[0][0]: not_silence_ranges[0][1]]
    # for ns in not_silence_ranges[1:]:
    #     non_silence.append(complete_audio[ns[0]: ns[1]])

    # print(f'channels: {non_silence.channels}')

    chunks = []
    for i in range(0, len(complete_audio), int(chunk_len * 1000)):
        audio_stream = BytesIO()
        complete_audio[i: i + chunk_len * 1000].set_frame_rate(8000).export(audio_stream, format='wav')
        chunks.append(audio_stream.getvalue())
    return chunks

client = None

def transcribe_file(audio_chunks, language_code='hi', **kwargs):
    """Transcribe the given audio file."""
    print(f'no. of audio chunks: {len(audio_chunks)}')
    global client
    if not client:
        client = KaldiClient()
    response = {}

    status_code = 200
    encoding = RecognitionConfig.AudioEncoding.LINEAR16
     
    audio = [RecognitionAudio(content=chunk) for chunk in audio_chunks]
    config = RecognitionConfig(
        sample_rate_hertz=kwargs.get('sampling_rate', 8000),
        encoding = encoding,
        language_code=language_code,
        max_alternatives=10,
        model=kwargs.get('model', None),
    )

    try:
        response = client.recognize(config, audio, uuid=kwargs.get('uuid', '5512341'), timeout=3)
    except Exception as e:
        status_code = 500
        traceback.print_exc()
        print(f'error: {str(e)}')

    return transcript_dict(response), status_code

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

def main():
    audio_path = '../audio/some2.wav'
    audio_chunks = get_chunks(audio_path)

    result = transcribe_file(audio_chunks)
    pprint(result)

if __name__ == "__main__":
    main()