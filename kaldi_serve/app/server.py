from flask import jsonify, request, make_response
from werkzeug import secure_filename
from pydub import AudioSegment
from pydub.silence import split_on_silence

from kaldi_serve import config, app, inference
import os
import json

@app.route("/transcribe/<lang>/<model>/", methods=["POST"])
def transcribe(lang: str='en', model: str='tdnn'):
    """
    Transcribe audio
    """
    if request.method == "POST":
        try:
            f = request.files['file']
            filename = secure_filename(f.filename)
            wav_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(wav_filename)
            complete_audio = AudioSegment.from_file(wav_filename)
            chunks = split_on_silence(complete_audio, silence_thresh=-26, min_silence_len=500, keep_silence=500)
            chunks = chunks if len(chunks)>0 else [complete_audio]
        except:
            return jsonify(status='error', description="Unable to find 'file'")

        try:
            transcriptions = []
            for i, chunk in enumerate(chunks):
                chunk_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename.strip(".wav")+"chunk"+str(i)+".wav")
                chunk.export(chunk_filename, format="wav")
                config_obj = config.config[lang][model]
                config_obj["wav_filename"] = chunk_filename
                transcription = inference.inference(config_obj)
                transcriptions.append(transcription)
        except:
            return jsonify(status='error', description="Wrong lang or model")

    else:
        return jsonify(status='error', description="Unsupported HTTP method")

    utf_rep = json.dumps(transcriptions, ensure_ascii=False).encode("utf8")
    return make_response(utf_rep)

def start_server(*args, **kwargs):
    app.run(*args, **kwargs)
