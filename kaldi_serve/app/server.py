from flask import jsonify, request, make_response
from werkzeug import secure_filename
from kaldi_serve import config, app, inference
import os

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

        except:
            return jsonify(status='error', description="Unable to find 'file'")

        try:
            config_obj = config.config[lang][model]
            config_obj["wav_filename"] = wav_filename
        except:
            return jsonify(status='error', description="Wrong lang or model")

        transcription = inference.inference(config_obj)
    else:
        return jsonify(status='error', description="Unsupported HTTP method")

    return make_response(transcription)

def start_server(*args, **kwargs):
    app.run(*args, **kwargs)
