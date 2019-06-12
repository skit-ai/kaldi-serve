import json
import os
from typing import Dict, List

from flask import Flask, request

import tdnn_decode
import utils
from celery import Celery

CELERY_BROKER_URL = "redis://{}:6379/{}".format(
    os.environ.get("REDIS_HOST", "localhost"), os.environ.get("REDIS_VIRTUAL_PORT", "0")
)
COMMA_HOST_URL = os.environ.get("COMMA_HOST", "localhost:8000")
REDIS_EXPIRY_TIME = 10800  # 3hrs

flask_app = Flask("kaldi-serve")
celery = Celery("asr-server", broker=CELERY_BROKER_URL)
celery.conf.update(
    task_routes={
        "preprocess-task": {"queue": "preprocess"},
        "asr-task": {"queue": "asr"},
        "asr-complete-task": {"queue": "preprocess"},
    }
)

config = {
    "en": {
        "word_syms_filename": "models/english/s6/exp/chain/tree_a_sp/graph/words.txt",
        "model_in_filename": "models/english/s6/exp/chain/tdnn1g_sp_online/final.mdl",
        "fst_in_str": "models/english/s6/exp/chain/tree_a_sp/graph/HCLG.fst",
        "mfcc_config": "models/english/s6/exp/chain/tdnn1g_sp_online/conf/mfcc.conf",
        "ie_conf_filename": "models/english/s6/exp/chain/tdnn1g_sp_online/conf/ivector_extractor.conf",
    },
    "en-bbq": {
        "word_syms_filename": "models/english-bbq/s6/exp/chain/tree_a_sp/graph/words.txt",
        "model_in_filename": "models/english-bbq/s6/exp/chain/tdnn1g_sp_online/final.mdl",
        "fst_in_str": "models/english-bbq/s6/exp/chain/tree_a_sp/graph/HCLG.fst",
        "mfcc_config": "models/english-bbq/s6/exp/chain/tdnn1g_sp_online/conf/mfcc.conf",
        "ie_conf_filename": "models/english-bbq/s6/exp/chain/tdnn1g_sp_online/conf/ivector_extractor.conf",
    },
    "hi": {
        "word_syms_filename": "models/hindi/s2/exp/chain/tree_a_sp/graph/words.txt",
        "model_in_filename": "models/hindi/s2/exp/chain/tdnn1g_sp_online/final.mdl",
        "fst_in_str": "models/hindi/s2/exp/chain/tree_a_sp/graph/HCLG.fst",
        "mfcc_config": "models/hindi/s2/exp/chain/tdnn1g_sp_online/conf/mfcc.conf",
        "ie_conf_filename": "models/hindi/s2/exp/chain/tdnn1g_sp_online/conf/ivector_extractor.conf",
    },
    "kn": {
        "word_syms_filename": "models/kannada/words.txt",
        "model_in_filename": "models/kannada/final.mdl",
        "fst_in_str": "models/kannada/HCLG.fst",
        "mfcc_config": "models/kannada/mfcc.conf",
        "ie_conf_filename": "models/kannada/ivector_extractor.conf",
    },
    "ml": {
        "word_syms_filename": "models/malayalam/words.txt",
        "model_in_filename": "models/malayalam/final.mdl",
        "fst_in_str": "models/malayalam/HCLG.fst",
        "mfcc_config": "models/malayalam/mfcc.conf",
        "ie_conf_filename": "models/malayalam/ivector_extractor.conf",
    },
    "bn": {
        "word_syms_filename": "models/bengali/words.txt",
        "model_in_filename": "models/bengali/final.mdl",
        "fst_in_str": "models/bengali/HCLG.fst",
        "mfcc_config": "models/bengali/mfcc.conf",
        "ie_conf_filename": "models/bengali/ivector_extractor.conf",
    },
    "te": {
        "word_syms_filename": "models/telugu/words.txt",
        "model_in_filename": "models/telugu/final.mdl",
        "fst_in_str": "models/telugu/HCLG.fst",
        "mfcc_config": "models/telugu/mfcc.conf",
        "ie_conf_filename": "models/telugu/ivector_extractor.conf",
    },
    "ta": {
        "word_syms_filename": "models/tamil/words.txt",
        "model_in_filename": "models/tamil/final.mdl",
        "fst_in_str": "models/tamil/HCLG.fst",
        "mfcc_config": "models/tamil/mfcc.conf",
        "ie_conf_filename": "models/tamil/ivector_extractor.conf",
    },
}

en_model = None
en_bbq_model = None
hi_model = None
kn_model = None
ml_model = None
te_model = None
ta_model = None
bn_model = None


@celery.task(name="asr-task")
def run_asr_async(operation_name: str, audio_uri: str, config: Dict):
    """
    :param operation_name: job id for this process
    :param audio_uri: audio url to read from [path to file on NFS]
    :param config:
                {
                    "language_code": "hi",
                    "sample_rate_hertz":"8000",
                    "encoding": "LINEAR16",
                    "max_alternatives": 1,
                    "punctuation": false,
                    "ontology" : {
                        "type": "",
                        "value": {}
                    }
                }
    """
    utils.copy_models()

    results, error = transcribe(audio_uri, config, operation_name)

    celery.send_task(
        "asr-complete-task",
        kwargs={"operation_name": operation_name, "results": results, "error": error},
    )


@flask_app.route("/run-asr/", methods=["POST"])
def run_asr_sync():
    """
    post data:
    {
    operation_name: "xxxx",
    audio_uri: "[PATH-TO-NFS]"
    config: {
                "language_code": "hi",
                "sample_rate_hertz":"16000",
                "encoding": "LINEAR16",
                "max_alternatives": 1,
                "punctuation": false,
                "ontology" : {
                    "type": "",
                    "value": {}
                }
            }
    }
    """
    utils.copy_models()
    data = request.get_json()

    # start the process here
    results, error = transcribe(
        data["audio_uri"], data["config"], data["operation_name"]
    )

    return json.dumps({"results": results, "error": error})


def get_model(lang: str, config: Dict):
    if lang == "en":
        global en_model
        if not en_model:
            en_model = tdnn_decode.load_model(
                13.0,
                7000,
                200,
                6.0,
                1.0,
                3,
                config["word_syms_filename"],
                config["model_in_filename"],
                config["fst_in_str"],
                config["mfcc_config"],
                config["ie_conf_filename"],
            )
        return en_model
    if lang == "en-bbq":
        global en_bbq_model
        if not en_bbq_model:
            en_bbq_model = tdnn_decode.load_model(
                13.0,
                7000,
                200,
                6.0,
                1.0,
                3,
                config["word_syms_filename"],
                config["model_in_filename"],
                config["fst_in_str"],
                config["mfcc_config"],
                config["ie_conf_filename"],
            )
        return en_bbq_model
    elif lang == "hi":
        global hi_model
        if not hi_model:
            hi_model = tdnn_decode.load_model(
                13.0,
                7000,
                200,
                6.0,
                1.0,
                3,
                config["word_syms_filename"],
                config["model_in_filename"],
                config["fst_in_str"],
                config["mfcc_config"],
                config["ie_conf_filename"],
            )
        return hi_model
    elif lang == "kn":
        global kn_model
        if not kn_model:
            kn_model = tdnn_decode.load_model(
                13.0,
                7000,
                200,
                6.0,
                1.0,
                3,
                config["word_syms_filename"],
                config["model_in_filename"],
                config["fst_in_str"],
                config["mfcc_config"],
                config["ie_conf_filename"],
            )
        return kn_model
    elif lang == "ml":
        global ml_model
        if not ml_model:
            ml_model = tdnn_decode.load_model(
                13.0,
                7000,
                200,
                6.0,
                1.0,
                3,
                config["word_syms_filename"],
                config["model_in_filename"],
                config["fst_in_str"],
                config["mfcc_config"],
                config["ie_conf_filename"],
            )
        return ml_model
    elif lang == "ta":
        global ta_model
        if not ta_model:
            ta_model = tdnn_decode.load_model(
                13.0,
                7000,
                200,
                6.0,
                1.0,
                3,
                config["word_syms_filename"],
                config["model_in_filename"],
                config["fst_in_str"],
                config["mfcc_config"],
                config["ie_conf_filename"],
            )
        return ta_model
    elif lang == "te":
        global te_model
        if not te_model:
            te_model = tdnn_decode.load_model(
                13.0,
                7000,
                200,
                6.0,
                1.0,
                3,
                config["word_syms_filename"],
                config["model_in_filename"],
                config["fst_in_str"],
                config["mfcc_config"],
                config["ie_conf_filename"],
            )
        return te_model
    elif lang == "bn":
        global bn_model
        if not bn_model:
            bn_model = tdnn_decode.load_model(
                13.0,
                7000,
                200,
                6.0,
                1.0,
                3,
                config["word_syms_filename"],
                config["model_in_filename"],
                config["fst_in_str"],
                config["mfcc_config"],
                config["ie_conf_filename"],
            )
        return bn_model


def add_punctuations(text: str, lang: str) -> str:
    return text


def transcribe(
    audio_uri: str, audio_config: Dict, operation_name: str
) -> (List[str], str):
    """
    Transcribe audio
    """
    try:
        lang = audio_config["language_code"]
        chunks = utils.get_chunks(audio_uri)

        transcriptions = []
        for i, chunk in enumerate(chunks):
            # write chunk to file
            chunk_filename = "/home/%s_chunk_%s.wav" % (operation_name, i)
            chunk.export(chunk_filename, format="wav")

            # load model
            config_obj = config[lang]
            _model = get_model(lang, config_obj)

            # call infer
            asr_results = tdnn_decode.infer(
                _model, chunk_filename, audio_config["max_alternatives"]
            )
            alternatives = []
            for (transcription, confidence) in asr_results:
                alternatives.append(
                    {
                        "transcript": add_punctuations(transcription, lang)
                        if audio_config["punctuation"]
                        else transcription,
                        "confidence": confidence,
                    }
                )
            transcriptions.append({"alternatives": alternatives})
    except Exception as e:
        return [], str(e)
    return transcriptions, None


if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port="8002")
