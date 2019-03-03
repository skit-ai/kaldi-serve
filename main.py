import os
import json
import subprocess
import shutil
from datetime import datetime
from typing import Dict, List

from celery import Celery
from pydub import AudioSegment
from pydub.silence import split_on_silence

import utils
from redis_utils import get_redis_data, set_redis_data

CELERY_BROKER_URL = 'redis://{}:6379/{}'.format(
    os.environ.get('REDIS_HOST', 'localhost'),
    os.environ.get('REDIS_VIRTUAL_PORT', '0')
)
REDIS_EXPIRY_TIME = 10800   # 3hrs

celery = Celery('asr-server', broker=CELERY_BROKER_URL)
celery.conf.update(
    task_routes={
        'preprocess-task': {'queue': 'preprocess'},
        'asr-task': {'queue': 'asr'}
    },
)

config = {
    "hi": {
        "tdnn": {
            "script": "./scripts/inference_tdnn_online.sh",
            "args": [
                "models/hindi/exp/chain/tdnn1g_sp_online/conf/online.conf",
                "models/hindi/exp/chain/tdnn1g_sp_online/final.mdl",
                "models/hindi/exp/chain/tree_a_sp/graph/HCLG.fst",
                "models/hindi/exp/chain/tree_a_sp/graph/words.txt"
            ]
        },
        "gmm": {
            "script": "./scripts/inference_gmm_online.sh",
            "args": [
                "models/hindi/exp/tri3b_online/conf/online_decoding.conf",
                "models/hindi/exp/tri3b/graph/HCLG.fst",
                "models/hindi/exp/tri3b/graph/words.txt"
            ]
        }
    },
    "en": {
       "gmm": {
           "script": "./scripts/inference_gmm_online.sh",
            "args": [
                "models/english/exp/tri3b_online/conf/online_decoding.conf",
                "models/english/exp/tri3b/graph/HCLG.fst",
                "models/english/exp/tri3b/graph/words.txt"
            ]
       }
    }
}

models_copied = False

def copy_models():
    if not os.path.exists("/home/app/models"):
        try:
            shutil.copytree("/vol/data/models", "/home/app/models")
        except OSError as e:
            print('models not copied. Error: %s' % e)

    global models_copied
    models_copied = True

@celery.task(name="asr-task")
def run_asr(operation_name: str, audio_uri: str, config: Dict):
    """
    :param operation_name: job id for this process
    :param audio_uri: audio url to read from [path to file on NFS]
    :param config:
                {
                    "language_code": "hi",
                    "sample_rate_hertz":"16000",
                    "encoding": "LINEAR16"
                }
    """
    global models_copied
    if not models_copied:
        copy_models()
    
    # start the process here
    results, error = transcribe(audio_uri, config["language_code"], operation_name)

    print("results", results, error)

    # process ends
    job_data = get_redis_data(operation_name)
    job_data["done"] = True
    job_data["metadata"]["lastUpdateTime"] = str(datetime.now())

    if error:
        job_data["error"] = True
        job_data["errorMessage"] = "Some internal error occurred"
    else:
        job_data["error"] = False
        final_results = []
        for r in results:
            final_results.append({
                "alternatives": [
                    {
                        "transcript": r,
                        "confidence": "-",
                    },
                ]
            })
        job_data["results"] = final_results

    # update results to redis
    set_redis_data(operation_name, job_data, REDIS_EXPIRY_TIME)


def inference(config: Dict):
    script_args = [
        config["script"],
        config["wav_filename"],
        *config["args"]
    ]
    decode_process = subprocess.Popen(script_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = decode_process.communicate()
    return str(stdout.decode("utf-8"))


def cleanup(tr):
    tr = tr.strip().replace("utterance-id1", "")
    tr = tr.replace("utterance-id-1", "")
    return tr


def transcribe(audio_uri: str, lang: str, operation_name:str, model: str=None) -> (List[str], str):
    """
    Transcribe audio
    """
    if lang == "en":
        model = "gmm"
    else:
        if model is None:
            model = "tdnn"
    try:
        chunks = utils.get_chunks(audio_uri)

        transcriptions = []
        for i, chunk in enumerate(chunks):
            chunk_filename = "/home/%s_chunk_%s.wav" % (operation_name, i)
            chunk.export(chunk_filename, format="wav")
            config_obj = config[lang][model]
            config_obj["wav_filename"] = chunk_filename
            transcription = inference(config_obj)
            transcriptions.append(cleanup(transcription))
    except:
        return None, "Wrong lang or model"

    return transcriptions, None
