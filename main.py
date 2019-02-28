import os
import json
import subprocess
import time
from celery import Celery
from pydub import AudioSegment
from pydub.silence import split_on_silence
from typing import Dict, List

from kaldi_serve import utils
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

    # start the process here
    print("asr run start")

    # results, error = transcribe(audio_uri, config["language_code"])
    time.sleep(15)

    # process ends
    results = [
        {
            "alternatives": [
                {
                    "transcript": "okay so what am I",
                    "confidence": 0.96096134,
                },
            ]
        },
        {
            "alternatives": [
                {
                    "transcript": "doing here",
                    "confidence": 0.96096134,
                },
            ]
        }
    ]

    # update results to redis
    job_data = get_redis_data(operation_name)
    job_data["done"] = True
    job_data["results"] = results

    set_redis_data(operation_name, job_data, REDIS_EXPIRY_TIME)


def inference(config: Dict):
    script_args = [
        config["script"],
        config["wav_filename"],
        *config["args"]
    ]
    decode_process = subprocess.Popen(script_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = decode_process.communicate()
    return json.dumps(stdout.decode("utf-8"), ensure_ascii=False)


def transcribe(audio_uri: str, lang: str, model: str='gmm', chunk: bool=True) -> (List[str], str):
    """
    Transcribe audio
    """
    try:
        wav_filename = audio_uri
        chunks = utils.get_chunks(wav_filename) if chunk else [complete_audio]
    except:
        return None, "Unable to find 'file'"

    try:
        transcriptions = []
        for i, chunk in enumerate(chunks):
            chunk_filename = wav_filename.strip(".wav") + "chunk" + str(i) + ".wav"
            chunk.export(chunk_filename, format="wav")
            config_obj = config[lang][model]
            config_obj["wav_filename"] = chunk_filename
            transcription = inference(config_obj)
            transcriptions.append(transcription)
    except:
        return None, "Wrong lang or model"

    return transcriptions, None
