import subprocess
import json
from typing import Dict

def inference(config: Dict):
    script_args = [
        config["script"],
        config["wav_filename"],
        *config["args"]
    ]
    decode_process = subprocess.Popen(script_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = decode_process.communicate()
    return json.dumps(stdout.decode("utf-8"), ensure_ascii=False).encode('utf8')

