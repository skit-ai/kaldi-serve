import json
import os

import redis

# Storage connection details (Redis)
redis_remote_host = os.environ.get('REDIS_HOST', 'localhost')
redis_remote_port = 6379

redis_connection = None


def set_redis_connection():
    global redis_connection
    if not redis_connection:
        redis_connection = redis.StrictRedis(
            host=redis_remote_host,
            port=redis_remote_port,
            db=0,
            charset="utf-8",
            decode_responses=True
        )


def get_redis_data(key: str):
    set_redis_connection()

    val = redis_connection.get(f"service:asr:{key}")
    return json.loads(val, encoding='utf-8') if val else None


def set_redis_data(key: str, data, expiry_time=None) -> bool:
    """
    :param key:
    :param data:
    :param expiry_time: in seconds, None for persistent storage
    :return:
    """
    set_redis_connection()
    if data:
        if expiry_time:
            if expiry_time >= 1:
                return redis_connection.set(f"service:asr:{key}", json.dumps(data), ex=int(expiry_time))
        else:
            return redis_connection.set(f"service:asr:{key}", json.dumps(data))

    return False
