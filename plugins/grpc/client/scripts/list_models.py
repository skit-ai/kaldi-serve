"""Script for listing all active/loaded models on the server."""
import traceback

from pprint import pprint
from kaldi_serve import KaldiServeClient


def list_models(client):
    try:
        response = client.list_models()
    except Exception as e:
        traceback.print_exc()
        print(f'error: {str(e)}')

    models = list(map(lambda model: {"name": model.name, "language": model.language_code}, response.models))
    pprint(models)


if __name__ == "__main__":
    client = KaldiServeClient()
    list_models(client)