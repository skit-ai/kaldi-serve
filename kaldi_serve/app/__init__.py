"""
Server component

Usage:
  kaldi-serve [--port=<port>]

Options:
  --port=<port>         Port [default: 8005]
"""

from kaldi_serve.app.server import start_server
from docopt import docopt
import sys

def main():
    args = docopt(__doc__, argv=sys.argv[1:])
    port = int(args["--port"])
    start_server("0.0.0.0", port, threaded=False)

