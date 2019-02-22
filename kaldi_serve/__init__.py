__version__ = '0.1.0'

from flask import Flask
import os

app = Flask("kaldi-serve")

home_dir = os.environ.get("HOME") or "/home/sujay"

app.config['UPLOAD_FOLDER'] = os.path.join(home_dir, "temp")
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
