FROM gcr.io/vernacular-voice-services/asr/kaldi:latest

RUN mkdir /home/app

# copy code
WORKDIR /home/app
COPY requirements.txt /home/app/requirements.txt
RUN pip3 install -r /home/app/requirements.txt

COPY . /home/app

ENV KALDI_ROOT="/home/kaldi"
ENV LD_LIBRARY_PATH="/home/kaldi/tools/openfst/lib:/home/kaldi/src/lib"
RUN python setup.py build && python setup.py install

ENV REDIS_HOST="localhost"
ENV REDIS_VIRTUAL_PORT=1
ENV MODELS_PATH="/vol/data/models"

CMD ["celery", "worker" "-A", "main.celery", "-Q", "asr", "--loglevel=info"]