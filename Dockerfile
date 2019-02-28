FROM python:3.6-slim

RUN apt-get update \
 && apt-get install -y \
      ca-certificates \
      build-essential \
      git \
      subversion \
      zlib1g-dev \
      automake \
      autoconf \
      unzip \
      wget \
      libtool \
      libatlas3-base \
      sox \
      libssl-dev \
      libbz2-dev \
      libreadline-dev \
      libsqlite3-dev \
      llvm \
      libncurses5-dev \
      libncursesw5-dev \
      xz-utils \
      tk-dev \
      libffi-dev \
      liblzma-dev \
      python-openssl

RUN mkdir /home/app

# build kaldi
WORKDIR /home
RUN git clone https://gitlab.com/vernacularai/research/kaldi.git
WORKDIR /home/kaldi/tools
RUN make -j$(nproc)

WORKDIR /home/kaldi/src
RUN ./configure --shared && make depend -j$(nproc) && make -j$(nproc)
RUN git checkout gmm-hmm-tdnn

# copy code
WORKDIR /home/app
COPY requirements.txt /home/app/requirements.txt
RUN pip3 install -r /home/app/requirements.txt

COPY . /home/app

ENV REDIS_HOST="localhost"
ENV REDIS_VIRTUAL_PORT=1

CMD ["celery", "worker" "-A", "main.celery", "-Q", "asr", "--loglevel=info"]