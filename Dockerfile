FROM library/ubuntu:16.04

RUN apt-get update --fix-missing \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      build-essential \
      git \
      subversion \
      zlib1g-dev \
      automake \
      autoconf \
      unzip \
      wget \
      curl \
      libtool \
      libatlas3-base \
      python \
      python3 \
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

RUN git clone https://gitlab.com/vernacularai/research/kaldi/ /home/sujay/kaldi-trunk
RUN cd /home/sujay/kaldi-trunk/tools
RUN make -j$(nproc)

RUN cd /home/sujay/kaldi-trunk/src
RUN ./configure --shared
RUN make depend -j$(nproc)
RUN make -j$(nproc)
RUN cd /home/sujay/kaldi-trunk
RUN git checkout gmm-hmm-tdnn

RUN cd /home/sujay/
RUN curl https://pyenv.run | bash
RUN echo 'export PATH="/home/sujay/.pyenv/bin:$PATH"' >> .bashrc
RUN echo 'eval "$(pyenv init -)"' >> .bashrc
RUN echo 'eval "$(pyenv virtualenv-init -)"' >> .bashrc
RUN source .bashrc
RUN pyenv install 3.6.5
RUN pyenv global 3.6.5

RUN pip3 install poetry
RUN mkdir /home/sujay/kaldi-serve
COPY . /home/sujay/kaldi-serve
WORKDIR /home/sujay/kaldi-serve
RUN poetry install --no-dev
CMD ["poetry", "run", "kaldi-serve"]