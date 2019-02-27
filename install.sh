sudo apt-get update --fix-missing
sudo apt-get install -y --no-install-recommends \
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
               python-openssl \
	       ffmpeg \
	       libavcodec-extra

# Go to home directory
cd
export HOME=`pwd`

echo 'export HOME=$HOME' >> .bashrc
git clone https://gitlab.com/vernacularai/research/kaldi/ $HOME/kaldi-trunk
echo 'export KALDI_ROOT=$HOME/kaldi-trunk' >> .bashrc

cd $HOME/kaldi-trunk/tools
make -j$(nproc)

cd $HOME/kaldi-trunk/src
./configure --shared
make depend -j$(nproc)
make -j$(nproc)
cd $HOME/kaldi-trunk
git checkout gmm-hmm-tdnn

cd $HOME
curl https://pyenv.run | bash
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> .bashrc
echo 'eval "$(pyenv init -)"' >> .bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> .bashrc
source .bashrc
pyenv install 3.6.5
pyenv global 3.6.5

pip3 install poetry
cd $HOME/kaldi-serve
poetry install --no-dev
