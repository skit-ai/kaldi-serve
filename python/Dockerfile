FROM vernacularai/kaldi-serve:latest

# install python3.6.5 through pyenv
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
        libtcmalloc-minimal4 \
        make build-essential libssl-dev zlib1g-dev \
        libbz2-dev libreadline-dev libsqlite3-dev \
        wget curl llvm libncurses5-dev libncursesw5-dev \
        xz-utils tk-dev libffi-dev liblzma-dev \
        python-openssl git

RUN curl https://pyenv.run | bash
RUN echo 'export PATH="~/.pyenv/bin:$PATH"' >> ~/.bashrc && \
    echo 'eval "$(pyenv init -)"' >> ~/.bashrc && \
    echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
RUN bash -c "source ~/.bashrc && pyenv install 3.6.5 && pyenv global 3.6.5"

# build python module using the C++ shared library
WORKDIR /root/kaldi-serve
COPY . .

RUN bash -c "source ~/.bashrc && cd build && \
             cmake .. -DBUILD_SHARED_LIBS=OFF -DBUILD_PYTHON_MODULE=ON -DBUILD_PYBIND11=ON -DPYTHON_EXECUTABLE=\$(pyenv which python) && \
             make -j$(nproc) VERBOSE=1"

RUN cp build/python/kaldiserve_pybind*.so python/kaldiserve/
RUN bash -c "source ~/.bashrc && cd python && pip install . -U"

ENV LD_PRELOAD="/opt/intel/mkl/lib/intel64/libmkl_rt.so:/usr/lib/libtcmalloc_minimal.so.4"

WORKDIR /root

#cleanup
RUN rm -rf kaldi-serve