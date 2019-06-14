# Kaldi-Serve

Server component for kaldi based ASR.

To build & run HTTP Server (8016),

```sh
python3 setup.py build
python3 setup.py install
python3 main.py
```

To build and run gRPC Server (5016),

```sh
cd kaldi
make
./kaldi_server
```