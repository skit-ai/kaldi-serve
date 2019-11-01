.PHONY: clean test

all: kaldi_serve_pb2_grpc.py

kaldi_serve_pb2_grpc.py: ../protos/kaldi_serve.proto
	poetry run python -m grpc_tools.protoc -I ../protos --python_out=./kaldi_serve --grpc_python_out=./kaldi_serve ../protos/kaldi_serve.proto
	# Hack to fix python import path issue in grpc code generation
	sed -i 's/import kaldi_serve_pb2 as kaldi__serve__pb2/import kaldi_serve.kaldi_serve_pb2 as kaldi__serve__pb2/' ./kaldi_serve/kaldi_serve_pb2_grpc.py

clean:
	rm ./kaldi_serve/kaldi_serve_pb2_grpc.py ./kaldi_serve/kaldi_serve_pb2.py

test:
	poetry run pytest
