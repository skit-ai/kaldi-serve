#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>

#include <grpc/grpc.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/security/server_credentials.h>

#include "kaldi_serve.grpc.pb.h"

#include "vendor/CLI11.hpp"

// Kaldi includes from here
#include "feat/wave-reader.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "nnet3/nnet-utils.h"
#include "online2/online-endpoint.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/onlinebin-util.h"
#include "util/kaldi-thread.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerReaderWriter;
using grpc::ServerWriter;
using grpc::Status;

using kaldi_serve::KaldiServe;

using kaldi_serve::RecognitionAudio;
using kaldi_serve::RecognitionConfig;
using kaldi_serve::RecognizeRequest;
using kaldi_serve::RecognizeResponse;
using kaldi_serve::SpeechContext;
using kaldi_serve::SpeechRecognitionAlternative;
using kaldi_serve::SpeechRecognitionResult;

using namespace std;
using std::chrono::system_clock;

void transcribe(const RecognitionConfig* config, const RecognitionAudio* audio,
                const string uuid, RecognizeResponse* recognizeResponse){
  std::cout << "UUID: \t" << uuid << std::endl;

  std::cout << "Encoding: " << config->encoding() << std::endl;
  std::cout << "Sample Rate Hertz: " << config->sample_rate_hertz() << std::endl;
  std::cout << "Language Code: " << config->language_code() << std::endl;
  std::cout << "Max Alternatives: " << config->max_alternatives() << std::endl;
  std::cout << "Punctuation: " << config->punctuation() << std::endl;
  std::cout << "Model: " << config->model() << std::endl;

  // std::cout << "Content: " << audio.content() << std::endl;
  // std::cout << "URI: " << audio.uri() << std::endl;

  // TDNN Decode & inference code goes here

  SpeechRecognitionResult* results = recognizeResponse->add_results();
  SpeechRecognitionAlternative* alternative = results->add_alternatives();
  alternative->set_transcript("Hi! Kaldi gRPC Server is up & running!!");
  alternative->set_confidence(1.0);

  alternative = results->add_alternatives();
  alternative->set_transcript("Please plug-in TDNN code");
  alternative->set_confidence(1.0);

  return;
}


class KaldiServeImpl final : public KaldiServe::Service {
public:
  explicit KaldiServeImpl() {
  }

  Status Recognize(ServerContext* context, const RecognizeRequest* recognizeRequest,
                   RecognizeResponse* recognizeResponse) override {
    RecognitionConfig config;
    RecognitionAudio audio;
    string uuid;

    config = recognizeRequest->config();
    audio  = recognizeRequest->audio();
    uuid   = recognizeRequest->uuid();

    transcribe(&config, &audio, uuid, recognizeResponse);

    return Status::OK;
  }
};


void RunServer(std::vector<std::string> models) {
  std::string server_address("0.0.0.0:5016");
  KaldiServeImpl service;

  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;
  server->Wait();
}


int main(int argc, char* argv[]) {
  CLI::App app{"Kaldi gRPC server"};

  std::vector<std::string> models;
  app.add_option("-m,--model", models, "Model specifier like `en/bbq`")
    ->required();

  CLI11_PARSE(app, argc, argv);

  std::cout << ":: Loading " << models.size() << " models" << std::endl;
  for (auto const& m: models) {
    std::cout << "::  - " << m << std::endl;
  }

  RunServer(models);
  return 0;
}
