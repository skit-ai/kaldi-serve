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

// #include "helper.h"
#include "kaldi.grpc.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerReaderWriter;
using grpc::ServerWriter;
using grpc::Status;

using kaldi::Kaldi;

using kaldi::RecognizeRequest;
using kaldi::RecognizeResponse;
using kaldi::RecognitionConfig;
using kaldi::RecognitionAudio;
using kaldi::SpeechRecognitionResult;
using kaldi::SpeechRecognitionAlternative;
using kaldi::SpeechContext;

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


class KaldiImpl final : public Kaldi::Service {

    public:
        explicit KaldiImpl() {
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

void RunServer(char* models ) {
  std::string server_address("0.0.0.0:5016");
  KaldiImpl service;

  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;
  server->Wait();
}


int main(int argc, char* argv[]) {
  // Expect models to be loaded ./kaldi_server en,hi,en/bbqn
  for(int i=0; i<argc; ++i){
    std::cout << "Argument ->> " << argv[i] << std::endl;
  }
  
  RunServer(argv[1]);

  return 0;
}


// For Asynchronous gRPC server to customize threading behaviour
// class KaldiImpl final : public Kaldi::AsyncService {
// };

