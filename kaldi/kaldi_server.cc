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


class KaldiImpl final : public Kaldi::Service {

    public:
        explicit KaldiImpl() {
        }

        Status Recognize(ServerContext* context, const RecognizeRequest* recognizeRequest,
                        RecognizeResponse* recognizeResponse) override {
            // Recognize App Code
            return Status::OK;
        }
};

void RunServer(int x) {
  std::string server_address("0.0.0.0:5016");
  KaldiImpl service;

  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;
  server->Wait();
}


int main(int argc, char** argv) {
  // Expect only arg: --models=path/to/models
  RunServer(1);

  return 0;
}


// For Asynchronous gRPC server to customize threading behaviour
// class KaldiImpl final : public Kaldi::AsyncService {
// };

