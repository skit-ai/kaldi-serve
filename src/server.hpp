// server.hpp - Server Interface
#pragma once

// stl includes
#include <iostream>
#include <unordered_map>
#include <memory>
#include <string>

// gRPC inludes
#include <grpc/grpc.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>

// local includes
#include "config.hpp"
#include "decoder.hpp"
#include "kaldi_serve.grpc.pb.h"


// KaldiServeImpl ::
// Defines the core server logic and request/response handlers.
// Keeps `Decoder` instances cached in a thread-safe
// multiple producer multiple consumer queue to handle each
// request with a separate `Decoder`.
class KaldiServeImpl final : public kaldi_serve::KaldiServe::Service {

  private:
    // Map of Thread-safe Decoder MPMC Queues for diff languages/models
    std::unordered_map<model_id_t, std::unique_ptr<DecoderQueue>, model_id_hash> decoder_queue_map_;

    // Tells if a given model name and language code is available for use.
    inline bool is_model_present(const model_id_t &) const noexcept;

  public:
    explicit KaldiServeImpl(const std::vector<ModelSpec> &) noexcept;

    // Non-Streaming Request Handler RPC service
    // Accepts a single `RecognizeRequest` message
    // Returns a single `RecognizeResponse` message
    grpc::Status Recognize(grpc::ServerContext *const,
                           const kaldi_serve::RecognizeRequest *const,
                           kaldi_serve::RecognizeResponse *const) override;

    // Streaming Request Handler RPC service
    // Accepts a stream of `RecognizeRequest` messages
    // Returns a single `RecognizeResponse` message
    grpc::Status StreamingRecognize(grpc::ServerContext *const,
                                    grpc::ServerReader<kaldi_serve::RecognizeRequest> *const,
                                    kaldi_serve::RecognizeResponse *const) override;

    // Bidirectional Streaming Request Handler RPC service
    // Accepts a stream of `RecognizeRequest` messages
    // Returns a stream of `RecognizeResponse` messages
    grpc::Status BidiStreamingRecognize(grpc::ServerContext *const,
                                        grpc::ServerReaderWriter<kaldi_serve::RecognizeResponse, kaldi_serve::RecognizeRequest>*) override;
};


// Runs the Server with the Kaldi Service
void run_server(const std::vector<ModelSpec> &model_specs) {
    KaldiServeImpl service(model_specs);

    std::string server_address("0.0.0.0:5016");

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());

    std::cout << "kaldi-serve gRPC Streaming Server listening on " << server_address << ENDL;
    server->Wait();
}


/**
NOTES:
------

VARIABLES THAT INFLUENCE SERVER RELIABILITY ::
    1. Length of Audio Stream (in secs)
    2. No. of chunks in the Audio Stream
    3. Time intervals between subsequent chunks of audio stream
    4. No. of Decoders in Queue
    5. Timeout for each request (chunk essentially)
    6. No. of concurrent streams being handled by the server
 */
