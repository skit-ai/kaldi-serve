/*
 * Server operations.
 */

// Include guard
#pragma once

// C++ stl includes
#include <algorithm>
#include <chrono>
#include <future>
#include <iostream>
#include <memory>
#include <string>

// gRPC inludes
#include <grpc/grpc.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>

// Kaldi includes
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

// Local includes
#include "decoder.hpp"
#include "kaldi_serve.grpc.pb.h"

class KaldiServeImpl final : public kaldi_serve::KaldiServe::Service {

  private:
    // decoder object (read-only)
    const std::shared_ptr<Decoder> decoder;

    void transcribe(const kaldi_serve::RecognitionConfig *config,
                    const kaldi_serve::RecognitionAudio *audio,
                    const std::string &uuid,
                    kaldi_serve::RecognizeResponse *recognizeResponse) {

        std::cout << "UUID: \t" << uuid << std::endl;

        std::cout << "Encoding: " << config->encoding() << std::endl;
        std::cout << "Sample Rate Hertz: " << config->sample_rate_hertz() << std::endl;
        std::cout << "Language Code: " << config->language_code() << std::endl;
        std::cout << "Max Alternatives: " << config->max_alternatives() << std::endl;
        std::cout << "Punctuation: " << config->punctuation() << std::endl;
        std::cout << "Model: " << config->model() << std::endl;

        std::cout << "NOTE: We are ignoring most of the parameters, other than the audio bytes, for now."
                  << std::endl;

        kaldi_serve::SpeechRecognitionResult *results = recognizeResponse->add_results();

        int32 n_best = config->max_alternatives();
        std::stringstream input_stream(audio->content());

        kaldi_serve::SpeechRecognitionAlternative *alternative;
        for (auto const &res : this->decoder->decode_stream(input_stream, n_best)) {
            alternative = results->add_alternatives();
            alternative->set_transcript(res.first.first);
            alternative->set_confidence(res.first.second);
        }

        return;
    }

  public:
    explicit KaldiServeImpl(const std::shared_ptr<Decoder> decoder) : decoder(decoder) {}

    grpc::Status Recognize(grpc::ServerContext *context,
                           const kaldi_serve::RecognizeRequest *recognizeRequest,
                           kaldi_serve::RecognizeResponse *recognizeResponse) override {

        // LOG REQUEST RESOLVE TIME --> START
        std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
        start = std::chrono::high_resolution_clock::now();

        kaldi_serve::RecognitionConfig config = recognizeRequest->config();
        kaldi_serve::RecognitionAudio audio = recognizeRequest->audio();
        std::string uuid = recognizeRequest->uuid();

        this->transcribe(&config, &audio, uuid, recognizeResponse);

        end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> res_duration = end - start;

        std::cout << "request resolved in: " << res_duration.count() << 's' << std::endl;
        // LOG REQUEST RESOLVE TIME --> END

        return grpc::Status::OK;
    }
};

void run_server(const std::shared_ptr<Decoder> &decoder) {
    // define a kaldi serve instance and pass the decoder
    KaldiServeImpl service(decoder);

    std::string server_address("0.0.0.0:5016");

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());

    std::cout << "Server listening on " << server_address << std::endl;
    server->Wait();
}
