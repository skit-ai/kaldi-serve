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
    Decoder *decoder_;

  public:
    KaldiServeImpl(Decoder *);

    grpc::Status Recognize(grpc::ServerContext *,
                           grpc::ServerReader<kaldi_serve::RecognizeRequest> *,
                           kaldi_serve::RecognizeResponse *) override;
};

KaldiServeImpl::KaldiServeImpl(Decoder *decoder) : decoder_(decoder) {}

grpc::Status KaldiServeImpl::Recognize(grpc::ServerContext *context,
                                       grpc::ServerReader<kaldi_serve::RecognizeRequest> *reader,
                                       kaldi_serve::RecognizeResponse *response) {

    kaldi::OnlineIvectorExtractorAdaptationState *adaptation_state;
    kaldi::OnlineNnet2FeaturePipeline *feature_pipeline;
    kaldi::nnet3::DecodableNnetSimpleLoopedInfo *decodable_info;
    kaldi::OnlineSilenceWeighting *silence_weighting;
    kaldi::SingleUtteranceNnet3Decoder *decoder;

    kaldi_serve::RecognizeRequest request_;

    std::chrono::system_clock::time_point start_time;
    decoder_->decode_stream_initialize(adaptation_state, feature_pipeline, decodable_info, silence_weighting, decoder);

    while (reader->Read(&request_)) {
        // LOG REQUEST RESOLVE TIME --> START (at the last request since that would be the actually )
        start_time = std::chrono::system_clock::now();

        kaldi_serve::RecognitionAudio audio = request_.audio();

        std::stringstream input_stream(audio.content());
        decoder_->decode_stream_process(feature_pipeline, silence_weighting, decoder, input_stream);
    }
    kaldi_serve::RecognitionConfig config = request_.config();
    std::string uuid = request_.uuid();

    int32 n_best = config.max_alternatives();
    kaldi_serve::SpeechRecognitionResult *results = response->add_results();
    kaldi_serve::SpeechRecognitionAlternative *alternative;

    for (auto const &res : decoder_->decode_stream_final(feature_pipeline, decoder, n_best)) {
        alternative = results->add_alternatives();
        alternative->set_transcript(res.first.first);
        alternative->set_confidence(res.first.second);
    }

    decoder_->cleanup(adaptation_state, feature_pipeline, decodable_info, silence_weighting, decoder);
    std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
    // LOG REQUEST RESOLVE TIME --> END

    auto secs = std::chrono::duration_cast<std::chrono::seconds>(
        end_time - start_time);
    std::cout << "request resolved in: " << secs.count() << 's' << std::endl;
}

void run_server(Decoder* decoder) {
    // define a kaldi serve instance and pass the decoder
    KaldiServeImpl service(decoder);

    std::string server_address("0.0.0.0:5016");

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());

    std::cout << "kaldi-serve gRPC Streaming Server listening on " << server_address << std::endl;
    server->Wait();
}
