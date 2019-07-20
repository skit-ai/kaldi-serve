/*
 * Server operations.
 */

// Include guard
#pragma once

#include "config.hpp"

// C++ stl includes
#include <iostream>
#include <unordered_map>
#include <memory>
#include <string>

#if DEBUG
#include <chrono>
#endif

// gRPC inludes
#include <grpc/grpc.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>

// Local includes
#include "decoder.hpp"
#include "kaldi_serve.grpc.pb.h"

// KaldiServeImpl :: Kaldi Service interface Implementation
// Defines the core server logic and request/response handlers.
// Keeps a few `Decoder` instances cached in a thread-safe
// multiple producer multiple consumer queue to handle each
// request with a different `Decoder`.
class KaldiServeImpl final : public kaldi_serve::KaldiServe::Service {

  private:
    // Map of Thread-safe Decoder MPMC Queues for diff languages/models
    std::unordered_map<model_id_t, std::unique_ptr<DecoderQueue>, model_id_hash> decoder_queue_map_;

    // Tells if a given model name and language code is available for use.
    inline bool is_model_present(const model_id_t &) const noexcept;

  public:
    // Main Constructor for Kaldi Service
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
};

KaldiServeImpl::KaldiServeImpl(const std::vector<ModelSpec> &model_specs) noexcept {
    for (auto const &model_spec : model_specs) {
        model_id_t model_id = std::make_pair(model_spec.name, model_spec.language_code);
        decoder_queue_map_[model_id] = std::make_unique<DecoderQueue>(model_spec.path, model_spec.n_decoders);
    }
}

inline bool KaldiServeImpl::is_model_present(const model_id_t &model_id) const noexcept {
    return decoder_queue_map_.find(model_id) != decoder_queue_map_.end();
}

grpc::Status KaldiServeImpl::Recognize(grpc::ServerContext *const context,
                                       const kaldi_serve::RecognizeRequest *const request,
                                       kaldi_serve::RecognizeResponse *const response) {
    kaldi_serve::RecognitionConfig config = request->config();
    int32 n_best = config.max_alternatives();
    std::string model_name = config.model();
    std::string language_code = config.language_code();
    model_id_t model_id = std::make_pair(model_name, language_code);

    if (!is_model_present(model_id)) {
        return grpc::Status(grpc::StatusCode::NOT_FOUND, "Model " + model_name + " (" + language_code + ") not found");
    }

    Decoder *decoder_ = decoder_queue_map_[model_id]->acquire();

#if DEBUG
    std::chrono::system_clock::time_point start_time;
    // LOG REQUEST RESOLVE TIME --> START (at the last request since that would be the actual latency)
    start_time = std::chrono::system_clock::now();
#endif
    kaldi_serve::RecognitionAudio audio = request->audio();
    std::stringstream input_stream(audio.content());

    kaldi_serve::SpeechRecognitionResult *sr_result = response->add_results();
    kaldi_serve::SpeechRecognitionAlternative *alternative;

    utterance_results_t k_results_;

    // TODO: take as parameter in request config
    const size_t chunk_size = 1;

    // decode speech signals in chunks
    decoder_->decode_stream_process_audio(input_stream, n_best, k_results_, chunk_size);

    // find alternatives on final `lattice` after all chunks have been processed
    for (auto const &res : k_results_) {
        alternative = sr_result->add_alternatives();
        alternative->set_transcript(res.first.first);
        alternative->set_confidence(res.first.second);
    }

    decoder_queue_map_[model_id]->release(decoder_);

#if DEBUG
    std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
    // LOG REQUEST RESOLVE TIME --> END
    auto secs = std::chrono::duration_cast<std::chrono::seconds>(
        end_time - start_time);
    std::cout << "request resolved in: " << secs.count() << 's' << ENDL;
#endif

    // return OK status when request is resolved
    return grpc::Status::OK;
}

grpc::Status KaldiServeImpl::StreamingRecognize(grpc::ServerContext *const context,
                                                grpc::ServerReader<kaldi_serve::RecognizeRequest> *const reader,
                                                kaldi_serve::RecognizeResponse *const response) {
    kaldi_serve::RecognizeRequest request_;
    reader->Read(&request_);

    // We first read the request to see if we have the correct model and
    // language to load Also assuming that the config won't change mid request
    kaldi_serve::RecognitionConfig config = request_.config();
    int32 n_best = config.max_alternatives();
    std::string model_name = config.model();
    std::string language_code = config.language_code();
    model_id_t model_id = std::make_pair(model_name, language_code);

    if (!is_model_present(model_id)) {
        return grpc::Status(grpc::StatusCode::NOT_FOUND, "Model " + model_name + " (" + language_code + ") not found");
    }

    // IMPORTANT ::
    // - Attain the lock and pop a decoder from the `free` queue
    // - Wait here until lock on queue is attained and a decoder is obtained.
    // - Each new stream gets it's own decoder instance.
    Decoder *decoder_ = decoder_queue_map_[model_id]->acquire();

    // IMPORTANT :: decoder state variables need to be statically initialized (on the stack) :: Kaldi errors out on heap
    kaldi::OnlineIvectorExtractorAdaptationState adaptation_state(decoder_->feature_info_->ivector_extractor_info);
    kaldi::OnlineNnet2FeaturePipeline feature_pipeline(*decoder_->feature_info_);
    feature_pipeline.SetAdaptationState(adaptation_state);

    kaldi::OnlineSilenceWeighting silence_weighting(decoder_->trans_model_, decoder_->feature_info_->silence_weighting_config,
                                                    decoder_->decodable_opts_.frame_subsampling_factor);
    kaldi::nnet3::DecodableNnetSimpleLoopedInfo decodable_info(decoder_->decodable_opts_, &decoder_->am_nnet_);

    kaldi::SingleUtteranceNnet3Decoder decoder(decoder_->lattice_faster_decoder_config_,
                                               decoder_->trans_model_, decodable_info, *decoder_->decode_fst_,
                                               &feature_pipeline);

#if DEBUG
    std::chrono::system_clock::time_point start_time;
#endif

    // read chunks until end of stream
    do {
#if DEBUG
        // LOG REQUEST RESOLVE TIME --> START (at the last request since that would be the actual latency)
        start_time = std::chrono::system_clock::now();
#endif
        kaldi_serve::RecognitionAudio audio = request_.audio();
        std::stringstream input_stream_chunk(audio.content());

        // decode intermediate speech signals
        // Assumption :: audio stream has already been chunked into desired length
        decoder_->decode_stream_process_chunk(feature_pipeline, silence_weighting, decoder, input_stream_chunk);
    } while (reader->Read(&request_));

    kaldi_serve::SpeechRecognitionResult *sr_result = response->add_results();
    kaldi_serve::SpeechRecognitionAlternative *alternative;

    utterance_results_t k_results_;
    decoder_->decode_stream_final(feature_pipeline, decoder, n_best, k_results_);

    // find alternatives on final `lattice` after all chunks have been processed
    for (auto const &res : k_results_) {
        alternative = sr_result->add_alternatives();
        alternative->set_transcript(res.first.first);
        alternative->set_confidence(res.first.second);
    }

    // IMPORTANT :: release the lock on the decoder and push back into `free` queue.
    // also notifies another request handler thread that a decoder is available.
    decoder_queue_map_[model_id]->release(decoder_);

#if DEBUG
    std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
    // LOG REQUEST RESOLVE TIME --> END
    auto secs = std::chrono::duration_cast<std::chrono::seconds>(
        end_time - start_time);
    std::cout << "request resolved in: " << secs.count() << 's' << ENDL;
#endif

    // return OK status when request is resolved
    return grpc::Status::OK;
}

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

VARIABLES ON WHICH SERVER RELIABILITY DEPENDS ::
    1. Length of Audio Stream (in secs)
    2. No. of chunks in the Audio Stream
    3. Time intervals between subsequent chunks of audio stream
    4. No. of Decoders in Queue
    5. Timeout for each request (chunk essentially)
    6. No. of concurrent streams being handled by the server
 */
