/*
 * Server operations.
 */

// Include guard
#pragma once

#include "config.hpp"

// C++ stl includes
#include <iostream>
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

struct ModelSpec {
  std::string name;
  std::string language_code;
  std::string path;
  std::size_t n_decoders = 1;
};

// KaldiServeImpl :: Kaldi Service interface Implementation
// Defines the core server logic and request/response handlers.
// Keeps a few `Decoder` instances cached in a thread-safe
// multiple producer multiple consumer queue to handle each
// request with a different `Decoder`.
class KaldiServeImpl final : public kaldi_serve::KaldiServe::Service {

  private:
    // Thread-safe Decoder MPMC Queue
    // TODO Keep a map from a tuple of model, lang to decoder_queue
    std::string model_name;
    std::string language_code;
    std::unique_ptr<DecoderQueue> decoder_queue_;

  public:
    // Main Constructor for Kaldi Service
    // Accepts the `model_dir` path and the number of decoders to cache.
    explicit KaldiServeImpl(const ModelSpec &);

    // Tell if a given model name and language code is available for use.
    bool is_model_present(const std::string &, const std::string &);

    // Request Handler RPC service
    // Accepts a stream of `RecognizeRequest` packets
    // Returns a single `RecognizeResponse` message
    grpc::Status StreamingRecognize(grpc::ServerContext *,
                                    grpc::ServerReader<kaldi_serve::RecognizeRequest> *,
                                    kaldi_serve::RecognizeResponse *) override;
};

KaldiServeImpl::KaldiServeImpl(const ModelSpec &model_spec) {
    model_name = model_spec.name;
    language_code = model_spec.language_code;
    decoder_queue_ = std::unique_ptr<DecoderQueue>(new DecoderQueue(model_spec.path, model_spec.n_decoders));
}

bool KaldiServeImpl::is_model_present(const std::string &model_name, const std::string &language_code) {
  return (this->model_name == model_name) && (this->language_code == language_code);
}

grpc::Status KaldiServeImpl::StreamingRecognize(grpc::ServerContext *context,
                                                grpc::ServerReader<kaldi_serve::RecognizeRequest> *reader,
                                                kaldi_serve::RecognizeResponse *response) {
    kaldi_serve::RecognizeRequest request_;
    reader->Read(&request_);
    // We first read the request to see if we have the correct model and
    // language to load Also assuming that the config won't change mid request
    kaldi_serve::RecognitionConfig config = request_.config();
    int32 n_best = config.max_alternatives();

    if (!is_model_present(config.model(), config.language_code())) {
      return grpc::Status(grpc::StatusCode::NOT_FOUND, "Model " + config.model() + "(" + config.language_code() + ")" + " not found");
    }

    // IMPORTANT :: attain the lock and pop a decoder from the `free` queue
    // waits here until lock on queue is attained and a decoder is obtained.
    // Each new stream gets it's own decoder instance.
    // TODO(1): set a timeout for wait and allocate a temp decoder
    // to resolve request if memory allows.
    Decoder *decoder_ = decoder_queue_->acquire();

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
        // LOG REQUEST RESOLVE TIME --> START (at the last request since that would be the actually )
        start_time = std::chrono::system_clock::now();
#endif
        kaldi_serve::RecognitionAudio audio = request_.audio();
        std::stringstream input_stream(audio.content());

        // decode intermediate speech signals
        decoder_->decode_stream_process(feature_pipeline, silence_weighting, decoder, input_stream);
    } while (reader->Read(&request_));

    kaldi_serve::SpeechRecognitionResult *results = response->add_results();
    kaldi_serve::SpeechRecognitionAlternative *alternative;

    // find alternatives on final `lattice` after all chunks have been processed
    for (auto const &res : decoder_->decode_stream_final(feature_pipeline, decoder, n_best)) {
        alternative = results->add_alternatives();
        alternative->set_transcript(res.first.first);
        alternative->set_confidence(res.first.second);
    }

    // IMPORTANT :: release the lock on the decoder and push back into `free` queue.
    // also notifies another request handler thread that a decoder is available.
    decoder_queue_->release(decoder_);

#if DEBUG
    std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
    // LOG REQUEST RESOLVE TIME --> END
    auto secs = std::chrono::duration_cast<std::chrono::seconds>(
        end_time - start_time);
    std::cout << "request resolved in: " << secs.count() << 's' << std::endl;
#endif

    // return OK status when request is resolved
    return grpc::Status::OK;
}

// Runs the Server with the Kaldi Service
void run_server(const ModelSpec &model_spec) {
    KaldiServeImpl service(model_spec);

    std::string server_address("0.0.0.0:5016");

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());

    std::cout << "kaldi-serve gRPC Streaming Server listening on " << server_address << std::endl;
    server->Wait();
}

/**
SOME NOTES:
-----------

VARIABLES ON WHICH SERVER RELIABILITY DEPENDS ::
    1. Length of Audio Stream (in secs)
    2. No. of chunks in the Audio Stream
    3. Time intervals between subsequent chunks of audio stream
    4. No. of Decoders in Queue
    5. Timeout for each request (chunk essentially)
    6. No. of concurrent streams being handled by the server

LAST BENCHMARK ::
    length of audio streams = 3s
    no. of chunks per stream = [1, 3] randomly
    time intervals between subsequent chunks = [1, 3]s randomly
    no. of decoders in queue = 60
    timeout = 80s
    concurrent requests = 600

    :: time taken per stream = 1.57 - avg :: 0.42 - min :: 1.76 - max.

    Roughly able to handle load pretty well, RAM used was ~10GB with 60 decoders in queue. CPU usage was optimum (100% on 8 cores).
 */
