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

class KaldiServeImpl final : public kaldi_serve::KaldiServe::Service {

  private:
    // thread-safe decoder queue
    std::unique_ptr<DecoderQueue> decoder_queue_;

  public:
    KaldiServeImpl(const std::string &, const int &);

    grpc::Status Recognize(grpc::ServerContext *,
                           grpc::ServerReader<kaldi_serve::RecognizeRequest> *,
                           kaldi_serve::RecognizeResponse *) override;
};

KaldiServeImpl::KaldiServeImpl(const std::string &model_dir, const int &n) {
    decoder_queue_ = std::unique_ptr<DecoderQueue>(new DecoderQueue(model_dir, n));
}

grpc::Status KaldiServeImpl::Recognize(grpc::ServerContext *context,
                                       grpc::ServerReader<kaldi_serve::RecognizeRequest> *reader,
                                       kaldi_serve::RecognizeResponse *response) {

    // IMPORTANT :: attain the lock and pop a decoder from the `free` queue
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

    kaldi_serve::RecognizeRequest request_;
    std::chrono::system_clock::time_point start_time;

    while (reader->Read(&request_)) {
#if DEBUG
        // LOG REQUEST RESOLVE TIME --> START (at the last request since that would be the actually )
        start_time = std::chrono::system_clock::now();
#endif
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

    // IMPORTANT :: release the lock on the decoder and push back into `free` queue.
    decoder_queue_->release(decoder_);
#ifdef DEBUG
    std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
    // LOG REQUEST RESOLVE TIME --> END

    auto secs = std::chrono::duration_cast<std::chrono::seconds>(
        end_time - start_time);
    std::cout << "request resolved in: " << secs.count() << 's' << std::endl;
#endif
    return grpc::Status::OK;
}

void run_server(const std::string &model_dir, const int &n) {
    // define a kaldi serve instance
    KaldiServeImpl service(model_dir, n);

    std::string server_address("0.0.0.0:5016");

    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());

    std::cout << "kaldi-serve gRPC Streaming Server listening on " << server_address << std::endl;
    server->Wait();
}
