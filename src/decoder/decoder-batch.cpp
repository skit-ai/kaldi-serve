// decoder-cuda.cpp - GPU Decoder Implementation

// local includes
#include "decoder.hpp"
#include "model.hpp"
#include "config.hpp"
#include "types.hpp"


namespace kaldiserve {

#if HAVE_CUDA == 1

BatchDecoder::BatchDecoder(ChainModel const* model) : model_(model) {
    ParseOptions po(usage);
    kaldi::CuDevice::RegisterDeviceOptions(&po);
    kaldi::RegisterCuAllocatorOptions(&po);
    batched_decoder_config_.Register(&po);

    const char *argv[] = {
        "decoder-batch.cpp",
        std::string("").c_str(),
        NULL
    };

    po.Read((sizeof(argv)/sizeof(argv[0])) - 1, argv);

    cuda_pipeline_ = NULL;
}

BatchDecoder::~BatchDecoder() {
    free_decoder();
}

void BatchDecoder::start_decoding() {
    kaldi::g_cuda_allocator.SetOptions(kaldi::g_allocator_options);
    kaldi::CuDevice::Instantiate().SelectGpuId("yes");
    kaldi::CuDevice::Instantiate().AllowMultithreading();

    cuda_pipeline_ = new kaldi::cuda_decoder::BatchedThreadedNnet3CudaPipeline2(
        batched_decoder_config_, *model_->decode_fst, model_->am_nnet, model_->trans_model);
}

void BatchDecoder::free_decoder() {
    audios_.clear();
    num_tasks_submitted_ = 0;

    if (cuda_pipeline) {
        delete cuda_pipeline_;
        cuda_pipeline_ = NULL;
    }
}

void BatchDecoder::decode_with_callback(std::istream &wav_stream,
                                        const int &n_best,
                                        const bool &word_level,
                                        const std::string &key,
                                        std::function<void(const utterance_results_t &results)> &user_callback) {
    
    audios_[key] = std::shared_ptr<kaldi::WaveData>();
    audios_[key]->Read(wav_stream);

    cuda_pipeline_.DecodeWithCallback(audios_[key], [&n_best, &word_level, &user_callback, &model_](kaldi::CompactLattice &clat) {
        utterance_results_t results;
        find_alternatives(clat, n_best, results, word_level, model_);
        user_callback(results);
    });
    num_tasks_submitted_++;
}

void BatchDecoder::wait_for_tasks() {
    std::cout << "#tasks submitted: " << num_tasks_submitted_ << std::endl;
    cuda_pipeline_.WaitForAllTasks();
    cudaDeviceSynchronize();
}

#endif

} // namespace kaldiserve