// decoder-cuda.cpp - GPU Decoder Implementation

// local includes
#include "decoder.hpp"
#include "model.hpp"
#include "config.hpp"
#include "types.hpp"


namespace kaldiserve {

#if HAVE_CUDA == 1

BatchDecoder::BatchDecoder(ChainModel *const model) : model_(model) {
    if (model_->wb_info != nullptr) options.enable_word_level = true;
    if (model_->rnnlm_info != nullptr) options.enable_rnnlm = true;

    // kaldi::CuDevice::RegisterDeviceOptions(&po); // only need if using fp16 (can't access device_options_ directly)
    // kaldi::g_allocator_options // only need if need to customize cuda memory usage

    batched_decoder_config_.cuda_online_pipeline_opts.use_gpu_feature_extraction = false;
    batched_decoder_config_.cuda_online_pipeline_opts.determinize_lattice = false;

    // decoder options
    batched_decoder_config_.cuda_online_pipeline_opts.decoder_opts.default_beam = model_->model_spec.beam;
    batched_decoder_config_.cuda_online_pipeline_opts.decoder_opts.lattice_beam = model_->model_spec.lattice_beam;
    batched_decoder_config_.cuda_online_pipeline_opts.decoder_opts.max_active = model_->model_spec.max_active;

    // feature pipeline options
    batched_decoder_config_.cuda_online_pipeline_opts.feature_opts.feature_type = "mfcc";
    std::string model_dir = model_->model_spec.path;
    std::string conf_dir = join_path(model_dir, "conf");
    std::string mfcc_conf_filepath = join_path(conf_dir, "mfcc.conf");
    std::string ivector_conf_filepath = join_path(conf_dir, "ivector_extractor.conf");

    batched_decoder_config_.cuda_online_pipeline_opts.feature_opts.mfcc_config = mfcc_conf_filepath;
    batched_decoder_config_.cuda_online_pipeline_opts.feature_opts.ivector_extraction_config = ivector_conf_filepath;
    batched_decoder_config_.cuda_online_pipeline_opts.feature_opts.silence_weighting_config.silence_weight = model_->model_spec.silence_weight;

    // compute options
    batched_decoder_config_.cuda_online_pipeline_opts.compute_opts.acoustic_scale = model_->model_spec.acoustic_scale;
    batched_decoder_config_.cuda_online_pipeline_opts.compute_opts.frame_subsampling_factor = model_->model_spec.frame_subsampling_factor;

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

    if (cuda_pipeline_) {
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

    cuda_pipeline_->DecodeWithCallback(audios_[key], [&n_best, &word_level, &user_callback, this](kaldi::CompactLattice &clat) {
        utterance_results_t results;
        find_alternatives(clat, n_best, results, word_level, this->model_, this->options);
        user_callback(results);
    });
    num_tasks_submitted_++;
}

void BatchDecoder::wait_for_tasks() {
    std::cout << "#tasks submitted: " << num_tasks_submitted_ << std::endl;
    cuda_pipeline_->WaitForAllTasks();
    cudaDeviceSynchronize();
}

#endif

} // namespace kaldiserve