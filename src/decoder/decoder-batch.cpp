// decoder-cuda.cpp - GPU Decoder Implementation

// local includes
#include "decoder.hpp"

namespace kaldiserve {

#if HAVE_CUDA == 1

BatchDecoder::BatchDecoder(ChainModel const* model) {
    // Multi-threaded CPU and batched GPU decoder
    BatchedThreadedNnet3CudaPipeline2Config batched_decoder_config;
    CuDevice::RegisterDeviceOptions(&po);
    RegisterCuAllocatorOptions(&po);
    batched_decoder_config.Register(&po);

    g_cuda_allocator.SetOptions(g_allocator_options);
    CuDevice::Instantiate().SelectGpuId("yes");
    CuDevice::Instantiate().AllowMultithreading();

    cuda_pipeline = kaldi::cuda_decoder::BatchedThreadedNnet3CudaPipeline2(
        batched_decoder_config_, *decode_fst, am_nnet, trans_model);
}

BatchDecoder::add_task(const std::string &uuid, std::istream &wav_stream) {

}

BatchDecoder::decode_with_callback(std::function<void>) {
    
}

#endif

} // namespace kaldiserve