// Decoding graph and operations.
#pragma once

// stl includes
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

// kaldi includes
#include "base/kaldi-common.h"
#include "rnnlm/rnnlm-lattice-rescoring.h"
#include "util/common-utils.h"
#include "lat/compose-lattice-pruned.h"
#include "feat/wave-reader.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "lat/word-align-lattice.h"
#include "lat/sausages.h"
#include "nnet3/nnet-utils.h"
#include "online2/online-endpoint.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/onlinebin-util.h"
#include "util/kaldi-thread.h"

// local includes
#include "config.hpp"
#include "types.hpp"
#include "model.hpp"
#include "utils.hpp"


namespace kaldiserve {

// Forward declare class for friendship (hack for now)
class ChainModel;


class Decoder final {

  public:
    explicit Decoder(ChainModel *const model);

    ~Decoder() noexcept;

    // SETUP METHODS
    void start_decoding(const std::string &uuid="") noexcept;

    void free_decoder() noexcept;

    // STREAMING METHODS

    // decode an intermediate frame/chunk of a wav audio stream
    void decode_stream_wav_chunk(std::istream &wav_stream);

    // decode an intermediate frame/chunk of a raw headerless wav audio stream
    void decode_stream_raw_wav_chunk(std::istream &wav_stream,
                                     const float &samp_freq,
                                     const int &data_bytes);

    // NON-STREAMING METHODS

    // decodes an (independent) wav audio stream
    // internally chunks a wav audio stream and decodes them
    void decode_wav_audio(std::istream &wav_stream,
                          const float &chunk_size=1);

    // decodes an (independent) raw headerless wav audio stream
    // internally chunks a wav audio stream and decodes them
    void decode_raw_wav_audio(std::istream &wav_stream,
                              const float &samp_freq,
                              const int &data_bytes,
                              const float &chunk_size=1);

    // LATTICE DECODING METHODS

    // get the final utterances based on the compact lattice
    void get_decoded_results(const int &n_best,
                             utterance_results_t &results,
                             const bool &word_level=false,
                             const bool &bidi_streaming=false);

    DecoderOptions options{false, false};

  private:
    // decodes an intermediate wavepart
    void _decode_wave(kaldi::SubVector<kaldi::BaseFloat> &wave_part,
                      std::vector<std::pair<int32, kaldi::BaseFloat>> &delta_weights,
                      const kaldi::BaseFloat &samp_freq);

    // gets the final decoded transcripts from lattice
    void _find_alternatives(kaldi::CompactLattice &clat,
                            const std::size_t &n_best,
                            utterance_results_t &results,
                            const bool &word_level) const;

    // model vars
    ChainModel *model_;

    // decoder vars (per utterance)
    kaldi::SingleUtteranceNnet3Decoder *decoder_;
    kaldi::OnlineNnet2FeaturePipeline *feature_pipeline_;
    kaldi::OnlineSilenceWeighting *silence_weighting_;
    kaldi::OnlineIvectorExtractorAdaptationState *adaptation_state_;

    // req-specific vars
    std::string uuid_;
};


// Factory for creating decoders with shared decoding graph and model parameters
// Caches the graph and params to be able to produce decoders on demand.
class DecoderFactory final {

  public:
    ModelSpec model_spec;

    explicit DecoderFactory(const ModelSpec &model_spec);

    inline Decoder *produce() const {
        return new Decoder(model_.get());
    }

    // friendly alias for the producer method
    inline Decoder *operator()() const {
        return produce();
    }

  private:
    std::unique_ptr<ChainModel> model_;
};


// Decoder Queue for providing thread safety to multiple request handler
// threads producing and consuming decoder instances on demand.
class DecoderQueue final {

  public:
    explicit DecoderQueue(const ModelSpec &);

    DecoderQueue(const DecoderQueue &) = delete; // disable copying

    DecoderQueue &operator=(const DecoderQueue &) = delete; // disable assignment

    ~DecoderQueue();

    // friendly alias for `pop`
    inline Decoder *acquire() {
        return pop_();
    }

    // friendly alias for `push`
    inline void release(Decoder *const decoder) {
        return push_(decoder);
    }

  private:
    // Push method that supports multi-threaded thread-safe concurrency
    // pushes a decoder object onto the queue
    void push_(Decoder *const);

    // Pop method that supports multi-threaded thread-safe concurrency
    // pops a decoder object from the queue
    Decoder *pop_();

    // underlying STL "unsafe" queue for storing decoder objects
    std::queue<Decoder*> queue_;
    // custom mutex to make queue "thread-safe"
    std::mutex mutex_;
    // helper for holding mutex and notification on waiting threads when concerned resources are available
    std::condition_variable cond_;
    // factory for producing new decoders on demand
    std::unique_ptr<DecoderFactory> decoder_factory_;
};


void find_alternatives(kaldi::CompactLattice &clat,
                       const std::size_t &n_best,
                       utterance_results_t &results,
                       const bool &word_level,
                       ChainModel *const model,
                       const DecoderOptions &options);


// Find confidence by merging lm and am scores. Taken from
// https://github.com/dialogflow/asr-server/blob/master/src/OnlineDecoder.cc#L90
// NOTE: This might not be very useful for us right now. Depending on the
//       situation, we might actually want to weigh components differently.
static inline double calculate_confidence(const float &lm_score, const float &am_score, const int &n_words) noexcept {
    return std::max(0.0, std::min(1.0, -0.0001466488 * (2.388449 * lm_score + am_score) / (n_words + 1) + 0.956));
}


static inline void print_wav_info(const kaldi::WaveInfo &wave_info) noexcept {
    std::cout << "sample freq: " << wave_info.SampFreq() << ENDL
              << "sample count: " << wave_info.SampleCount() << ENDL
              << "num channels: " << wave_info.NumChannels() << ENDL
              << "reverse bytes: " << wave_info.ReverseBytes() << ENDL
              << "dat bytes: " << wave_info.DataBytes() << ENDL
              << "is streamed: " << wave_info.IsStreamed() << ENDL
              << "block align: " << wave_info.BlockAlign() << ENDL;
}


static void read_raw_wav_stream(std::istream &wav_stream,
                                const size_t &data_bytes,
                                kaldi::Matrix<kaldi::BaseFloat> &wav_data,
                                const size_t &num_channels = 1,
                                const size_t &sample_width = 2) {
    const size_t bits_per_sample = sample_width * 8;
    const size_t block_align = num_channels * sample_width;

    std::vector<char> buffer(data_bytes);
    wav_stream.read(&buffer[0], data_bytes);

    if (wav_stream.bad())
        KALDI_ERR << "WaveData: file read error";

    if (buffer.size() == 0)
        KALDI_ERR << "WaveData: empty file (no data)";

    if (buffer.size() < data_bytes) {
        KALDI_WARN << "Expected " << data_bytes << " bytes of wave data, "
                   << "but read only " << buffer.size() << " bytes. "
                   << "Truncated file?";
    }

    uint16 *data_ptr = reinterpret_cast<uint16 *>(&buffer[0]);

    // The matrix is arranged row per channel, column per sample.
    wav_data.Resize(num_channels, data_bytes / block_align);
    for (uint32 i = 0; i < wav_data.NumCols(); ++i) {
        for (uint32 j = 0; j < wav_data.NumRows(); ++j) {
            int16 k = *data_ptr++;
            wav_data(j, i) = k;
        }
    }
}

} // namespace kaldiserve