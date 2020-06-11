// Decoding graph and operations.
#pragma once

#include "config.hpp"

// stl includes
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>
#include <chrono>

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
#include "model.hpp"
#include "utils.hpp"


struct Word {
  float start_time, end_time, confidence;
  std::string word;
};

// An alternative defines a single hypothesis and certain details about the
// parse (only scores for now).
struct Alternative {
  std::string transcript;
  double confidence;
  float am_score, lm_score;
  std::vector<Word> words;
};

// Options for decoder
struct DecoderOptions {
  bool enable_word_level;
  bool enable_rnnlm;
};

// Result for one continuous utterance
using utterance_results_t = std::vector<Alternative>;


// Find confidence by merging lm and am scores. Taken from
// https://github.com/dialogflow/asr-server/blob/master/src/OnlineDecoder.cc#L90
// NOTE: This might not be very useful for us right now. Depending on the
//       situation, we might actually want to weigh components differently.
inline double calculate_confidence(const float &lm_score, const float &am_score, const std::size_t &n_words) noexcept {
    return std::max(0.0, std::min(1.0, -0.0001466488 * (2.388449 * lm_score + am_score) / (n_words + 1) + 0.956));
}


inline void print_wav_info(const kaldi::WaveInfo &wave_info) noexcept {
    std::cout << "sample freq: " << wave_info.SampFreq() << ENDL
              << "sample count: " << wave_info.SampleCount() << ENDL
              << "num channels: " << wave_info.NumChannels() << ENDL
              << "reverse bytes: " << wave_info.ReverseBytes() << ENDL
              << "dat bytes: " << wave_info.DataBytes() << ENDL
              << "is streamed: " << wave_info.IsStreamed() << ENDL
              << "block align: " << wave_info.BlockAlign() << ENDL;
}


void read_raw_wav_stream(std::istream &wav_stream,
                         const size_t &data_bytes,
                         kaldi::Matrix<kaldi::BaseFloat> &wav_data) {
    constexpr size_t num_channels = 1;     // mono-channel audio
    constexpr size_t bits_per_sample = 16; // LINEAR16 PCM audio
    constexpr size_t block_align = num_channels * bits_per_sample / 8;

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


class Decoder final {

  public:
    explicit Decoder(Model *const model);

    ~Decoder() noexcept;

    // SETUP METHODS
    void start_decoding(const std::string &uuid) noexcept;

    void free_decoder() noexcept;

    // STREAMING METHODS

    // decode an intermediate frame/chunk of a wav audio stream
    void decode_stream_wav_chunk(std::istream &wav_stream);

    // decode an intermediate frame/chunk of a raw headerless wav audio stream
    void decode_stream_raw_wav_chunk(std::istream &wav_stream,
                                     const kaldi::BaseFloat& samp_freq,
                                     const size_t &data_bytes);

    // NON-STREAMING METHODS

    // decodes an (independent) wav audio stream
    // internally chunks a wav audio stream and decodes them
    void decode_wav_audio(std::istream &wav_stream,
                          const kaldi::BaseFloat &chunk_size=1);

    // decodes an (independent) raw headerless wav audio stream
    // internally chunks a wav audio stream and decodes them
    void decode_raw_wav_audio(std::istream &wav_stream,
                              const kaldi::BaseFloat &samp_freq,
                              const size_t &data_bytes,
                              const kaldi::BaseFloat &chunk_size=1);

    // LATTICE DECODING METHODS

    // get the final utterances based on the compact lattice
    void get_decoded_results(const std::size_t &n_best,
                             utterance_results_t &results,
                             const bool &word_level,
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
    Model *model_;

    // decoder vars (per utterance)
    kaldi::SingleUtteranceNnet3Decoder *decoder_;
    kaldi::OnlineNnet2FeaturePipeline *feature_pipeline_;
    kaldi::OnlineSilenceWeighting *silence_weighting_;
    kaldi::OnlineIvectorExtractorAdaptationState *adaptation_state_;

    // req-specific vars
    std::string uuid_;
};

Decoder::Decoder(Model *const model) : model_(model) {

    if (model_->wb_info_ != nullptr) options.enable_word_level = true;
    if (model_->rnnlm_info_ != nullptr) options.enable_rnnlm = true;

    // decoder vars initialization
    decoder_ = NULL;
    feature_pipeline_ = NULL;
    silence_weighting_ = NULL;
    adaptation_state_ = NULL;
}

Decoder::~Decoder() noexcept {
    free_decoder();
}

void Decoder::start_decoding(const std::string &uuid) noexcept {
    free_decoder();
    feature_pipeline_ = new kaldi::OnlineNnet2FeaturePipeline(*model_->feature_info_);
    
    if (model_->feature_info_->use_ivectors) {
        adaptation_state_ = new kaldi::OnlineIvectorExtractorAdaptationState(model_->feature_info_->ivector_extractor_info);
        feature_pipeline_->SetAdaptationState(*adaptation_state_);
        silence_weighting_ = new kaldi::OnlineSilenceWeighting(model_->trans_model_,
                                                       model_->feature_info_->silence_weighting_config,
                                                       model_->decodable_opts_.frame_subsampling_factor);
    }
    else
    {
      adaptation_state_ = NULL;
      silence_weighting_ = NULL;
    }

    decoder_ = new kaldi::SingleUtteranceNnet3Decoder(model_->lattice_faster_decoder_config_,
                                                      model_->trans_model_, *model_->decodable_info_,
                                                      *model_->decode_fst_, feature_pipeline_);
    decoder_->InitDecoding();

    uuid_ = uuid;
}

void Decoder::free_decoder() noexcept {
    if (decoder_) {
        delete decoder_;
        decoder_ = NULL;
    }
    if (adaptation_state_) {
        delete adaptation_state_;
        adaptation_state_ = NULL;
    }
    if (feature_pipeline_) {
        delete feature_pipeline_; 
        feature_pipeline_ = NULL;
    }
    if (silence_weighting_) {
        delete silence_weighting_;
        silence_weighting_ = NULL;
    }
    uuid_ = "";
}

void Decoder::decode_stream_wav_chunk(std::istream &wav_stream) {
    kaldi::WaveData wave_data;
    wave_data.Read(wav_stream);

    const kaldi::BaseFloat samp_freq = wave_data.SampFreq();

    // get the data for channel zero (if the signal is not mono, we only
    // take the first channel).
    kaldi::SubVector<kaldi::BaseFloat> wave_part(wave_data.Data(), 0);
    std::vector<std::pair<int32, kaldi::BaseFloat>> delta_weights;
    _decode_wave(wave_part, delta_weights, samp_freq);
}

void Decoder::decode_stream_raw_wav_chunk(std::istream &wav_stream,
                                          const kaldi::BaseFloat& samp_freq,
                                          const size_t &data_bytes) {
    kaldi::Matrix<kaldi::BaseFloat> wave_matrix;    
    read_raw_wav_stream(wav_stream, data_bytes, wave_matrix);

    // get the data for channel zero (if the signal is not mono, we only
    // take the first channel).
    kaldi::SubVector<kaldi::BaseFloat> wave_part(wave_matrix, 0);
    std::vector<std::pair<int32, kaldi::BaseFloat>> delta_weights;

    std::chrono::system_clock::time_point start_time;
    if (DEBUG) start_time = std::chrono::system_clock::now();

    _decode_wave(wave_part, delta_weights, samp_freq);

    if (DEBUG) {
        std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[" << timestamp_now() << "] uuid: " << uuid_ << " _decode_wave executed in: " << ms.count() << "ms" << ENDL;
    }
}

void Decoder::decode_wav_audio(std::istream &wav_stream,
                               const kaldi::BaseFloat &chunk_size) {
    kaldi::WaveData wave_data;
    wave_data.Read(wav_stream);

    // get the data for channel zero (if the signal is not mono, we only
    // take the first channel).
    kaldi::SubVector<kaldi::BaseFloat> data(wave_data.Data(), 0);
    const kaldi::BaseFloat samp_freq = wave_data.SampFreq();

    int32 chunk_length;
    if (chunk_size > 0) {
        chunk_length = int32(samp_freq * chunk_size);
        if (chunk_length == 0)
            chunk_length = 1;
    } else {
        chunk_length = std::numeric_limits<int32>::max();
    }

    int32 samp_offset = 0;
    std::vector<std::pair<int32, kaldi::BaseFloat>> delta_weights;

    while (samp_offset < data.Dim()) {
        int32 samp_remaining = data.Dim() - samp_offset;
        int32 num_samp = chunk_length < samp_remaining ? chunk_length : samp_remaining;

        kaldi::SubVector<kaldi::BaseFloat> wave_part(data, samp_offset, num_samp);
        _decode_wave(wave_part, delta_weights, samp_freq);

        samp_offset += num_samp;
    }
}

void Decoder::decode_raw_wav_audio(std::istream &wav_stream,
                                   const kaldi::BaseFloat &samp_freq,
                                   const size_t &data_bytes,
                                   const kaldi::BaseFloat &chunk_size) {
    kaldi::Matrix<kaldi::BaseFloat> wave_matrix;
    read_raw_wav_stream(wav_stream, data_bytes, wave_matrix);

    // get the data for channel zero (if the signal is not mono, we only
    // take the first channel).
    kaldi::SubVector<kaldi::BaseFloat> data(wave_matrix, 0);

    int32 chunk_length;
    if (chunk_size > 0) {
        chunk_length = int32(samp_freq * chunk_size);
        if (chunk_length == 0)
            chunk_length = 1;
    } else {
        chunk_length = std::numeric_limits<int32>::max();
    }

    int32 samp_offset = 0;
    std::vector<std::pair<int32, kaldi::BaseFloat>> delta_weights;

    while (samp_offset < data.Dim()) {
        int32 samp_remaining = data.Dim() - samp_offset;
        int32 num_samp = chunk_length < samp_remaining ? chunk_length : samp_remaining;

        kaldi::SubVector<kaldi::BaseFloat> wave_part(data, samp_offset, num_samp);
        _decode_wave(wave_part, delta_weights, samp_freq);

        samp_offset += num_samp;
    }
}

void Decoder::get_decoded_results(const std::size_t &n_best,
                                  utterance_results_t &results,
                                  const bool &word_level,
                                  const bool &bidi_streaming) {
    if (!bidi_streaming) {
        feature_pipeline_->InputFinished();
        decoder_->AdvanceDecoding();
        decoder_->FinalizeDecoding();
    }

    if (decoder_->NumFramesDecoded() == 0) {
        KALDI_WARN << "audio may be empty :: decoded no frames";
        return;
    }

    kaldi::CompactLattice clat;
    try {
        decoder_->GetLattice(true, &clat);
        _find_alternatives(clat, n_best, results, word_level);
    } catch (std::exception &e) {
        KALDI_ERR << "unexpected error during decoding lattice :: " << e.what(); 
    }
}

void Decoder::_decode_wave(kaldi::SubVector<kaldi::BaseFloat> &wave_part,
                           std::vector<std::pair<int32, kaldi::BaseFloat>> &delta_weights,
                           const kaldi::BaseFloat &samp_freq) {

    feature_pipeline_->AcceptWaveform(samp_freq, wave_part);

    std::chrono::system_clock::time_point start_time;
    if (DEBUG) start_time = std::chrono::system_clock::now();
    
    if (model_->feature_info_->use_ivectors) {
        if (silence_weighting_->Active() && feature_pipeline_->IvectorFeature() != NULL) {
            silence_weighting_->ComputeCurrentTraceback(decoder_->Decoder());
            silence_weighting_->GetDeltaWeights(feature_pipeline_->NumFramesReady(),
                                                &delta_weights);

            if (DEBUG) {
                std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                std::cout << "[" << timestamp_now() << "] uuid: " << uuid_ << " silence weighting done in: " << ms.count() << "ms" << ENDL;
            }

            if (DEBUG) start_time = std::chrono::system_clock::now();

            feature_pipeline_->IvectorFeature()->UpdateFrameWeights(delta_weights);

            if (DEBUG) {
                std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                std::cout << "[" << timestamp_now() << "] uuid: " << uuid_ << " ivector frame weights updated in: " << ms.count() << "ms" << ENDL;
            }

            if (DEBUG) start_time = std::chrono::system_clock::now();
        }
    }
    
    decoder_->AdvanceDecoding();

    if (DEBUG) {
        std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "[" << timestamp_now() << "] uuid: " << uuid_ << " decoder advance done in: " << ms.count() << "ms" << ENDL;
    }
}

void Decoder::_find_alternatives(kaldi::CompactLattice &clat,
                                 const std::size_t &n_best,
                                 utterance_results_t &results,
                                 const bool &word_level) const {
    if (clat.NumStates() == 0) {
        KALDI_LOG << "Empty lattice.";
    }

    if (options.enable_rnnlm) {
        std::unique_ptr<kaldi::rnnlm::KaldiRnnlmDeterministicFst> lm_to_add_orig = 
            std::make_unique<kaldi::rnnlm::KaldiRnnlmDeterministicFst>(model_->model_spec.max_ngram_order, *model_->rnnlm_info_);

        fst::DeterministicOnDemandFst<fst::StdArc> *lm_to_add =
            new fst::ScaleDeterministicOnDemandFst(model_->rnnlm_weight_, lm_to_add_orig.get());

        // Before composing with the LM FST, we scale the lattice weights
        // by the inverse of "lm_scale".  We'll later scale by "lm_scale".
        // We do it this way so we can determinize and it will give the
        // right effect (taking the "best path" through the LM) regardless
        // of the sign of lm_scale.
        if (model_->decodable_opts_.acoustic_scale != 1.0) {
            fst::ScaleLattice(fst::AcousticLatticeScale(model_->decodable_opts_.acoustic_scale), &clat);
        }
        kaldi::TopSortCompactLatticeIfNeeded(&clat);

        std::unique_ptr<fst::ScaleDeterministicOnDemandFst> lm_to_subtract_det_scale =
            std::make_unique<fst::ScaleDeterministicOnDemandFst>(-model_->rnnlm_weight_, model_->lm_to_subtract_det_backoff_.get());
        fst::ComposeDeterministicOnDemandFst<fst::StdArc> combined_lms(lm_to_subtract_det_scale.get(), lm_to_add);

        // Composes lattice with language model.
        kaldi::CompactLattice composed_clat;
        kaldi::ComposeCompactLatticePruned(model_->compose_opts_, clat,
                                           &combined_lms, &composed_clat);

        if (composed_clat.NumStates() == 0) {
            // Something went wrong.  A warning will already have been printed.
            KALDI_WARN << "Empty lattice after RNNLM rescoring.";
        } else {
            clat = composed_clat;
        }
        
        delete lm_to_add;
    }

    auto lat = std::make_unique<kaldi::Lattice>();
    fst::ConvertLattice(clat, lat.get());

    kaldi::Lattice nbest_lat;
    std::vector<kaldi::Lattice> nbest_lats;

    fst::ShortestPath(*lat, &nbest_lat, n_best);
    fst::ConvertNbestToVector(nbest_lat, &nbest_lats);

    if (nbest_lats.empty()) {
        KALDI_WARN << "no N-best entries";
        return;
    }

    for (auto const &l : nbest_lats) {
        // NOTE: Check why int32s specifically are used here
        std::vector<int32> input_ids;
        std::vector<int32> word_ids;
        std::vector<std::string> word_strings;
        std::string sentence;

        kaldi::LatticeWeight weight;
        fst::GetLinearSymbolSequence(l, &input_ids, &word_ids, &weight);

        for (auto const &wid : word_ids) {
            word_strings.push_back(model_->word_syms_->Find(wid));
        }
        string_join(word_strings, " ", sentence);

        Alternative alt;
        alt.transcript = sentence;
        alt.lm_score = float(weight.Value1());
        alt.am_score = float(weight.Value2());
        alt.confidence = calculate_confidence(alt.lm_score, alt.am_score, word_ids.size());

        results.push_back(alt);
    }

    if (!(options.enable_word_level && word_level))
      return;

    kaldi::CompactLattice aligned_clat;
    kaldi::BaseFloat max_expand = 0.0;
    int32 max_states;

    if (max_expand > 0)
        max_states = 1000 + max_expand * clat.NumStates();
    else
        max_states = 0;

    bool ok = kaldi::WordAlignLattice(clat, model_->trans_model_, *model_->wb_info_, max_states, &aligned_clat);

    if (!ok) {
        if (aligned_clat.Start() != fst::kNoStateId) {
            KALDI_WARN << "Outputting partial lattice";
            kaldi::TopSortCompactLatticeIfNeeded(&aligned_clat);
            ok = true;
        } else {
            KALDI_WARN << "Empty aligned lattice, producing no output.";
        }
    } else {
        if (aligned_clat.Start() == fst::kNoStateId) {
            KALDI_WARN << "Lattice was empty";
            ok = false;
        } else {
            kaldi::TopSortCompactLatticeIfNeeded(&aligned_clat);
        }
    }

    std::vector<Word> words;

    // compute confidences and times only if alignment was ok
    if (ok) {
        kaldi::BaseFloat frame_shift = 0.01;
        kaldi::BaseFloat lm_scale = 1.0;
        kaldi::MinimumBayesRiskOptions mbr_opts;
        mbr_opts.decode_mbr = false;

        fst::ScaleLattice(fst::LatticeScale(lm_scale, model_->decodable_opts_.acoustic_scale), &aligned_clat);
        auto mbr = std::make_unique<kaldi::MinimumBayesRisk>(aligned_clat, mbr_opts);

        const std::vector<kaldi::BaseFloat> &conf = mbr->GetOneBestConfidences();
        const std::vector<int32> &best_words = mbr->GetOneBest();
        const std::vector<std::pair<kaldi::BaseFloat, kaldi::BaseFloat>> &times = mbr->GetOneBestTimes();

        KALDI_ASSERT(conf.size() == best_words.size() && best_words.size() == times.size());

        for (size_t i = 0; i < best_words.size(); i++) {
            KALDI_ASSERT(best_words[i] != 0 || mbr_opts.print_silence); // Should not have epsilons.

            Word word;
            kaldi::BaseFloat time_unit = frame_shift * model_->decodable_opts_.frame_subsampling_factor;
            word.start_time = times[i].first * time_unit;
            word.end_time = times[i].second * time_unit;
            word.word = model_->word_syms_->Find(best_words[i]); // lookup word in SymbolTable
            word.confidence = conf[i];

            words.push_back(word);
        }
    }

    if (!results.empty() and !words.empty()) {
        results[0].words = words;
    }
}

// Factory for creating decoders with shared decoding graph and model parameters
// Caches the graph and params to be able to produce decoders on demand.
class DecoderFactory final {
  private:
    std::unique_ptr<Model> model_;

  public:
    ModelSpec model_spec;

    explicit DecoderFactory(const ModelSpec &model_spec);

    inline Decoder *produce() const;

    // friendly alias for the producer method
    inline Decoder *operator()() const;
};

DecoderFactory::DecoderFactory(const ModelSpec &model_spec) : model_spec(model_spec) {
    model_ = std::make_unique<Model>(model_spec);
}

inline Decoder *DecoderFactory::produce() const {
    return new Decoder(model_.get());
}

inline Decoder *DecoderFactory::operator()() const {
    return produce();
}

// Decoder Queue for providing thread safety to multiple request handler
// threads producing and consuming decoder instances on demand.
class DecoderQueue final {

  private:
    // underlying STL "unsafe" queue for storing decoder objects
    std::queue<Decoder *> queue_;
    // custom mutex to make queue "thread-safe"
    std::mutex mutex_;
    // helper for holding mutex and notification on waiting threads when concerned resources are available
    std::condition_variable cond_;
    // factory for producing new decoders on demand
    std::unique_ptr<DecoderFactory> decoder_factory_;

    // Push method that supports multi-threaded thread-safe concurrency
    // pushes a decoder object onto the queue
    void push_(Decoder *const);

    // Pop method that supports multi-threaded thread-safe concurrency
    // pops a decoder object from the queue
    Decoder *pop_();

  public:
    explicit DecoderQueue(const ModelSpec &);

    DecoderQueue(const DecoderQueue &) = delete; // disable copying

    DecoderQueue &operator=(const DecoderQueue &) = delete; // disable assignment

    ~DecoderQueue();

    // friendly alias for `pop`
    inline Decoder *acquire();

    // friendly alias for `push`
    inline void release(Decoder *const);
};

DecoderQueue::DecoderQueue(const ModelSpec &model_spec) {
    std::cout << ":: Loading model from " << model_spec.path << ENDL;

    std::chrono::system_clock::time_point start_time;
    if (DEBUG) {
        // LOG MODELS LOAD TIME --> START
        start_time = std::chrono::system_clock::now();
    }
    decoder_factory_ = std::unique_ptr<DecoderFactory>(new DecoderFactory(model_spec));
    for (size_t i = 0; i < model_spec.n_decoders; i++) {
        queue_.push(decoder_factory_->produce());
    }

    if (DEBUG) {
        std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
        // LOG MODELS LOAD TIME --> END
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << ":: Decoder models concurrent queue init in: " << ms.count() << "ms" << ENDL;
    }
}

DecoderQueue::~DecoderQueue() {
    while (!queue_.empty()) {
        auto decoder = queue_.front();
        queue_.pop();
        delete decoder;
    }
}

void DecoderQueue::push_(Decoder *const item) {
    std::unique_lock<std::mutex> mlock(mutex_);
    queue_.push(item);
    mlock.unlock();
    cond_.notify_one(); // condition var notifies another suspended thread (help up in `pop`)
}

Decoder *DecoderQueue::pop_() {
    std::unique_lock<std::mutex> mlock(mutex_);
    // waits until a decoder object is available
    while (queue_.empty()) {
        // suspends current thread execution and awaits condition notification
        cond_.wait(mlock);
    }
    auto item = queue_.front();
    queue_.pop();
    return item;
}

inline Decoder *DecoderQueue::acquire() {
    return pop_();
}

inline void DecoderQueue::release(Decoder *const decoder) {
    return push_(decoder);
}
