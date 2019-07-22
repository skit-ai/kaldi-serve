/*
 * Decoding graph and operations.
 */

// Include guard
#pragma once

#include "config.hpp"

// C++ stl includes
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#if DEBUG
#include <chrono>
#endif

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
#include "utils.hpp"

// An alternative is a pair of string (hypothesis) and it's confidence
using alternative_t = std::pair<std::string, double>;

// Result for one continuous utterance
using utterance_results_t = std::vector<alternative_t>;

// Find confidence by merging lm and am scores. Taken from
// https://github.com/dialogflow/asr-server/blob/master/src/OnlineDecoder.cc#L90
// NOTE: This might not be very useful for us right now. Depending on the
//       situation, we might actually want to weigh components differently.
inline void calculate_confidence(const float &lm_score, const float &am_score, const std::size_t &n_words, double &confidence) noexcept {
    confidence = std::max(0.0, std::min(1.0, -0.0001466488 * (2.388449 * lm_score + am_score) / (n_words + 1) + 0.956));
}

// Computes n-best alternative from lattice. Output symbols are converted to words
// based on word-syms.
void find_alternatives(const fst::SymbolTable *word_syms,
                       const kaldi::CompactLattice &clat,
                       const std::size_t &n_best,
                       utterance_results_t &results) noexcept {
    if (clat.NumStates() == 0) {
        KALDI_LOG << "Empty lattice.";
    }

    kaldi::Lattice *lat = new kaldi::Lattice();
    fst::ConvertLattice(clat, lat);

    kaldi::Lattice nbest_lat;
    std::vector<kaldi::Lattice> nbest_lats;
    fst::ShortestPath(*lat, &nbest_lat, n_best);
    fst::ConvertNbestToVector(nbest_lat, &nbest_lats);

    if (nbest_lats.empty()) {
        KALDI_WARN << "no N-best entries";
        return;
    }

    // NOTE: Check why int32s specifically are used here
    std::vector<int32> input_ids;
    std::vector<int32> word_ids;
    std::vector<std::string> words;
    std::string sentence;

    for (auto const &l : nbest_lats) {
        kaldi::LatticeWeight weight;
        fst::GetLinearSymbolSequence(l, &input_ids, &word_ids, &weight);

        for (auto const &wid : word_ids) {
            words.push_back(word_syms->Find(wid));
        }
        string_join(words, " ", sentence);

        double confidence;
        calculate_confidence(float(weight.Value1()),
                             float(weight.Value2()),
                             word_ids.size(),
                             confidence);
        results.push_back(std::make_pair(sentence, confidence));

        input_ids.clear();
        word_ids.clear();
        words.clear();
        sentence.clear();
    }
}

class Decoder final {

  private:
    std::unique_ptr<fst::SymbolTable> word_syms_;

  public:
    fst::Fst<fst::StdArc> *const decode_fst_;
    mutable kaldi::nnet3::AmNnetSimple am_nnet_; // TODO: check why kaldi decodable_info needs a non-const ref of am_net model
    kaldi::TransitionModel trans_model_;

    kaldi::OnlineNnet2FeaturePipelineConfig feature_opts_;
    std::unique_ptr<kaldi::OnlineNnet2FeaturePipelineInfo> feature_info_;

    kaldi::LatticeFasterDecoderConfig lattice_faster_decoder_config_;
    kaldi::nnet3::NnetSimpleLoopedComputationOptions decodable_opts_;

    explicit Decoder(const kaldi::BaseFloat &, const std::size_t &,
                     const std::size_t &, const kaldi::BaseFloat &,
                     const kaldi::BaseFloat &, const std::size_t &,
                     const std::string &, const std::string &,
                     const std::string &, const std::string &,
                     fst::Fst<fst::StdArc> *const) noexcept;

    // Decoding processes

    // decode an intermediate frame/chunk of a wav audio stream
    void decode_stream_process_chunk(kaldi::OnlineNnet2FeaturePipeline &,
                                     kaldi::OnlineSilenceWeighting &,
                                     kaldi::SingleUtteranceNnet3Decoder &,
                                     std::istream &) const;

    // chunk a wav audio stream and decode the frames/chunks
    void decode_stream_process_audio(std::istream &,
                                     const size_t &,
                                     utterance_results_t &,
                                     const kaldi::BaseFloat & = 1) const;

    // get the final utterances based on the compact lattice
    void decode_stream_final(kaldi::OnlineNnet2FeaturePipeline &,
                             kaldi::SingleUtteranceNnet3Decoder &,
                             const std::size_t &,
                             utterance_results_t &) const;
};

Decoder::Decoder(const kaldi::BaseFloat &beam,
                 const std::size_t &max_active,
                 const std::size_t &min_active,
                 const kaldi::BaseFloat &lattice_beam,
                 const kaldi::BaseFloat &acoustic_scale,
                 const std::size_t &frame_subsampling_factor,
                 const std::string &word_syms_filepath,
                 const std::string &model_filepath,
                 const std::string &mfcc_conf_filepath,
                 const std::string &ie_conf_filepath,
                 fst::Fst<fst::StdArc> *const decode_fst) noexcept
    : decode_fst_(decode_fst) {

    try {
        feature_opts_.mfcc_config = mfcc_conf_filepath;
        feature_opts_.ivector_extraction_config = ie_conf_filepath;
        lattice_faster_decoder_config_.max_active = max_active;
        lattice_faster_decoder_config_.min_active = min_active;
        lattice_faster_decoder_config_.beam = beam;
        lattice_faster_decoder_config_.lattice_beam = lattice_beam;
        decodable_opts_.acoustic_scale = acoustic_scale;
        decodable_opts_.frame_subsampling_factor = frame_subsampling_factor;

        // IMPORTANT :: DO NOT REMOVE CURLY BRACES (some memory dealloc issue arises if done so)
        {
            bool binary;
            kaldi::Input ki(model_filepath, &binary);

            trans_model_.Read(ki.Stream(), binary);
            am_nnet_.Read(ki.Stream(), binary);

            kaldi::nnet3::SetBatchnormTestMode(true, &(am_nnet_.GetNnet()));
            kaldi::nnet3::SetDropoutTestMode(true, &(am_nnet_.GetNnet()));
            kaldi::nnet3::CollapseModel(kaldi::nnet3::CollapseModelConfig(), &(am_nnet_.GetNnet()));
        }

        if (word_syms_filepath != "" && !(word_syms_ = std::unique_ptr<fst::SymbolTable>(fst::SymbolTable::ReadText(word_syms_filepath)))) {
            KALDI_ERR << "Could not read symbol table from file " << word_syms_filepath;
        }
        feature_info_ = std::make_unique<kaldi::OnlineNnet2FeaturePipelineInfo>(feature_opts_);

    } catch (const std::exception &e) {
        KALDI_ERR << e.what();
    }
}

void Decoder::decode_stream_process_chunk(kaldi::OnlineNnet2FeaturePipeline &feature_pipeline,
                                          kaldi::OnlineSilenceWeighting &silence_weighting,
                                          kaldi::SingleUtteranceNnet3Decoder &decoder,
                                          std::istream &wav_stream) const {
    kaldi::WaveData wave_data;
    wave_data.Read(wav_stream);

    // get the data for channel zero (if the signal is not mono, we only
    // take the first channel).
    kaldi::SubVector<kaldi::BaseFloat> wave_part(wave_data.Data(), 0);
    kaldi::BaseFloat samp_freq = wave_data.SampFreq();
    std::vector<std::pair<int32, kaldi::BaseFloat>> delta_weights;

    feature_pipeline.AcceptWaveform(samp_freq, wave_part);

    if (silence_weighting.Active() && feature_pipeline.IvectorFeature() != NULL) {
        silence_weighting.ComputeCurrentTraceback(decoder.Decoder());
        silence_weighting.GetDeltaWeights(feature_pipeline.NumFramesReady(),
                                          &delta_weights);
        feature_pipeline.IvectorFeature()->UpdateFrameWeights(delta_weights);
    }
    decoder.AdvanceDecoding();
}

void Decoder::decode_stream_process_audio(std::istream &wav_stream,
                                          const size_t &n_best,
                                          utterance_results_t &results,
                                          const kaldi::BaseFloat &chunk_size) const {
    // IMPORTANT :: decoder state variables need to be statically initialized (on the stack) :: Kaldi errors out on heap
    kaldi::OnlineIvectorExtractorAdaptationState adaptation_state(feature_info_->ivector_extractor_info);
    kaldi::OnlineNnet2FeaturePipeline feature_pipeline(*feature_info_);
    feature_pipeline.SetAdaptationState(adaptation_state);

    kaldi::OnlineSilenceWeighting silence_weighting(trans_model_, feature_info_->silence_weighting_config,
                                                    decodable_opts_.frame_subsampling_factor);
    kaldi::nnet3::DecodableNnetSimpleLoopedInfo decodable_info(decodable_opts_, &am_nnet_);

    kaldi::SingleUtteranceNnet3Decoder decoder(lattice_faster_decoder_config_,
                                               trans_model_, decodable_info, *decode_fst_,
                                               &feature_pipeline);

    kaldi::WaveData wave_data;
    wave_data.Read(wav_stream);

    // get the data for channel zero (if the signal is not mono, we only
    // take the first channel).
    kaldi::SubVector<kaldi::BaseFloat> data(wave_data.Data(), 0);
    kaldi::BaseFloat samp_freq = wave_data.SampFreq();

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
        feature_pipeline.AcceptWaveform(samp_freq, wave_part);

        if (silence_weighting.Active() && feature_pipeline.IvectorFeature() != NULL) {
            silence_weighting.ComputeCurrentTraceback(decoder.Decoder());
            silence_weighting.GetDeltaWeights(feature_pipeline.NumFramesReady(),
                                              &delta_weights);
            feature_pipeline.IvectorFeature()->UpdateFrameWeights(delta_weights);
        }
        decoder.AdvanceDecoding();
        samp_offset += num_samp;
    }

    decode_stream_final(feature_pipeline, decoder, n_best, results);
}

void Decoder::decode_stream_final(kaldi::OnlineNnet2FeaturePipeline &feature_pipeline,
                                  kaldi::SingleUtteranceNnet3Decoder &decoder,
                                  const std::size_t &n_best,
                                  utterance_results_t &results) const {
    feature_pipeline.InputFinished();
    decoder.FinalizeDecoding();

    kaldi::CompactLattice clat;
    try {
        decoder.GetLattice(true, &clat);
        find_alternatives(word_syms_.get(), clat, n_best, results);
    } catch (const std::exception &e) {
        std::cout << "ERROR :: client timed out" << ENDL;
    }
}

// Factory for creating decoders with shared decoding graph and model parameters
// Caches the graph and params to be able to produce uniform decoders later in queue.
class DecoderFactory final {

  private:
    const std::unique_ptr<fst::Fst<fst::StdArc>> decode_fst_;

    const kaldi::BaseFloat beam_;
    const std::size_t max_active_;
    const std::size_t min_active_;
    const kaldi::BaseFloat lattice_beam_;
    const kaldi::BaseFloat acoustic_scale_;
    const std::size_t frame_subsampling_factor_;
    const std::string word_syms_filepath_;
    const std::string model_filepath_;
    const std::string mfcc_conf_filepath_;
    const std::string ie_conf_filepath_;

  public:
    // Constructor for DecoderFactory
    // Accepts an HCLG filepath and other decoder config parameters
    // to share across all decoders produced by factory.
    explicit DecoderFactory(const std::string &,
                            const kaldi::BaseFloat &,
                            const std::size_t &,
                            const std::size_t &,
                            const kaldi::BaseFloat &,
                            const kaldi::BaseFloat &,
                            const std::size_t &,
                            const std::string &,
                            const std::string &,
                            const std::string &,
                            const std::string &) noexcept;

    // Producer method for the Factory.
    // Does the actual work :: produces a new Decoder object
    // using the shared config and returns a pointer to it.
    inline Decoder *produce() const noexcept;

    // friendly alias for the producer method
    inline Decoder *operator()() const noexcept;
};

DecoderFactory::DecoderFactory(const std::string &hclg_filepath,
                               const kaldi::BaseFloat &beam,
                               const std::size_t &max_active,
                               const std::size_t &min_active,
                               const kaldi::BaseFloat &lattice_beam,
                               const kaldi::BaseFloat &acoustic_scale,
                               const std::size_t &frame_subsampling_factor,
                               const std::string &word_syms_filepath,
                               const std::string &model_filepath,
                               const std::string &mfcc_conf_filepath,
                               const std::string &ie_conf_filepath) noexcept
    : decode_fst_(fst::ReadFstKaldiGeneric(hclg_filepath)),
      beam_(beam), max_active_(max_active), min_active_(min_active), lattice_beam_(lattice_beam),
      acoustic_scale_(acoustic_scale), frame_subsampling_factor_(frame_subsampling_factor),
      word_syms_filepath_(word_syms_filepath), model_filepath_(model_filepath),
      mfcc_conf_filepath_(mfcc_conf_filepath), ie_conf_filepath_(ie_conf_filepath) {}

inline Decoder *DecoderFactory::produce() const noexcept {
    return new Decoder(beam_, max_active_, min_active_, lattice_beam_,
                       acoustic_scale_, frame_subsampling_factor_,
                       word_syms_filepath_, model_filepath_,
                       mfcc_conf_filepath_, ie_conf_filepath_,
                       decode_fst_.get());
}

inline Decoder *DecoderFactory::operator()() const noexcept {
    return produce();
}

// Decoder Queue for providing thread safety to multiple request handler
// threads producing and consuming decoder instances on demand.
class DecoderQueue final {

  private:
    // underlying STL "unsafe" queue for holding decoders
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
    explicit DecoderQueue(const std::string &, const size_t &);

    DecoderQueue(const DecoderQueue &) = delete; // disable copying

    DecoderQueue &operator=(const DecoderQueue &) = delete; // disable assignment

    ~DecoderQueue() noexcept;

    // friendly alias for `pop`
    inline Decoder *acquire();

    // friendly alias for `push`
    inline void release(Decoder *const);
};

DecoderQueue::DecoderQueue(const std::string &model_dir, const size_t &n) {
    std::cout << ":: Loading model from " << model_dir << ENDL;

    // TODO: Better organize a kaldi model for distribution
#if DEBUG
    std::string hclg_filepath = model_dir + "/tree_a_sp/graph/HCLG.fst";
    std::string words_filepath = model_dir + "/tree_a_sp/graph/words.txt";
    std::string model_filepath = model_dir + "/tdnn1g_sp_online/final.mdl";
    std::string mfcc_conf_filepath = model_dir + "/tdnn1g_sp_online/conf/mfcc.conf";
    std::string ivec_conf_filepath = model_dir + "/tdnn1g_sp_online/conf/ivector_extractor.conf";
#else
    std::string hclg_filepath = model_dir + "/HCLG.fst";
    std::string words_filepath = model_dir + "/words.txt";
    std::string model_filepath = model_dir + "/final.mdl";
    std::string mfcc_conf_filepath = model_dir + "/mfcc.conf";
    std::string ivec_conf_filepath = model_dir + "/ivector_extractor.conf";
#endif

#if DEBUG
    // LOG MODELS LOAD TIME --> START
    std::chrono::system_clock::time_point start_time = std::chrono::system_clock::now();
#endif
    decoder_factory_ = std::unique_ptr<DecoderFactory>(new DecoderFactory(hclg_filepath, 13.0, 7000, 200,
                                                                          6.0, 1.0, 3, words_filepath,
                                                                          model_filepath, mfcc_conf_filepath,
                                                                          ivec_conf_filepath));
    for (size_t i = 0; i < n; i++) {
        queue_.push(decoder_factory_->produce());
    }

#if DEBUG
    std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
    // LOG MODELS LOAD TIME --> END

    auto secs = std::chrono::duration_cast<std::chrono::seconds>(
        end_time - start_time);
    std::cout << ":: Decoder models concurrent queue init in: " << secs.count() << 's' << ENDL;
#endif
}

DecoderQueue::~DecoderQueue() noexcept {
    while (!queue_.empty()) {
        auto decoder = queue_.front();
        queue_.pop();
        delete decoder;
    }
}

void DecoderQueue::push_(Decoder *const item) {
    // acquires a lock on the queue (mutex)
    std::unique_lock<std::mutex> mlock(mutex_);
    // pushes item into queue
    queue_.push(item);
    // releases the lock on queue
    mlock.unlock();
    // condition var notifies another suspended thread (in `pop`)
    cond_.notify_one();
}

Decoder *DecoderQueue::pop_() {
    // acquires a lock on the queue (mutex)
    std::unique_lock<std::mutex> mlock(mutex_);
    // waits until queue is not empty
    while (queue_.empty()) {
        // suspends current thread execution (as well as lock on queue)
        // and waits for condition var notification
        cond_.wait(mlock);
    }
    // obtains an item from front of queue
    auto item = queue_.front();
    // pops it from queue
    queue_.pop();
    return item;
}

inline Decoder *DecoderQueue::acquire() {
    return pop_();
}

inline void DecoderQueue::release(Decoder *const decoder) {
    return push_(decoder);
}
