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

// An alignment tells start and end time (consider total audio to be of length 1) of a word
using alignment_t = std::tuple<std::string, float, float>;

// Result for one continuous utterance
using utterance_results_t = std::vector<std::pair<alternative_t, std::vector<alignment_t>>>;

// Find confidence by merging lm and am scores. Taken from
// https://github.com/dialogflow/asr-server/blob/master/src/OnlineDecoder.cc#L90
// NOTE: This might not be very useful for us right now. Depending on the
//       situation, we might actually want to weigh components differently.
inline double calculate_confidence(const float &lm_score, const float &am_score, const std::size_t &n_words) {
    return std::max(0.0, std::min(1.0, -0.0001466488 * (2.388449 * lm_score + am_score) / (n_words + 1) + 0.956));
}

// Computes n-best alternative from lattice. Output symbols are converted to words
// based on word-syms.
void find_alternatives(const fst::SymbolTable *word_syms,
                       const kaldi::CompactLattice &clat,
                       const std::size_t &n_best,
                       utterance_results_t &results) {
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
    } else {
        for (auto const &l : nbest_lats) {
            kaldi::LatticeWeight weight;

            // NOTE: Check why int32s specifically are used here
            std::vector<int32> input_ids;
            std::vector<int32> word_ids;
            fst::GetLinearSymbolSequence(l, &input_ids, &word_ids, &weight);

            std::vector<std::string> words;
            for (auto const &wid : word_ids) {
                words.push_back(word_syms->Find(wid));
            }

            // An alignment is a tuple like (word, start, end). Note that the start
            // and end time at the moment are not based on any unit and represent
            // relative values depending on the total number of input symbols. This,
            // when multiplied into audio lengths should give actual second values.
            std::vector<alignment_t> alignments;
            bool in_word = false;
            if (words.size() > 0) {
                size_t n_input_tokens = input_ids.size();
                // HACK: We assume inputs below 3 to be whitespaces. This is not 'the'
                //       right way since we haven't looked into the exact set of input
                //       symbols.
                float start, end;
                int word_index = 0;
                for (auto i = 0; i < n_input_tokens; i++) {
                    if (input_ids[i] < 3) {
                        if (in_word) {
                            // Exited a word
                            end = (i - 1) / (float)n_input_tokens;
                            alignments.push_back(std::make_tuple(words[word_index], start, end));
                            word_index += 1;
                        }
                        in_word = false;
                    } else {
                        if (!in_word) {
                            // Entered a new word
                            start = (i - 1) / (float)n_input_tokens;
                        }
                        in_word = true;
                    }
                }
            }
            std::string sentence;
            string_join(words, " ", sentence);

            alternative_t alt = std::make_pair(sentence, calculate_confidence(float(weight.Value1()),
                                                                              float(weight.Value2()),
                                                                              word_ids.size()));
            results.push_back(std::make_pair(alt, alignments));
        }
    }
}

class Decoder {

  private:
    fst::SymbolTable *word_syms_;

  public:
    fst::Fst<fst::StdArc> *const decode_fst_;
    kaldi::nnet3::AmNnetSimple am_nnet_;
    kaldi::TransitionModel trans_model_;

    kaldi::OnlineNnet2FeaturePipelineConfig feature_opts_;
    kaldi::OnlineNnet2FeaturePipelineInfo *feature_info_;

    kaldi::LatticeFasterDecoderConfig lattice_faster_decoder_config_;
    kaldi::nnet3::NnetSimpleLoopedComputationOptions decodable_opts_;

    explicit Decoder(const kaldi::BaseFloat &, const std::size_t &,
                     const std::size_t &, const kaldi::BaseFloat &,
                     const kaldi::BaseFloat &, const std::size_t &,
                     const std::string &, const std::string &,
                     const std::string &, const std::string &,
                     fst::Fst<fst::StdArc> *const);

    ~Decoder();

    // Decoding processes

    // decode intermediate frames of wav streams
    void decode_stream_process(kaldi::OnlineNnet2FeaturePipeline &,
                               kaldi::OnlineSilenceWeighting &,
                               kaldi::SingleUtteranceNnet3Decoder &,
                               std::istream &) const;

    // get the final utterances based on the compact lattice
    utterance_results_t decode_stream_final(kaldi::OnlineNnet2FeaturePipeline &,
                                            kaldi::SingleUtteranceNnet3Decoder &,
                                            const std::size_t &) const;
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
                 fst::Fst<fst::StdArc> *const decode_fst)
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

        word_syms_ = NULL;
        if (word_syms_filepath != "" && !(word_syms_ = fst::SymbolTable::ReadText(word_syms_filepath))) {
            KALDI_ERR << "Could not read symbol table from file " << word_syms_filepath;
        }
        feature_info_ = new kaldi::OnlineNnet2FeaturePipelineInfo(feature_opts_);

    } catch (const std::exception &e) {
        KALDI_ERR << e.what();
    }
}

Decoder::~Decoder() {
    delete word_syms_;
    delete feature_info_;
}

void Decoder::decode_stream_process(kaldi::OnlineNnet2FeaturePipeline &feature_pipeline,
                                    kaldi::OnlineSilenceWeighting &silence_weighting,
                                    kaldi::SingleUtteranceNnet3Decoder &decoder,
                                    std::istream &wav_stream) const {
    kaldi::WaveData wave_data;
    wave_data.Read(wav_stream);

    // get the data for channel zero (if the signal is not mono, we only
    // take the first channel).
    kaldi::SubVector<kaldi::BaseFloat> data(wave_data.Data(), 0);
    kaldi::BaseFloat samp_freq = wave_data.SampFreq();

    int32 chunk_length;
    kaldi::BaseFloat chunk_length_secs = 1;
    if (chunk_length_secs > 0) {
        chunk_length = int32(samp_freq * chunk_length_secs);
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
}

utterance_results_t Decoder::decode_stream_final(kaldi::OnlineNnet2FeaturePipeline &feature_pipeline,
                                                 kaldi::SingleUtteranceNnet3Decoder &decoder,
                                                 const std::size_t &n_best) const {
    feature_pipeline.InputFinished();
    decoder.FinalizeDecoding();

    kaldi::CompactLattice clat;
    utterance_results_t results;
    try {
        decoder.GetLattice(true, &clat);
        find_alternatives(word_syms_, clat, n_best, results);
    } catch (const std::exception &e) {
        std::cout << "ERROR :: client timed out" << std::endl;
    }
    return results;
}

// Factory for creating decoders with shared decoding graph and model parameters
// Caches the graph and params to be able to produce uniform decoders later in queue.
class DecoderFactory {

  private:
    fst::Fst<fst::StdArc> *const decode_fst_;

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
                            const std::string &);

    ~DecoderFactory();

    // Producer method for the Factory.
    // Does the actual work :: produces a new Decoder object
    // using the shared config and returns a pointer to it.
    inline Decoder *produce() const;

    // friendly alias for the producer method
    inline Decoder *operator()() const;
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
                               const std::string &ie_conf_filepath)
    : decode_fst_(fst::ReadFstKaldiGeneric(hclg_filepath)),
      beam_(beam), max_active_(max_active), min_active_(min_active), lattice_beam_(lattice_beam),
      acoustic_scale_(acoustic_scale), frame_subsampling_factor_(frame_subsampling_factor),
      word_syms_filepath_(word_syms_filepath), model_filepath_(model_filepath),
      mfcc_conf_filepath_(mfcc_conf_filepath), ie_conf_filepath_(ie_conf_filepath) {}

DecoderFactory::~DecoderFactory() {
    delete decode_fst_;
}

inline Decoder *DecoderFactory::produce() const {
    return new Decoder(beam_, max_active_, min_active_, lattice_beam_,
                       acoustic_scale_, frame_subsampling_factor_,
                       word_syms_filepath_, model_filepath_,
                       mfcc_conf_filepath_, ie_conf_filepath_,
                       decode_fst_);
}

inline Decoder *DecoderFactory::operator()() const {
    return produce();
}

// Decoder Queue for providing thread safety to multiple request handler
// threads producing and consuming decoder instances on demand.
class DecoderQueue {

  private:
    // underlying STL "unsafe" queue for holding decoders
    std::queue<Decoder *> queue_;
    // custom mutex to make queue "thread-safe"
    std::mutex mutex_;
    // helper for holding mutex and notification of waiting threads when concerned resources are available
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

    ~DecoderQueue();

    // friendly alias for `pop`
    inline Decoder *acquire();

    // friendly alias for `push`
    inline void release(Decoder *const);
};

DecoderQueue::DecoderQueue(const std::string &model_dir, const size_t &n) {
    std::cout << ":: Loading model from " << model_dir << std::endl;

    // TODO: Better organize a kaldi model for distribution
    std::string hclg_filepath = model_dir + "/tree_a_sp/graph/HCLG.fst";
    std::string words_filepath = model_dir + "/tree_a_sp/graph/words.txt";
    std::string model_filepath = model_dir + "/tdnn1g_sp_online/final.mdl";
    std::string mfcc_conf_filepath = model_dir + "/tdnn1g_sp_online/conf/mfcc.conf";
    std::string ivec_conf_filepath = model_dir + "/tdnn1g_sp_online/conf/ivector_extractor.conf";

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
    std::cout << ":: Decoder models concurrent queue init in: " << secs.count() << 's' << std::endl;
#endif
}

DecoderQueue::~DecoderQueue() {
    while (!queue_.empty()) {
        Decoder *decoder = queue_.front();
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
    Decoder *item = queue_.front();
    // pops it from queue
    queue_.pop();
    return item;
}

inline Decoder *DecoderQueue::acquire() {
    return pop_();
}

inline void DecoderQueue::release(Decoder *const decoder) {
    push_(decoder);
}