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

#if DEBUG
#include <chrono>
#endif

// kaldi includes
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
#include "utils.hpp"

struct Word {
    float startTime, endTime, confidence;
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

  private:
    std::unique_ptr<fst::SymbolTable> word_syms_;
    kaldi::WordBoundaryInfo* wb_info_;

    // internal decoding process
    void _decode_wave(kaldi::OnlineNnet2FeaturePipeline &,
                      kaldi::OnlineSilenceWeighting &,
                      kaldi::SingleUtteranceNnet3Decoder &,
                      kaldi::SubVector<kaldi::BaseFloat> &,
                      std::vector<std::pair<int32, kaldi::BaseFloat>> &,
                      const kaldi::BaseFloat &) const;

    void _find_alternatives(const kaldi::CompactLattice &clat,
                            const std::size_t &n_best,
                            utterance_results_t &results) const;

  public:
    fst::Fst<fst::StdArc> *const decode_fst_;
    mutable kaldi::nnet3::AmNnetSimple am_nnet_; // TODO: check why kaldi decodable_info needs a non-const ref of am_net model
    kaldi::TransitionModel trans_model_;

    std::unique_ptr<kaldi::OnlineNnet2FeaturePipelineInfo> feature_info_;

    kaldi::LatticeFasterDecoderConfig lattice_faster_decoder_config_;
    kaldi::nnet3::NnetSimpleLoopedComputationOptions decodable_opts_;

    explicit Decoder(const kaldi::BaseFloat &, const std::size_t &,
                     const std::size_t &, const kaldi::BaseFloat &,
                     const kaldi::BaseFloat &, const std::size_t &,
                     const std::string &, fst::Fst<fst::StdArc> *const) noexcept;

    // decode an intermediate frame/chunk of a wav audio stream
    void decode_stream_wav_chunk(kaldi::OnlineNnet2FeaturePipeline &,
                                 kaldi::OnlineSilenceWeighting &,
                                 kaldi::SingleUtteranceNnet3Decoder &,
                                 std::istream &) const;

    // decode an intermediate frame/chunk of a raw headerless wav audio stream
    void decode_stream_raw_wav_chunk(kaldi::OnlineNnet2FeaturePipeline &,
                                     kaldi::OnlineSilenceWeighting &,
                                     kaldi::SingleUtteranceNnet3Decoder &,
                                     std::istream &,
                                     const size_t &) const;

    // decodes an (independent) wav audio stream
    // internally chunks a wav audio stream and decodes them
    void decode_wav_audio(std::istream &,
                          const size_t &,
                          utterance_results_t &,
                          const kaldi::BaseFloat & = 1) const;

    // decodes an (independent) raw headerless wav audio stream
    // internally chunks a wav audio stream and decodes them
    void decode_raw_wav_audio(std::istream &,
                              const size_t &,
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
                 const std::size_t &min_active,
                 const std::size_t &max_active,
                 const kaldi::BaseFloat &lattice_beam,
                 const kaldi::BaseFloat &acoustic_scale,
                 const std::size_t &frame_subsampling_factor,
                 const std::string &model_dir,
                 fst::Fst<fst::StdArc> *const decode_fst) noexcept
    : decode_fst_(decode_fst) {

    try {
        lattice_faster_decoder_config_.min_active = min_active;
        lattice_faster_decoder_config_.max_active = max_active;
        lattice_faster_decoder_config_.beam = beam;
        lattice_faster_decoder_config_.lattice_beam = lattice_beam;
        decodable_opts_.acoustic_scale = acoustic_scale;
        decodable_opts_.frame_subsampling_factor = frame_subsampling_factor;

        std::string word_syms_filepath = join_path(model_dir, "words.txt");
        std::string word_boundary_filepath = join_path(model_dir, "word_boundary.int");
        std::string model_filepath = join_path(model_dir, "final.mdl");
        std::string conf_dir = join_path(model_dir, "conf");
        std::string mfcc_conf_filepath = join_path(conf_dir, "mfcc.conf");
        std::string ivector_conf_filepath = join_path(conf_dir, "ivector_extractor.conf");

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

        kaldi::WordBoundaryInfoNewOpts opts;
        wb_info_ = new kaldi::WordBoundaryInfo(opts, word_boundary_filepath);

        feature_info_ = std::make_unique<kaldi::OnlineNnet2FeaturePipelineInfo>();
        feature_info_->feature_type = "mfcc";
        kaldi::ReadConfigFromFile(mfcc_conf_filepath, &(feature_info_->mfcc_opts));

        feature_info_->use_ivectors = true;
        kaldi::OnlineIvectorExtractionConfig ivector_extraction_opts;
        kaldi::ReadConfigFromFile(ivector_conf_filepath, &ivector_extraction_opts);

        // Expand paths if relative provided. We use model_dir as the base in
        // such cases.
        ivector_extraction_opts.lda_mat_rxfilename = expand_relative_path(ivector_extraction_opts.lda_mat_rxfilename, model_dir);
        ivector_extraction_opts.global_cmvn_stats_rxfilename = expand_relative_path(ivector_extraction_opts.global_cmvn_stats_rxfilename, model_dir);
        ivector_extraction_opts.diag_ubm_rxfilename = expand_relative_path(ivector_extraction_opts.diag_ubm_rxfilename, model_dir);
        ivector_extraction_opts.ivector_extractor_rxfilename = expand_relative_path(ivector_extraction_opts.ivector_extractor_rxfilename, model_dir);
        ivector_extraction_opts.cmvn_config_rxfilename = expand_relative_path(ivector_extraction_opts.cmvn_config_rxfilename, model_dir);
        ivector_extraction_opts.splice_config_rxfilename = expand_relative_path(ivector_extraction_opts.splice_config_rxfilename, model_dir);

        feature_info_->ivector_extractor_info.Init(ivector_extraction_opts);
    }
    catch (const std::exception &e) {
        KALDI_ERR << e.what();
    }
}

void Decoder::_decode_wave(kaldi::OnlineNnet2FeaturePipeline &feature_pipeline,
                           kaldi::OnlineSilenceWeighting &silence_weighting,
                           kaldi::SingleUtteranceNnet3Decoder &decoder,
                           kaldi::SubVector<kaldi::BaseFloat> &wave_part,
                           std::vector<std::pair<int32, kaldi::BaseFloat>> &delta_weights,
                           const kaldi::BaseFloat &samp_freq) const {

    feature_pipeline.AcceptWaveform(samp_freq, wave_part);

    if (silence_weighting.Active() && feature_pipeline.IvectorFeature() != NULL) {
        silence_weighting.ComputeCurrentTraceback(decoder.Decoder());
        silence_weighting.GetDeltaWeights(feature_pipeline.NumFramesReady(),
                                          &delta_weights);
        feature_pipeline.IvectorFeature()->UpdateFrameWeights(delta_weights);
    }
    decoder.AdvanceDecoding();
}


// Computes n-best alternative from lattice. Output symbols are converted to words
// based on word-syms.
void Decoder::_find_alternatives(const kaldi::CompactLattice &clat,
                                 const std::size_t &n_best,
                                 utterance_results_t &results) const {
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

    for (auto const &l : nbest_lats) {
        // NOTE: Check why int32s specifically are used here
        std::vector<int32> input_ids;
        std::vector<int32> word_ids;
        std::vector<std::string> words;
        std::string sentence;

        kaldi::LatticeWeight weight;
        fst::GetLinearSymbolSequence(l, &input_ids, &word_ids, &weight);

        for (auto const &wid : word_ids) {
            words.push_back(word_syms_->Find(wid));
        }
        string_join(words, " ", sentence);

        Alternative alt;
        alt.transcript = sentence;
        alt.lm_score = float(weight.Value1());
        alt.am_score = float(weight.Value2());
        alt.confidence = calculate_confidence(alt.lm_score, alt.am_score, word_ids.size());

        results.push_back(alt);
    }

    kaldi::CompactLattice aligned_clat;
    kaldi::BaseFloat max_expand = 0.0;
    int32 max_states;

    if (max_expand > 0) max_states = 1000 + max_expand * clat.NumStates();
    else max_states = 0;
    
    bool ok = kaldi::WordAlignLattice(clat, trans_model_, *wb_info_, max_states, &aligned_clat);

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

        fst::ScaleLattice(fst::LatticeScale(lm_scale, decodable_opts_.acoustic_scale), &aligned_clat);
        kaldi::MinimumBayesRisk *mbr = new kaldi::MinimumBayesRisk(aligned_clat, mbr_opts);

        const std::vector<kaldi::BaseFloat> &conf = mbr->GetOneBestConfidences();
        const std::vector<int32> &best_words = mbr->GetOneBest();
        const std::vector<std::pair<kaldi::BaseFloat, kaldi::BaseFloat>> &times = mbr->GetOneBestTimes();

        KALDI_ASSERT(conf.size() == best_words.size() && best_words.size() == times.size());

        for (size_t i = 0; i < best_words.size(); i++) {
            KALDI_ASSERT(best_words[i] != 0 || mbr_opts.print_silence); // Should not have epsilons.

            Word word;
            word.startTime = frame_shift * times[i].first;
            word.endTime = frame_shift * times[i].second;
            word.word = word_syms_->Find(best_words[i]); // lookup word in SymbolTable
            word.confidence = conf[i];
            
            words.push_back(word);
        }
    }

    if (!results.empty() and !words.empty()) {
        results[0].words = words;
    }
}


void Decoder::decode_stream_wav_chunk(kaldi::OnlineNnet2FeaturePipeline &feature_pipeline,
                                      kaldi::OnlineSilenceWeighting &silence_weighting,
                                      kaldi::SingleUtteranceNnet3Decoder &decoder,
                                      std::istream &wav_stream) const {

    kaldi::WaveData wave_data;
    wave_data.Read(wav_stream);

    const kaldi::BaseFloat samp_freq = wave_data.SampFreq();

    // get the data for channel zero (if the signal is not mono, we only
    // take the first channel).
    kaldi::SubVector<kaldi::BaseFloat> wave_part(wave_data.Data(), 0);
    std::vector<std::pair<int32, kaldi::BaseFloat>> delta_weights;
    _decode_wave(feature_pipeline, silence_weighting, decoder, wave_part, delta_weights, samp_freq);
}

void Decoder::decode_stream_raw_wav_chunk(kaldi::OnlineNnet2FeaturePipeline &feature_pipeline,
                                          kaldi::OnlineSilenceWeighting &silence_weighting,
                                          kaldi::SingleUtteranceNnet3Decoder &decoder,
                                          std::istream &wav_stream,
                                          const size_t &data_bytes) const {

    kaldi::Matrix<kaldi::BaseFloat> wave_matrix;
    read_raw_wav_stream(wav_stream, data_bytes, wave_matrix);

    constexpr kaldi::BaseFloat samp_freq = 8000;

    // get the data for channel zero (if the signal is not mono, we only
    // take the first channel).
    kaldi::SubVector<kaldi::BaseFloat> wave_part(wave_matrix, 0);
    std::vector<std::pair<int32, kaldi::BaseFloat>> delta_weights;
    _decode_wave(feature_pipeline, silence_weighting, decoder, wave_part, delta_weights, samp_freq);
}

void Decoder::decode_wav_audio(std::istream &wav_stream,
                               const size_t &n_best,
                               utterance_results_t &results,
                               const kaldi::BaseFloat &chunk_size) const {
    // decoder state variables need to be statically initialized
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
        _decode_wave(feature_pipeline, silence_weighting, decoder, wave_part, delta_weights, samp_freq);

        samp_offset += num_samp;
    }

    decode_stream_final(feature_pipeline, decoder, n_best, results);
}

void Decoder::decode_raw_wav_audio(std::istream &wav_stream,
                                   const size_t &data_bytes,
                                   const size_t &n_best,
                                   utterance_results_t &results,
                                   const kaldi::BaseFloat &chunk_size) const {
    // decoder state variables need to be statically initialized
    kaldi::OnlineIvectorExtractorAdaptationState adaptation_state(feature_info_->ivector_extractor_info);
    kaldi::OnlineNnet2FeaturePipeline feature_pipeline(*feature_info_);
    feature_pipeline.SetAdaptationState(adaptation_state);

    kaldi::OnlineSilenceWeighting silence_weighting(trans_model_, feature_info_->silence_weighting_config,
                                                    decodable_opts_.frame_subsampling_factor);
    kaldi::nnet3::DecodableNnetSimpleLoopedInfo decodable_info(decodable_opts_, &am_nnet_);

    kaldi::SingleUtteranceNnet3Decoder decoder(lattice_faster_decoder_config_,
                                               trans_model_, decodable_info, *decode_fst_,
                                               &feature_pipeline);

    kaldi::Matrix<kaldi::BaseFloat> wave_matrix;
    read_raw_wav_stream(wav_stream, data_bytes, wave_matrix);

    // get the data for channel zero (if the signal is not mono, we only
    // take the first channel).
    kaldi::SubVector<kaldi::BaseFloat> data(wave_matrix, 0);
    constexpr kaldi::BaseFloat samp_freq = 8000;

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
        _decode_wave(feature_pipeline, silence_weighting, decoder, wave_part, delta_weights, samp_freq);

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

    decoder.GetLattice(true, &clat);
    _find_alternatives(clat, n_best, results);
}

// Factory for creating decoders with shared decoding graph and model parameters
// Caches the graph and params to be able to produce decoders on demand.
class DecoderFactory final {

  private:
    const std::unique_ptr<fst::Fst<fst::StdArc>> decode_fst_;

    const std::string model_dir_;
    const kaldi::BaseFloat beam_;
    const std::size_t min_active_;
    const std::size_t max_active_;
    const kaldi::BaseFloat lattice_beam_;
    const kaldi::BaseFloat acoustic_scale_;
    const std::size_t frame_subsampling_factor_;

  public:
    explicit DecoderFactory(const std::string &,
                            const kaldi::BaseFloat &,
                            const std::size_t &,
                            const std::size_t &,
                            const kaldi::BaseFloat &,
                            const kaldi::BaseFloat &,
                            const std::size_t &) noexcept;

    inline Decoder *produce() const noexcept;

    // friendly alias for the producer method
    inline Decoder *operator()() const noexcept;
};

DecoderFactory::DecoderFactory(const std::string &model_dir,
                               const kaldi::BaseFloat &beam,
                               const std::size_t &min_active,
                               const std::size_t &max_active,
                               const kaldi::BaseFloat &lattice_beam,
                               const kaldi::BaseFloat &acoustic_scale,
                               const std::size_t &frame_subsampling_factor) noexcept
  : decode_fst_(fst::ReadFstKaldiGeneric(join_path(model_dir, "HCLG.fst"))),
    model_dir_ (model_dir),
    beam_(beam), min_active_(min_active), max_active_(max_active), lattice_beam_(lattice_beam),
    acoustic_scale_(acoustic_scale), frame_subsampling_factor_(frame_subsampling_factor) {}

inline Decoder *DecoderFactory::produce() const noexcept {
    return new Decoder(beam_, min_active_, max_active_, lattice_beam_,
                       acoustic_scale_, frame_subsampling_factor_,
                       model_dir_, decode_fst_.get());
}

inline Decoder *DecoderFactory::operator()() const noexcept {
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

    ~DecoderQueue() noexcept;

    // friendly alias for `pop`
    inline Decoder *acquire();

    // friendly alias for `push`
    inline void release(Decoder *const);
};

DecoderQueue::DecoderQueue(const ModelSpec &model_spec) {
    std::cout << ":: Loading model from " << model_spec.path << ENDL;

#if DEBUG
    // LOG MODELS LOAD TIME --> START
    std::chrono::system_clock::time_point start_time = std::chrono::system_clock::now();
#endif
    decoder_factory_ = std::unique_ptr<DecoderFactory>(new DecoderFactory(model_spec.path,
                                                                          model_spec.beam,
                                                                          model_spec.min_active,
                                                                          model_spec.max_active,
                                                                          model_spec.lattice_beam,
                                                                          model_spec.acoustic_scale,
                                                                          model_spec.frame_subsampling_factor));
    for (size_t i = 0; i < model_spec.n_decoders; i++) {
        queue_.push(decoder_factory_->produce());
    }

#if DEBUG
    std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
    // LOG MODELS LOAD TIME --> END

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << ":: Decoder models concurrent queue init in: " << ms.count() << "ms" << ENDL;
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
