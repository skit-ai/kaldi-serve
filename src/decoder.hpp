/*
 * Decoding graph and operations.
 */

// Include guard
#pragma once

// C++ stl includes
#include <future>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

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

// Return n-best alternative from lattice. Output symbols are converted to words
// based on word-syms.
utterance_results_t find_alternatives(const fst::SymbolTable *word_syms,
                                      const kaldi::CompactLattice &clat,
                                      const std::size_t &n_best) {
    utterance_results_t results;

    if (clat.NumStates() == 0) {
        KALDI_LOG << "Empty lattice.";
        return results;
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
    return results;
}

class Decoder {

  private:
    const fst::Fst<fst::StdArc> *decode_fst;
    std::unique_ptr<fst::SymbolTable> word_syms;
    kaldi::OnlineNnet2FeaturePipelineInfo *feature_info;
    kaldi::nnet3::AmNnetSimple am_nnet;
    kaldi::TransitionModel trans_model;

    kaldi::LatticeFasterDecoderConfig lattice_faster_decoder_config;
    kaldi::nnet3::NnetSimpleLoopedComputationOptions decodable_opts;

  public:
    explicit Decoder(const kaldi::BaseFloat &, const std::size_t &,
                     const std::size_t&, const kaldi::BaseFloat &,
                     const kaldi::BaseFloat &, const std::size_t &,
                     const std::string &, const std::string &,
                     const std::string &, const std::string &,
                     const fst::Fst<fst::StdArc> *);

    Decoder *clone() const;

    // Decoding processes

    // initialize decoding state variables (per stream)
    void decode_stream_initialize(kaldi::OnlineIvectorExtractorAdaptationState*,
                                  kaldi::OnlineNnet2FeaturePipeline*,
                                  kaldi::nnet3::DecodableNnetSimpleLoopedInfo*,
                                  kaldi::OnlineSilenceWeighting*,
                                  kaldi::SingleUtteranceNnet3Decoder*);
    
    // decode intermediate frames of wav streams
    void decode_stream_process(kaldi::OnlineNnet2FeaturePipeline*,
                               kaldi::OnlineSilenceWeighting*,
                               kaldi::SingleUtteranceNnet3Decoder*,
                               std::istream &) const;


    // get the final utterances based on the compact lattice
    utterance_results_t decode_stream_final(kaldi::OnlineNnet2FeaturePipeline*,
                                            kaldi::SingleUtteranceNnet3Decoder*,
                                            const std::size_t &) const;

    // cleanup decoding state variables (per stream)
    void cleanup(kaldi::OnlineIvectorExtractorAdaptationState*,
                 kaldi::OnlineNnet2FeaturePipeline*,
                 kaldi::nnet3::DecodableNnetSimpleLoopedInfo*,
                 kaldi::OnlineSilenceWeighting*,
                 kaldi::SingleUtteranceNnet3Decoder*) const;
};

void Decoder::cleanup(kaldi::OnlineIvectorExtractorAdaptationState* adaptation_state,
                      kaldi::OnlineNnet2FeaturePipeline* feature_pipeline,
                      kaldi::nnet3::DecodableNnetSimpleLoopedInfo* decodable_info,
                      kaldi::OnlineSilenceWeighting* silence_weighting,
                      kaldi::SingleUtteranceNnet3Decoder* decoder) const {
    delete adaptation_state;
    delete feature_pipeline;
    delete silence_weighting;
    delete decodable_info;
    delete decoder;
    
    adaptation_state = NULL;
    feature_pipeline = NULL;
    silence_weighting = NULL;
    decodable_info = NULL;
    decoder = NULL;
}

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
                 const fst::Fst<fst::StdArc> *decode_fst) 
    : decode_fst(decode_fst) {
    
    try {
        kaldi::OnlineNnet2FeaturePipelineConfig feature_opts;

        feature_opts.mfcc_config = mfcc_conf_filepath;
        feature_opts.ivector_extraction_config = ie_conf_filepath;
        lattice_faster_decoder_config.max_active = max_active;
        lattice_faster_decoder_config.min_active = min_active;
        lattice_faster_decoder_config.beam = beam;
        lattice_faster_decoder_config.lattice_beam = lattice_beam;
        decodable_opts.acoustic_scale = acoustic_scale;
        decodable_opts.frame_subsampling_factor = frame_subsampling_factor;

        bool binary;
        kaldi::Input ki(model_filepath, &binary);

        trans_model.Read(ki.Stream(), binary);
        am_nnet.Read(ki.Stream(), binary);
        
        kaldi::nnet3::SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
        kaldi::nnet3::SetDropoutTestMode(true, &(am_nnet.GetNnet()));
        kaldi::nnet3::CollapseModel(kaldi::nnet3::CollapseModelConfig(), &(am_nnet.GetNnet()));

        word_syms = NULL;
        if (word_syms_filepath != "" && !(word_syms = std::unique_ptr<fst::SymbolTable>(fst::SymbolTable::ReadText(word_syms_filepath)))) {
            KALDI_ERR << "Could not read symbol table from file " << word_syms_filepath;
        }

        feature_info = new kaldi::OnlineNnet2FeaturePipelineInfo(feature_opts);

    } catch (const std::exception &e) {
        KALDI_ERR << e.what();
    }
}

Decoder *Decoder::clone() const {
    // implement deep copy here
    return NULL;
}

void Decoder::decode_stream_initialize(kaldi::OnlineIvectorExtractorAdaptationState* adaptation_state,
                                       kaldi::OnlineNnet2FeaturePipeline* feature_pipeline,
                                       kaldi::nnet3::DecodableNnetSimpleLoopedInfo* decodable_info,
                                       kaldi::OnlineSilenceWeighting* silence_weighting,
                                       kaldi::SingleUtteranceNnet3Decoder* decoder) {
    adaptation_state = new kaldi::OnlineIvectorExtractorAdaptationState(feature_info->ivector_extractor_info);
    feature_pipeline = new kaldi::OnlineNnet2FeaturePipeline(*feature_info);
    feature_pipeline->SetAdaptationState(*adaptation_state);
    
    decodable_info = new kaldi::nnet3::DecodableNnetSimpleLoopedInfo(decodable_opts, &am_nnet);
    silence_weighting = new kaldi::OnlineSilenceWeighting(trans_model, feature_info->silence_weighting_config,
                                                          decodable_opts.frame_subsampling_factor);
    decoder = new kaldi::SingleUtteranceNnet3Decoder(lattice_faster_decoder_config,
                                                     trans_model, *decodable_info,
                                                     *decode_fst, feature_pipeline);
}

void Decoder::decode_stream_process(kaldi::OnlineNnet2FeaturePipeline* feature_pipeline,
                                    kaldi::OnlineSilenceWeighting* silence_weighting,
                                    kaldi::SingleUtteranceNnet3Decoder* decoder,
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
        feature_pipeline->AcceptWaveform(samp_freq, wave_part);

        samp_offset += num_samp;
        if (samp_offset == data.Dim()) {
            return;
        }

        if (silence_weighting->Active() && feature_pipeline->IvectorFeature() != NULL) {
            silence_weighting->ComputeCurrentTraceback(decoder->Decoder());
            silence_weighting->GetDeltaWeights(feature_pipeline->NumFramesReady(),
                                              &delta_weights);
            feature_pipeline->IvectorFeature()->UpdateFrameWeights(delta_weights);
        }
        decoder->AdvanceDecoding();
    }
}

utterance_results_t Decoder::decode_stream_final(kaldi::OnlineNnet2FeaturePipeline* feature_pipeline,
                                                 kaldi::SingleUtteranceNnet3Decoder* decoder,
                                                 const std::size_t &n_best) const {
    feature_pipeline->InputFinished();

    decoder->AdvanceDecoding();
    decoder->FinalizeDecoding();

    kaldi::CompactLattice clat;
    decoder->GetLattice(true, &clat);

    return find_alternatives(word_syms.get(), clat, n_best);
}

// Factory for creating decoders with shared decoding graph.
class DecoderFactory {

  private:
    const std::unique_ptr<fst::Fst<fst::StdArc>> decode_fst;

  public:
    explicit DecoderFactory(const std::string &hclg_filepath) : decode_fst(std::unique_ptr<fst::Fst<fst::StdArc>>(fst::ReadFstKaldiGeneric(hclg_filepath))) {}

    std::unique_ptr<Decoder> operator()(const kaldi::BaseFloat &beam,
                                        const std::size_t &max_active,
                                        const std::size_t &min_active,
                                        const kaldi::BaseFloat &lattice_beam,
                                        const kaldi::BaseFloat &acoustic_scale,
                                        const std::size_t &frame_subsampling_factor,
                                        const std::string &word_syms_filepath,
                                        const std::string &model_filepath,
                                        const std::string &mfcc_conf_filepath,
                                        const std::string &ie_conf_filepath) {
        return std::unique_ptr<Decoder>(new Decoder(beam, max_active, min_active, lattice_beam,
                                                    acoustic_scale, frame_subsampling_factor,
                                                    word_syms_filepath, model_filepath,
                                                    mfcc_conf_filepath, ie_conf_filepath,
                                                    this->decode_fst.get()));
    }
};
