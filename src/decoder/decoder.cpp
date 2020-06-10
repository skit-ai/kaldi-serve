// decoder-cpu.cpp - CPU Decoder Implementation

// local includes
#include "decoder.hpp"
#include "types.hpp"


namespace kaldiserve {

Decoder::Decoder(ChainModel *const model) : model_(model) {

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

    adaptation_state_ = new kaldi::OnlineIvectorExtractorAdaptationState(model_->feature_info_->ivector_extractor_info);

    feature_pipeline_ = new kaldi::OnlineNnet2FeaturePipeline(*model_->feature_info_);
    feature_pipeline_->SetAdaptationState(*adaptation_state_);

    decoder_ = new kaldi::SingleUtteranceNnet3Decoder(model_->lattice_faster_decoder_config_,
                                                      model_->trans_model_, *model_->decodable_info_,
                                                      *model_->decode_fst_, feature_pipeline_);
    decoder_->InitDecoding();

    silence_weighting_ = new kaldi::OnlineSilenceWeighting(model_->trans_model_,
                                                           model_->feature_info_->silence_weighting_config,
                                                           model_->decodable_opts_.frame_subsampling_factor);

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
                                          const float& samp_freq,
                                          const int &data_bytes) {
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
                               const float &chunk_size) {
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
                                   const float &samp_freq,
                                   const int &data_bytes,
                                   const float &chunk_size) {
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

void Decoder::get_decoded_results(const int &n_best,
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

} // namespace kaldiserve