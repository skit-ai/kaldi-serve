// decoder-common.cpp - Decoder Common methods Implementation

// local includes
#include "config.hpp"
#include "decoder.hpp"
#include "types.hpp"


namespace kaldiserve {

void find_alternatives(kaldi::CompactLattice &clat,
                       const std::size_t &n_best,
                       utterance_results_t &results,
                       const bool &word_level,
                       ChainModel *const model,
                       const DecoderOptions &options) {
    if (clat.NumStates() == 0) {
        KALDI_LOG << "Empty lattice.";
    }

    if (options.enable_rnnlm) {
        // rnnlm.fst
        std::unique_ptr<kaldi::rnnlm::KaldiRnnlmDeterministicFst> lm_to_add_orig = 
            make_uniq<kaldi::rnnlm::KaldiRnnlmDeterministicFst>(model->model_spec.max_ngram_order, *model->rnnlm_info);
        std::unique_ptr<fst::ScaleDeterministicOnDemandFst> lm_to_add =
            make_uniq<fst::ScaleDeterministicOnDemandFst>(model->rnnlm_weight, lm_to_add_orig.get());

        // G.fst
        std::unique_ptr<fst::BackoffDeterministicOnDemandFst<fst::StdArc>> lm_to_subtract_det_backoff =
            make_uniq<fst::BackoffDeterministicOnDemandFst<fst::StdArc>>(*model->lm_to_subtract_fst);
        std::unique_ptr<fst::ScaleDeterministicOnDemandFst> lm_to_subtract_det_scale =
            make_uniq<fst::ScaleDeterministicOnDemandFst>(-model->rnnlm_weight, lm_to_subtract_det_backoff.get());
        
        // combine both LM fsts
        fst::ComposeDeterministicOnDemandFst<fst::StdArc> combined_lms(lm_to_subtract_det_scale.get(), lm_to_add.get());

        // Before composing with the LM FST, we scale the lattice weights
        // by the inverse of "lm_scale".  We'll later scale by "lm_scale".
        // We do it this way so we can determinize and it will give the
        // right effect (taking the "best path" through the LM) regardless
        // of the sign of lm_scale.
        if (model->decodable_opts.acoustic_scale != 1.0) {
            fst::ScaleLattice(fst::AcousticLatticeScale(model->decodable_opts.acoustic_scale), &clat);
        }
        kaldi::TopSortCompactLatticeIfNeeded(&clat);

        // compose lattice with combined language model.
        kaldi::CompactLattice composed_clat;
        kaldi::ComposeCompactLatticePruned(model->compose_opts, clat,
                                           &combined_lms, &composed_clat);

        if (composed_clat.NumStates() == 0) {
            // Something went wrong.  A warning will already have been printed.
            KALDI_WARN << "Empty lattice after RNNLM rescoring.";
        } else {
            clat = composed_clat;
        }
    }

    auto lat = make_uniq<kaldi::Lattice>();
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
            word_strings.push_back(model->word_syms->Find(wid));
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

    bool ok = kaldi::WordAlignLattice(clat, model->trans_model, *model->wb_info, max_states, &aligned_clat);

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

        fst::ScaleLattice(fst::LatticeScale(lm_scale, model->decodable_opts.acoustic_scale), &aligned_clat);
        auto mbr = make_uniq<kaldi::MinimumBayesRisk>(aligned_clat, mbr_opts);

        const std::vector<kaldi::BaseFloat> &conf = mbr->GetOneBestConfidences();
        const std::vector<int32> &best_words = mbr->GetOneBest();
        const std::vector<std::pair<kaldi::BaseFloat, kaldi::BaseFloat>> &times = mbr->GetOneBestTimes();

        KALDI_ASSERT(conf.size() == best_words.size() && best_words.size() == times.size());

        for (size_t i = 0; i < best_words.size(); i++) {
            KALDI_ASSERT(best_words[i] != 0 || mbr_opts.print_silence); // Should not have epsilons.

            Word word;
            kaldi::BaseFloat time_unit = frame_shift * model->decodable_opts.frame_subsampling_factor;
            word.start_time = times[i].first * time_unit;
            word.end_time = times[i].second * time_unit;
            word.word = model->word_syms->Find(best_words[i]); // lookup word in SymbolTable
            word.confidence = conf[i];

            words.push_back(word);
        }
    }

    if (!results.empty() and !words.empty()) {
        results[0].words = words;
    }
}

} // namespace kaldiserve