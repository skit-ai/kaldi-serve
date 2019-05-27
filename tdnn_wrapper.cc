#include "tdnn_wrapper.h"

#define VERBOSE 0

namespace kaldi {

    Model::Model(BaseFloat beam, int32 max_active, int32 min_active,
        BaseFloat lattice_beam, BaseFloat acoustic_scale,
        int32 frame_subsampling_factor, char* &word_syms_filename,
        char* &model_in_filename, char* &fst_in_str,
        char* &mfcc_config, char* &ie_conf_filename) {

        try {
            using namespace fst;

            #if VERBOSE
                KALDI_LOG << "model_in_filename:         " << model_in_filename;
                KALDI_LOG << "fst_in_str:                " << fst_in_str;
                KALDI_LOG << "mfcc_config:               " << mfcc_config;
                KALDI_LOG << "ie_conf_filename:          " << ie_conf_filename;
            #endif

            // feature_config includes configuration for the iVector adaptation,
            // as well as the basic features.
            OnlineNnet2FeaturePipelineConfig feature_opts;

            feature_opts.mfcc_config                   = mfcc_config;
            feature_opts.ivector_extraction_config     = ie_conf_filename;
            lattice_faster_decoder_config.max_active                    = max_active;
            lattice_faster_decoder_config.min_active                    = min_active;
            lattice_faster_decoder_config.beam                          = beam;
            lattice_faster_decoder_config.lattice_beam                  = lattice_beam;
            decodable_opts.acoustic_scale              = acoustic_scale;
            decodable_opts.frame_subsampling_factor    = frame_subsampling_factor;

            {
                bool binary;
                Input ki(model_in_filename, &binary);
                trans_model.Read(ki.Stream(), binary);
                am_nnet.Read(ki.Stream(), binary);
                SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
                SetDropoutTestMode(true, &(am_nnet.GetNnet()));
                nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(am_nnet.GetNnet()));
            }

            decode_fst = ReadFstKaldiGeneric(fst_in_str);

            word_syms = NULL;
            if (word_syms_filename != "" && !(word_syms = fst::SymbolTable::ReadText(word_syms_filename))) {
                KALDI_ERR << "Could not read symbol table from file " << word_syms_filename;
            }

            feature_info = new OnlineNnet2FeaturePipelineInfo(feature_opts);
        } catch (const std::exception &e) {
            KALDI_ERR << e.what(); // model not loaded
        }
    }

    Model::~Model() {
        delete decode_fst;
        delete word_syms;  // will delete if non-NULL.
    }

    std::vector<Model::result_tuple> Model::CInfer(std::string wav_file_path, int32 max_alternatives) {
        using namespace fst;

        BaseFloat chunk_length_secs = 1;

        OnlineIvectorExtractorAdaptationState adaptation_state(feature_info->ivector_extractor_info);
        OnlineNnet2FeaturePipeline feature_pipeline(*feature_info);
        feature_pipeline.SetAdaptationState(adaptation_state);

        OnlineSilenceWeighting silence_weighting(trans_model,
            feature_info->silence_weighting_config, decodable_opts.frame_subsampling_factor);
        nnet3::DecodableNnetSimpleLoopedInfo decodable_info(decodable_opts, &am_nnet);

        SingleUtteranceNnet3Decoder decoder(
            lattice_faster_decoder_config, trans_model, decodable_info, *decode_fst, &feature_pipeline
        );
        std::ifstream file_data(wav_file_path, std::ifstream::binary);
        WaveData wave_data;
        wave_data.Read(file_data);
        // get the data for channel zero (if the signal is not mono, we only
        // take the first channel).
        SubVector<BaseFloat> data(wave_data.Data(), 0);
        BaseFloat samp_freq = wave_data.SampFreq();
        int32 chunk_length;
        if (chunk_length_secs > 0) {
            chunk_length = int32(samp_freq * chunk_length_secs);
            if (chunk_length == 0) chunk_length = 1;
        } else {
            chunk_length = std::numeric_limits<int32>::max();
        }
        int32 samp_offset = 0;
        std::vector<std::pair<int32, BaseFloat> > delta_weights;

        while (samp_offset < data.Dim()) {
            int32 samp_remaining = data.Dim() - samp_offset;
            int32 num_samp = chunk_length < samp_remaining ? chunk_length : samp_remaining;

            SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);
            feature_pipeline.AcceptWaveform(samp_freq, wave_part);

            samp_offset += num_samp;
            if (samp_offset == data.Dim()) {
            // no more input. flush out last frames
            feature_pipeline.InputFinished();
            }

            if (silence_weighting.Active() && feature_pipeline.IvectorFeature() != NULL) {
            silence_weighting.ComputeCurrentTraceback(decoder.Decoder());
            silence_weighting.GetDeltaWeights(feature_pipeline.NumFramesReady(), &delta_weights);
            feature_pipeline.IvectorFeature()->UpdateFrameWeights(delta_weights);
            }
            decoder.AdvanceDecoding();
        }
        decoder.FinalizeDecoding();

        CompactLattice clat;
        decoder.GetLattice(true, &clat);

        return get_decoded_string(word_syms, clat, max_alternatives);
    }

    std::vector<Model::result_tuple> Model::CInferObject(int32 num_frames, BaseFloat* frames, int32 max_alternatives) {
        using namespace fst;

        BaseFloat chunk_length_secs = 1;
        BaseFloat samp_freq = 8000;

        OnlineIvectorExtractorAdaptationState adaptation_state(feature_info->ivector_extractor_info);
        OnlineNnet2FeaturePipeline feature_pipeline(*feature_info);
        feature_pipeline.SetAdaptationState(adaptation_state);

        OnlineSilenceWeighting silence_weighting(trans_model,
            feature_info->silence_weighting_config, decodable_opts.frame_subsampling_factor);
        nnet3::DecodableNnetSimpleLoopedInfo decodable_info(decodable_opts, &am_nnet);

        SingleUtteranceNnet3Decoder decoder(
            lattice_faster_decoder_config, trans_model, decodable_info, *decode_fst, &feature_pipeline
        );

        Vector<BaseFloat> data(num_frames, kUndefined);
        for (int i=0; i<num_frames; i++) {
            data(i) = frames[i];
        }
        int32 chunk_length;
        if (chunk_length_secs > 0) {
            chunk_length = int32(samp_freq * chunk_length_secs);
            if (chunk_length == 0) chunk_length = 1;
        } else {
            chunk_length = std::numeric_limits<int32>::max();
        }
        int32 samp_offset = 0;
        std::vector<std::pair<int32, BaseFloat> > delta_weights;

        while (samp_offset < data.Dim()) {
            int32 samp_remaining = data.Dim() - samp_offset;
            int32 num_samp = chunk_length < samp_remaining ? chunk_length : samp_remaining;

            SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);
            feature_pipeline.AcceptWaveform(samp_freq, wave_part);

            samp_offset += num_samp;
            if (samp_offset == data.Dim()) {
            // no more input. flush out last frames
            feature_pipeline.InputFinished();
            }

            if (silence_weighting.Active() && feature_pipeline.IvectorFeature() != NULL) {
            silence_weighting.ComputeCurrentTraceback(decoder.Decoder());
            silence_weighting.GetDeltaWeights(feature_pipeline.NumFramesReady(), &delta_weights);
            feature_pipeline.IvectorFeature()->UpdateFrameWeights(delta_weights);
            }
            decoder.AdvanceDecoding();
        }
        decoder.FinalizeDecoding();

        CompactLattice clat;
        decoder.GetLattice(true, &clat);

        return get_decoded_string(word_syms, clat, max_alternatives);
    }

    std::vector<Model::result_tuple> Model::get_decoded_string(
        const fst::SymbolTable *word_syms,
        const CompactLattice &clat,
        int32 max_alternatives) {

        std::vector<Model::result_tuple> results;

        if (clat.NumStates() == 0) {
          KALDI_LOG << "Empty lattice.";
          return results;
        }
        // convert from compact to normal lattice
        Lattice* lat = new Lattice();
        fst::ConvertLattice(clat, lat);

        // get n-best path lattice
        Lattice nbest_lat;
        std::vector<Lattice> nbest_lats;
        ShortestPath(*lat, &nbest_lat, max_alternatives);
        fst::ConvertNbestToVector(nbest_lat, &nbest_lats);

        if (nbest_lats.empty()) {
            KALDI_WARN << "no N-best entries";
        } else {
            for (int32 k = 0; k < static_cast<int32>(nbest_lats.size()); k++) {
                LatticeWeight  weight;
                std::vector<int32> alignment;
                std::vector<int32> words;
                std::string answer = "";

                GetLinearSymbolSequence(nbest_lats[k], &alignment, &words, &weight);

                for (size_t i = 0; i < words.size(); i++) {
                    std::string s = word_syms->Find(words[i]);
                    answer += s + " ";
                }
                results.push_back(std::make_tuple(
                    answer,
                    get_confidence(float(weight.Value1()), float(weight.Value2()), words.size())
                ));
            }
        }
        return results;
    }

    double Model::get_confidence(float lmScore, float amScore, int32 numWords) {
        return std::max(
            0.0,
            std::min(
                1.0, -0.0001466488 * (2.388449*lmScore + amScore) / (numWords + 1) + 0.956
            )
        );
    }
}