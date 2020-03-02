// Decoding graph and operations.
#pragma once

#include "config.hpp"

// stl includes
#include <iostream>
#include <string>

// kaldi includes
#include "fstext/fstext-lib.h"
#include "nnet3/nnet-utils.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/onlinebin-util.h"
#include "util/kaldi-thread.h"

// local includes
#include "decoder.hpp"
#include "utils.hpp"


// Model is a data class that holds all the immutable
// ASR components that can be shared across Decoder instances
class Model final {

    friend class Decoder;

  public:
    explicit Model(const ModelSpec &model_spec);

    ModelSpec model_spec;

  private:
    std::unique_ptr<fst::Fst<fst::StdArc>> decode_fst_;

    kaldi::nnet3::AmNnetSimple am_nnet_;
    kaldi::TransitionModel trans_model_;

    std::unique_ptr<fst::SymbolTable> word_syms_;

    std::unique_ptr<kaldi::WordBoundaryInfo> wb_info_;
    std::unique_ptr<kaldi::OnlineNnet2FeaturePipelineInfo> feature_info_;
    std::unique_ptr<kaldi::nnet3::DecodableNnetSimpleLoopedInfo> decodable_info_;
    
    kaldi::LatticeFasterDecoderConfig lattice_faster_decoder_config_;
    kaldi::nnet3::NnetSimpleLoopedComputationOptions decodable_opts_;
};

Model::Model(const ModelSpec &model_spec) : model_spec(model_spec) {
    std::string model_dir = model_spec.path;

    try {
        std::string hclg_filepath = join_path(model_dir, "HCLG.fst");
        std::string model_filepath = join_path(model_dir, "final.mdl");
        std::string word_syms_filepath = join_path(model_dir, "words.txt");
        std::string word_boundary_filepath = join_path(model_dir, "word_boundary.int");

        std::string conf_dir = join_path(model_dir, "conf");
        std::string mfcc_conf_filepath = join_path(conf_dir, "mfcc.conf");
        std::string ivector_conf_filepath = join_path(conf_dir, "ivector_extractor.conf");

        decode_fst_ = std::unique_ptr<fst::Fst<fst::StdArc>>(fst::ReadFstKaldiGeneric(hclg_filepath));

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

        if (exists(word_boundary_filepath)) {
            kaldi::WordBoundaryInfoNewOpts word_boundary_opts;
            wb_info_ = std::make_unique<kaldi::WordBoundaryInfo>(word_boundary_opts, word_boundary_filepath);
        } else {
            KALDI_WARN << "Word boundary file" << word_boundary_filepath
                       << " not found. Disabling word level features.";
        }

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
        feature_info_->silence_weighting_config.silence_weight = model_spec.silence_weight;

        lattice_faster_decoder_config_.min_active = model_spec.min_active;
        lattice_faster_decoder_config_.max_active = model_spec.max_active;
        lattice_faster_decoder_config_.beam = model_spec.beam;
        lattice_faster_decoder_config_.lattice_beam = model_spec.lattice_beam;

        decodable_opts_.acoustic_scale = model_spec.acoustic_scale;
        decodable_opts_.frame_subsampling_factor = model_spec.frame_subsampling_factor;
        decodable_info_ = std::make_unique<kaldi::nnet3::DecodableNnetSimpleLoopedInfo>(decodable_opts_, &am_nnet_);
    
    } catch (const std::exception &e) {
        KALDI_ERR << e.what();
    }
}