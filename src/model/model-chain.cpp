// model-chain.cpp - Chain Model Implementation

// stl includes
#include <iostream>
#include <string>

// local includes
#include "model.hpp"
#include "utils.hpp"
#include "types.hpp"


namespace kaldiserve {

ChainModel::ChainModel(const ModelSpec &model_spec) : model_spec(model_spec) {
    std::string model_dir = model_spec.path;

    try {
        std::string hclg_filepath = join_path(model_dir, "HCLG.fst");
        std::string model_filepath = join_path(model_dir, "final.mdl");
        std::string word_syms_filepath = join_path(model_dir, "words.txt");
        std::string word_boundary_filepath = join_path(model_dir, "word_boundary.int");

        std::string conf_dir = join_path(model_dir, "conf");
        std::string mfcc_conf_filepath = join_path(conf_dir, "mfcc.conf");
        std::string ivector_conf_filepath = join_path(conf_dir, "ivector_extractor.conf");

        std::string rnnlm_dir = join_path(model_dir, "rnnlm");

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
            wb_info_ = make_uniq<kaldi::WordBoundaryInfo>(word_boundary_opts, word_boundary_filepath);
        } else {
            KALDI_WARN << "Word boundary file" << word_boundary_filepath
                       << " not found. Disabling word level features.";
        }

        if (exists(rnnlm_dir) && 
            exists(join_path(rnnlm_dir, "final.raw")) && 
            exists(join_path(rnnlm_dir, "word_embedding.mat")) && 
            exists(join_path(rnnlm_dir, "G.fst"))) {
            
            lm_to_subtract_fst_ =
                std::unique_ptr<const fst::VectorFst<fst::StdArc>>(fst::ReadAndPrepareLmFst(join_path(rnnlm_dir, "G.fst")));
            rnnlm_weight_ = model_spec.rnnlm_weight;
            std::cout << "rnnlm_weight: " << rnnlm_weight_ << ENDL;

            kaldi::ReadKaldiObject(join_path(rnnlm_dir, "final.raw"), &rnnlm_);
            KALDI_ASSERT(IsSimpleNnet(rnnlm_));
            kaldi::ReadKaldiObject(join_path(rnnlm_dir, "word_embedding.mat"), &word_embedding_mat_);
            
            std::cout << "# Word Embeddings (RNNLM): " << word_embedding_mat_.NumRows() << ENDL;

            // hack: RNNLM compute opts only takes values from parsed options like in cmd-line
            const char *usage = "Usage: model.hpp [options]";
            kaldi::ParseOptions po(usage);
            rnnlm_opts_.Register(&po);

            std::string bos_opt = "--bos-symbol=" + model_spec.bos_index;
            std::string eos_opt = "--eos-symbol=" + model_spec.eos_index;

            const char *argv[] = {
                "model.hpp",
                bos_opt.c_str(),
                eos_opt.c_str(),
                NULL
            };

            po.Read((sizeof(argv)/sizeof(argv[0])) - 1, argv);
            rnnlm_info_ =
                make_uniq<const kaldi::rnnlm::RnnlmComputeStateInfo>(rnnlm_opts_, rnnlm_, word_embedding_mat_);
        } else {
            KALDI_WARN << "RNNLM artefacts not found. Disabling RNNLM rescoring feature.";
        }

        feature_info_ = make_uniq<kaldi::OnlineNnet2FeaturePipelineInfo>();
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
        decodable_info_ = make_uniq<kaldi::nnet3::DecodableNnetSimpleLoopedInfo>(decodable_opts_, &am_nnet_);
    
    } catch (const std::exception &e) {
        KALDI_ERR << e.what();
    }
}

} // namespace kaldiserve