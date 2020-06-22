// model.hpp - Model Wrapper Interface
#pragma once

// stl includes
#include <memory>

// kaldi includes
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "rnnlm/rnnlm-lattice-rescoring.h"
#include "fstext/fstext-lib.h"
#include "nnet3/nnet-utils.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/onlinebin-util.h"
#include "util/kaldi-thread.h"

// local includes
#include "config.hpp"
#include "types.hpp"
#include "decoder.hpp"
#include "utils.hpp"


namespace kaldiserve {

// Chain (DNN-HMM NNet3) Model is a data class that holds all the
// immutable ASR Model components that can be shared across Decoder instances.
class ChainModel final {

  public:
    explicit ChainModel(const ModelSpec &model_spec);

    // Model Config
    ModelSpec model_spec;

    // HCLG.fst graph
    std::unique_ptr<fst::Fst<fst::StdArc>> decode_fst;

    // NNet3 AM
    kaldi::nnet3::AmNnetSimple am_nnet;
    // Transition Model (HMM)
    kaldi::TransitionModel trans_model;

    // Word Symbols table (int->word)
    std::unique_ptr<fst::SymbolTable> word_syms;

    // Online Feature Pipeline options
    std::unique_ptr<kaldi::OnlineNnet2FeaturePipelineInfo> feature_info;
    // 
    std::unique_ptr<kaldi::nnet3::DecodableNnetSimpleLoopedInfo> decodable_info;
    
    kaldi::LatticeFasterDecoderConfig lattice_faster_decoder_config;
    kaldi::nnet3::NnetSimpleLoopedComputationOptions decodable_opts;

    // Word Boundary info (for word level timings)
    std::unique_ptr<kaldi::WordBoundaryInfo> wb_info;

    // NNet3 RNNLM
    kaldi::nnet3::Nnet rnnlm;
    // Word Embeddings matrix
    kaldi::CuMatrix<kaldi::BaseFloat> word_embedding_mat;
    // Original G.fst LM
    std::unique_ptr<const fst::VectorFst<fst::StdArc>> lm_to_subtract_fst;  
    // RNNLM info object (encapsulates RNNLM, Word Embeddings and RNNLM options)
    std::unique_ptr<const kaldi::rnnlm::RnnlmComputeStateInfo> rnnlm_info;
    
    // RNNLM interpolation weight
    kaldi::BaseFloat rnnlm_weight;
    // RNNLM options
    kaldi::rnnlm::RnnlmComputeStateComputationOptions rnnlm_opts;
    // LM composition options
    kaldi::ComposeLatticePrunedOptions compose_opts;
};

} // namespace kaldiserve