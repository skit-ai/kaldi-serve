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

    friend class Decoder;

  public:
    explicit ChainModel(const ModelSpec &model_spec);

    ModelSpec model_spec;

  private:
    std::unique_ptr<fst::Fst<fst::StdArc>> decode_fst_;

    kaldi::nnet3::AmNnetSimple am_nnet_;
    kaldi::TransitionModel trans_model_;

    std::unique_ptr<fst::SymbolTable> word_syms_;

    std::unique_ptr<kaldi::OnlineNnet2FeaturePipelineInfo> feature_info_;
    std::unique_ptr<kaldi::nnet3::DecodableNnetSimpleLoopedInfo> decodable_info_;
    
    kaldi::LatticeFasterDecoderConfig lattice_faster_decoder_config_;
    kaldi::nnet3::NnetSimpleLoopedComputationOptions decodable_opts_;

    // Optional stuff

    // word-boundary info
    std::unique_ptr<kaldi::WordBoundaryInfo> wb_info_;

    // RNNLM objects
    kaldi::nnet3::Nnet rnnlm_;
    kaldi::CuMatrix<kaldi::BaseFloat> word_embedding_mat_;
    
    std::unique_ptr<fst::BackoffDeterministicOnDemandFst<fst::StdArc>> lm_to_subtract_det_backoff_;
    std::unique_ptr<const kaldi::rnnlm::RnnlmComputeStateInfo> rnnlm_info_;

    kaldi::ComposeLatticePrunedOptions compose_opts_;
    kaldi::BaseFloat rnnlm_weight_;
};

} // namespace kaldiserve