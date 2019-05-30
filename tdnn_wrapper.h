#include <string>

#include "feat/wave-reader.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "lat/kaldi-lattice.h"
#include "util/kaldi-thread.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {

class Model {
    public:
        Model(
            BaseFloat beam,
            int32 max_active,
            int32 min_active,
            BaseFloat lattice_beam,
            BaseFloat acoustic_scale,
            int32 frame_subsampling_factor,
            char* &word_syms_filename,
            char* &model_in_filename,
            char* &fst_in_str,
            char* &mfcc_config,
            char* &ie_conf_filename
        );

        ~Model();

        typedef std::tuple<std::string,double> result_tuple;

        std::vector<result_tuple> CInfer(std::string wav_file_path, int32 max_alternatives);

        std::vector<result_tuple> CInferObject(int32 num_frames, BaseFloat* frames, int32 max_alternatives);


    private:
        fst::SymbolTable *word_syms;
        LatticeFasterDecoderConfig lattice_faster_decoder_config;
        OnlineNnet2FeaturePipelineInfo *feature_info;
        nnet3::AmNnetSimple am_nnet;
        TransitionModel trans_model;
        nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
        fst::Fst<fst::StdArc> *decode_fst;

        // takes generated lattice and word symbols table to give decoded string
        std::vector<result_tuple> get_decoded_string(
            const fst::SymbolTable *word_syms,
            const CompactLattice &clat,
            int32 max_alternatives
        );

        double get_confidence(BaseFloat lmScore, BaseFloat amScore, int32 numWords);
};

}