#include <Python.h>

#include "feat/wave-reader.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "util/kaldi-thread.h"
#include "nnet3/nnet-utils.h"

#define VERBOSE 1
#define CAPSULE_NAME "TDNN_DECODER_MODEL"

namespace kaldi {

  class Model {
    public:
      Model(BaseFloat beam, int32 max_active, int32 min_active,
            BaseFloat lattice_beam, BaseFloat acoustic_scale,
            int32 frame_subsampling_factor, char* &word_syms_filename,
            char* &model_in_filename, char* &fst_in_str,
            char* &mfcc_config, char* &ie_conf_filename) {
          try {
            using namespace kaldi;
            using namespace fst;

            typedef int32 int32;
            typedef int64 int64;

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
            decoder_opts.max_active                    = max_active;
            decoder_opts.min_active                    = min_active;
            decoder_opts.beam                          = beam;
            decoder_opts.lattice_beam                  = lattice_beam;
            decodable_opts.acoustic_scale              = acoustic_scale;
            decodable_opts.frame_subsampling_factor    = frame_subsampling_factor;

            nnet3::AmNnetSimple am_nnet;
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

            // this object contains precomputed stuff that is used by all decodable
            // objects.  It takes a pointer to am_nnet because if it has iVectors it has
            // to modify the nnet to accept iVectors at intervals.
            decodable_info = new nnet3::DecodableNnetSimpleLoopedInfo(decodable_opts, &am_nnet);
            feature_info = new OnlineNnet2FeaturePipelineInfo(feature_opts);
          } catch (const std::exception &e) {
            std::cout << e.what(); // model not loaded
          }
      }

      ~Model() {
        delete decode_fst;
        delete word_syms;  // will delete if non-NULL.
      }

    private:
      fst::SymbolTable *word_syms;
      LatticeFasterDecoderConfig decoder_opts;
      OnlineNnet2FeaturePipelineInfo *feature_info;

      nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
      TransitionModel trans_model;
      nnet3::DecodableNnetSimpleLoopedInfo *decodable_info;
      fst::Fst<fst::StdArc> *decode_fst;
  };

  void GetDiagnosticsAndPrintOutput(const std::string &utt,
                                    const fst::SymbolTable *word_syms,
                                    const CompactLattice &clat,
                                    int64 *tot_num_frames,
                                    double *tot_like) {
    if (clat.NumStates() == 0) {
      std::cout << "Empty lattice.";
      return;
    }
    CompactLattice best_path_clat;
    CompactLatticeShortestPath(clat, &best_path_clat);

    Lattice best_path_lat;
    ConvertLattice(best_path_clat, &best_path_lat);

    double likelihood;
    LatticeWeight weight;
    int32 num_frames;
    std::vector<int32> alignment;
    std::vector<int32> words;
    GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);
    num_frames = alignment.size();
    likelihood = -(weight.Value1() + weight.Value2());
    *tot_num_frames += num_frames;
    *tot_like += likelihood;
    std::cout << "Likelihood per frame for utterance " << utt << " is "
              << (likelihood / num_frames) << " over " << num_frames << " frames.";

    if (word_syms != NULL) {
      std::cerr << utt << ' ';
      for (size_t i = 0; i < words.size(); i++) {
        std::string s = word_syms->Find(words[i]);
        if (s == "") {
          std::cout << "Word-id " << words[i] << " not in symbol table.";
        }
        std::cerr << s << ' ';
      }
      std::cerr << std::endl;
    }
  }

  char* CInfer(int n, int k) {
    using namespace fst;

    BaseFloat chunk_length_secs = 0.18;
    return "hello";
  }

  static void capsule_destructor(PyObject* capsule) {
    Model *model = (Model*)PyCapsule_GetPointer(capsule, CAPSULE_NAME);
    delete model;
  }
}

static PyObject *load_model(PyObject *self, PyObject *args) {
  kaldi::BaseFloat beam;
  int32 max_active;
  int32 min_active;
  kaldi::BaseFloat lattice_beam;
  kaldi::BaseFloat acoustic_scale;
  int32 frame_subsampling_factor;
  char* word_syms_filename;
  char* model_in_filename;
  char* fst_in_str;
  char* mfcc_config;
  char* ie_conf_filename;

  if (!PyArg_ParseTuple(
      args,
      "fiiffisssss",
      &beam, &max_active, &min_active, &lattice_beam,&acoustic_scale, &frame_subsampling_factor,
      &word_syms_filename, &model_in_filename, &fst_in_str,&mfcc_config, &ie_conf_filename
    )
  ) return NULL;


  kaldi::Model model(beam, max_active, min_active, lattice_beam, acoustic_scale,
    frame_subsampling_factor, word_syms_filename, model_in_filename, fst_in_str,
    mfcc_config, ie_conf_filename
  );

  PyObject* model_py = PyCapsule_New((void*)&model, CAPSULE_NAME, kaldi::capsule_destructor);

  if (!model_py)
    return NULL;

  return model_py;
}

static PyObject *infer(PyObject *self, PyObject *args) {
  int n, k;

  if (!PyArg_ParseTuple(args, "ii", &n, &k)) return NULL;
  return Py_BuildValue("s", kaldi::CInfer(n, k));
}

// Our Module's Function Definition struct
// We require this `NULL` to signal the end of our method definition
static PyMethodDef moduleMethods[] = {
  {"load_model", load_model, METH_VARARGS, "Loads TDNN Model"},
  {"infer", infer, METH_VARARGS, "Converts audio to text"},
  {NULL, NULL, 0, NULL}
};

// Our Module Definition struct
static struct PyModuleDef tdnnDecode = {
  PyModuleDef_HEAD_INIT,
  "tdnn_decode",
  "Kaldi bindings for online TDNN decode",
  -1,
  moduleMethods
};

// Initializes our module using our above struct
PyMODINIT_FUNC PyInit_tdnn_decode(void) {
  return PyModule_Create(&tdnnDecode);
}