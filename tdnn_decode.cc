#include <Python.h>
#include <string>

#include "feat/wave-reader.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "util/kaldi-thread.h"
#include "nnet3/nnet-utils.h"

#define VERBOSE 0
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

      std::string CInfer(std::string wav_file_path) {
        using namespace fst;

        BaseFloat chunk_length_secs = 1;
        int32 num_err = 0;
        double tot_like = 0.0;

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
        bool end_of_utterance = true;
        decoder.GetLattice(end_of_utterance, &clat);

        std::string answer = "";
        get_decoded_string(word_syms, clat, &tot_like, answer);

        #if VERBOSE
          KALDI_LOG << "Decoded string: " << answer;
        #endif

        return answer;
      }

      ~Model() {
        delete decode_fst;
        delete word_syms;  // will delete if non-NULL.
      }

    private:
      fst::SymbolTable *word_syms;
      LatticeFasterDecoderConfig lattice_faster_decoder_config;
      OnlineNnet2FeaturePipelineInfo *feature_info;
      nnet3::AmNnetSimple am_nnet;
      TransitionModel trans_model;
      nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
      fst::Fst<fst::StdArc> *decode_fst;

      void get_decoded_string(const fst::SymbolTable *word_syms, const CompactLattice &clat,
              double *tot_like, std::string& answer) {
        if (clat.NumStates() == 0) {
          KALDI_LOG << "Empty lattice.";
          return;
        }
        CompactLattice best_path_clat;
        CompactLatticeShortestPath(clat, &best_path_clat);

        Lattice best_path_lat;
        ConvertLattice(best_path_clat, &best_path_lat);

        double             likelihood;
        LatticeWeight      weight;
        int32              num_frames;
        std::vector<int32> alignment;
        std::vector<int32> words;
        
        GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);
        num_frames = alignment.size();
        likelihood = -(weight.Value1() + weight.Value2());
        *tot_like += likelihood;

        #if VERBOSE
          KALDI_LOG << "Likelihood per frame is " << (likelihood / num_frames) 
                << " over " << num_frames << " frames.";
        #endif

        // std::string decoded_string = "";
        for (size_t i = 0; i < words.size(); i++) {
          std::string s = word_syms->Find(words[i]);
          if (s == "") {
            KALDI_LOG << "Word-id " << words[i] << " not in symbol table.";
          }
          answer += s + " ";
        }
      }
  };
}

static void capsule_destructor(PyObject* capsule) {
  kaldi::Model *model = (kaldi::Model*)PyCapsule_GetPointer(capsule, CAPSULE_NAME);
  delete model;
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

  kaldi::Model* model = (kaldi::Model*) PyMem_Malloc(sizeof(kaldi::Model*));
  model = new kaldi::Model(beam, max_active, min_active, lattice_beam, acoustic_scale,
    frame_subsampling_factor, word_syms_filename, model_in_filename, fst_in_str,
    mfcc_config, ie_conf_filename
  );

  PyObject* model_py = PyCapsule_New((void*)model, CAPSULE_NAME, capsule_destructor);

  if (!model_py)
    return NULL;

  return model_py;
}

static PyObject* infer(PyObject* self, PyObject* args) {
  PyObject* model_py;
  char* wav_file_path;

  if (!PyArg_ParseTuple(args, "Os", &model_py, &wav_file_path)) return NULL;

  kaldi::Model* model = (kaldi::Model*)PyCapsule_GetPointer(model_py, CAPSULE_NAME);
  return Py_BuildValue("s", (char*)model->CInfer(wav_file_path).c_str());
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