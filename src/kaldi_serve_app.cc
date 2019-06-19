#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <grpc/grpc.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/security/server_credentials.h>

#include "kaldi_serve.grpc.pb.h"

#include "vendor/CLI11.hpp"
#include "utils.hpp"

// Kaldi includes from here
#include "feat/wave-reader.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "nnet3/nnet-utils.h"
#include "online2/online-endpoint.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/onlinebin-util.h"
#include "util/kaldi-thread.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerReaderWriter;
using grpc::ServerWriter;
using grpc::Status;

using kaldi_serve::KaldiServe;

using kaldi_serve::RecognitionAudio;
using kaldi_serve::RecognitionConfig;
using kaldi_serve::RecognizeRequest;
using kaldi_serve::RecognizeResponse;
using kaldi_serve::SpeechContext;
using kaldi_serve::SpeechRecognitionAlternative;
using kaldi_serve::SpeechRecognitionResult;

using namespace std;
using std::chrono::system_clock;


// An alternative is a pair of string (hypothesis) and it's confidence
using alternative_t = std::pair<std::string, double>;

// An alignment tells start and end time (consider total audio to be of length 1) of a word
using alignment_t = std::tuple<std::string, float, float>;

// Result for one continuous utterance
using utterance_results_t = std::vector<std::pair<alternative_t, std::vector<alignment_t>>>;


// Find confidence by merging lm and am scores. Taken from
// https://github.com/dialogflow/asr-server/blob/master/src/OnlineDecoder.cc#L90
// NOTE: This might not be very useful for us right now. Depending on the
//       situation, we might actually want to weigh components differently.
double calculate_confidence(float lm_score, float am_score, std::size_t n_words) {
  return std::max(0.0, std::min(1.0, -0.0001466488 * (2.388449 * lm_score + am_score) / (n_words + 1) + 0.956));
}


// Return n-best alternative from lattice. Output symbols are converted to words
// based on word-syms.
utterance_results_t find_alternatives(const fst::SymbolTable *word_syms,
                                      const kaldi::CompactLattice &clat,
                                      std::size_t n_best) {
  utterance_results_t results;

  if (clat.NumStates() == 0) {
    KALDI_LOG << "Empty lattice.";
    return results;
  }

  kaldi::Lattice *lat = new kaldi::Lattice();
  fst::ConvertLattice(clat, lat);

  kaldi::Lattice nbest_lat;
  std::vector<kaldi::Lattice> nbest_lats;
  fst::ShortestPath(*lat, &nbest_lat, n_best);
  fst::ConvertNbestToVector(nbest_lat, &nbest_lats);

  if (nbest_lats.empty()) {
    KALDI_WARN << "no N-best entries";
  } else {
    for (auto const& l: nbest_lats) {
      kaldi::LatticeWeight weight;
      // NOTE: Check why int32s specifically are used here
      std::vector<int32> input_ids;
      std::vector<int32> word_ids;
      fst::GetLinearSymbolSequence(l, &input_ids, &word_ids, &weight);

      std::vector<std::string> words;
      for (auto const& wid: word_ids) {
        words.push_back(word_syms->Find(wid));
      }

      // An alignment is a tuple like (word, start, end). Note that the start
      // and end time at the moment are not based on any unit and represent
      // relative values depending on the total number of input symbols. This,
      // when multiplied into audio lengths should give actual second values.
      std::vector<alignment_t> alignments;
      bool in_word = false;
      if (words.size() > 0) {
        size_t n_input_tokens = input_ids.size();
        // HACK: We assume inputs below 3 to be whitespaces. This is not 'the'
        //       right way since we haven't looked into the exact set of input
        //       symbols.
        float start, end;
        int word_index = 0;
        for (auto i = 0; i < n_input_tokens; i++) {
          if (input_ids[i] < 3) {
            if (in_word) {
              // Exited a word
              end = (i - 1) / (float)n_input_tokens;
              alignments.push_back(std::make_tuple(words[word_index], start, end));
              word_index += 1;
            }
            in_word = false;
          } else {
            if (!in_word) {
              // Entered a new word
              start = (i - 1) / (float)n_input_tokens;
            }
            in_word = true;
          }
        }
      }

      std::string sentence;
      string_join(words, " ", sentence);

      alternative_t alt = std::make_pair(sentence, calculate_confidence(float(weight.Value1()),
                                                                        float(weight.Value2()),
                                                                        word_ids.size()));
      results.push_back(std::make_pair(alt, alignments));
    }
  }
  return results;
}


class Decoder {
public:
  Decoder(kaldi::BaseFloat beam, std::size_t max_active, std::size_t min_active,
          kaldi::BaseFloat lattice_beam, kaldi::BaseFloat acoustic_scale,
          std::size_t frame_subsampling_factor, std::string word_syms_filepath,
          std::string model_filepath, std::string mfcc_conf_filepath,
          std::string ie_conf_filepath, const fst::Fst<fst::StdArc>* decode_fst);

  ~Decoder();

  utterance_results_t decode_file(std::string wav_filepath, std::size_t n_best);
  utterance_results_t decode_stream(std::istream& wav_stream, std::size_t n_best);

private:
  const fst::Fst<fst::StdArc>* decode_fst;
  fst::SymbolTable *word_syms;
  kaldi::LatticeFasterDecoderConfig lattice_faster_decoder_config;
  kaldi::OnlineNnet2FeaturePipelineInfo *feature_info;
  kaldi::nnet3::AmNnetSimple am_nnet;
  kaldi::TransitionModel trans_model;
  kaldi::nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
};


Decoder::Decoder(kaldi::BaseFloat beam, std::size_t max_active,
                 std::size_t min_active, kaldi::BaseFloat lattice_beam,
                 kaldi::BaseFloat acoustic_scale, std::size_t frame_subsampling_factor,
                 std::string word_syms_filepath, std::string model_filepath,
                 std::string mfcc_conf_filepath, std::string ie_conf_filepath,
                 const fst::Fst<fst::StdArc> *decode_fst): decode_fst(decode_fst) {
  try {
    kaldi::OnlineNnet2FeaturePipelineConfig feature_opts;

    feature_opts.mfcc_config = mfcc_conf_filepath;
    feature_opts.ivector_extraction_config = ie_conf_filepath;
    lattice_faster_decoder_config.max_active = max_active;
    lattice_faster_decoder_config.min_active = min_active;
    lattice_faster_decoder_config.beam = beam;
    lattice_faster_decoder_config.lattice_beam = lattice_beam;
    decodable_opts.acoustic_scale = acoustic_scale;
    decodable_opts.frame_subsampling_factor = frame_subsampling_factor;

    {
      bool binary;
      kaldi::Input ki(model_filepath, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
      kaldi::nnet3::SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
      kaldi::nnet3::SetDropoutTestMode(true, &(am_nnet.GetNnet()));
      kaldi::nnet3::CollapseModel(kaldi::nnet3::CollapseModelConfig(), &(am_nnet.GetNnet()));
    }

    word_syms = NULL;
    if (word_syms_filepath != "" && !(word_syms = fst::SymbolTable::ReadText(word_syms_filepath))) {
      KALDI_ERR << "Could not read symbol table from file " << word_syms_filepath;
    }

    feature_info = new kaldi::OnlineNnet2FeaturePipelineInfo(feature_opts);
  } catch (const std::exception &e) {
    KALDI_ERR << e.what();
  }
}


Decoder::~Decoder() {
  delete word_syms;
}


utterance_results_t Decoder::decode_file(std::string wav_filepath, std::size_t n_best) {
  std::ifstream wav_stream(wav_filepath, std::ifstream::binary);
  return decode_stream(wav_stream, n_best);
}

utterance_results_t Decoder::decode_stream(std::istream& wav_stream, std::size_t n_best) {
  kaldi::OnlineIvectorExtractorAdaptationState adaptation_state(feature_info->ivector_extractor_info);
  kaldi::OnlineNnet2FeaturePipeline feature_pipeline(*feature_info);
  feature_pipeline.SetAdaptationState(adaptation_state);

  kaldi::OnlineSilenceWeighting silence_weighting(trans_model, feature_info->silence_weighting_config,
                                                  decodable_opts.frame_subsampling_factor);
  kaldi::nnet3::DecodableNnetSimpleLoopedInfo decodable_info(decodable_opts, &am_nnet);

  kaldi::SingleUtteranceNnet3Decoder decoder(lattice_faster_decoder_config,
                                             trans_model, decodable_info, *decode_fst,
                                             &feature_pipeline);

  kaldi::WaveData wave_data;
  wave_data.Read(wav_stream);

  // get the data for channel zero (if the signal is not mono, we only
  // take the first channel).
  kaldi::SubVector<kaldi::BaseFloat> data(wave_data.Data(), 0);
  kaldi::BaseFloat samp_freq = wave_data.SampFreq();

  int32 chunk_length;
  kaldi::BaseFloat chunk_length_secs = 1;
  if (chunk_length_secs > 0) {
    chunk_length = int32(samp_freq * chunk_length_secs);
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
    feature_pipeline.AcceptWaveform(samp_freq, wave_part);

    samp_offset += num_samp;
    if (samp_offset == data.Dim()) {
      feature_pipeline.InputFinished();
    }

    if (silence_weighting.Active() && feature_pipeline.IvectorFeature() != NULL) {
      silence_weighting.ComputeCurrentTraceback(decoder.Decoder());
      silence_weighting.GetDeltaWeights(feature_pipeline.NumFramesReady(),
                                        &delta_weights);
      feature_pipeline.IvectorFeature()->UpdateFrameWeights(delta_weights);
    }
    decoder.AdvanceDecoding();
  }
  decoder.FinalizeDecoding();

  kaldi::CompactLattice clat;
  decoder.GetLattice(true, &clat);

  return find_alternatives(word_syms, clat, n_best);
}


// Factory for creating decoders with shared decoding graph.
class DecoderFactory {
public:
  DecoderFactory(std::string hclg_filepath) {
    decode_fst = fst::ReadFstKaldiGeneric(hclg_filepath);
  }

  Decoder* operator()(kaldi::BaseFloat beam, std::size_t max_active,
                      std::size_t min_active, kaldi::BaseFloat lattice_beam,
                      kaldi::BaseFloat acoustic_scale, std::size_t frame_subsampling_factor,
                      std::string word_syms_filepath,
                      std::string model_filepath,
                      std::string mfcc_conf_filepath,
                      std::string ie_conf_filepath) {
    return new Decoder(beam, max_active, min_active, lattice_beam,
                       acoustic_scale, frame_subsampling_factor,
                       word_syms_filepath, model_filepath,
                       mfcc_conf_filepath, ie_conf_filepath, decode_fst);
  }

  ~DecoderFactory() {
    delete decode_fst;
  };

  const fst::Fst<fst::StdArc>* decode_fst;
};

// HACK: This global var will go away;
Decoder *decoder;

void transcribe(const RecognitionConfig* config, const RecognitionAudio* audio,
                const string uuid, RecognizeResponse* recognizeResponse){
  std::cout << "UUID: \t" << uuid << std::endl;

  std::cout << "Encoding: " << config->encoding() << std::endl;
  std::cout << "Sample Rate Hertz: " << config->sample_rate_hertz() << std::endl;
  std::cout << "Language Code: " << config->language_code() << std::endl;
  std::cout << "Max Alternatives: " << config->max_alternatives() << std::endl;
  std::cout << "Punctuation: " << config->punctuation() << std::endl;
  std::cout << "Model: " << config->model() << std::endl;

  std::cout << "NOTE: We are ignoring most of the parameters, other than the audio bytes, for now."
            << std::endl;

  SpeechRecognitionResult* results = recognizeResponse->add_results();
  SpeechRecognitionAlternative* alternative;

  int32 n_best = config->max_alternatives();
  std::istringstream input_stream(audio->content());
  for (auto const &res : decoder->decode_stream(input_stream, n_best)) {
    alternative = results->add_alternatives();
    alternative->set_transcript(res.first.first);
    alternative->set_confidence(res.first.second);
  }

  return;
}


class KaldiServeImpl final : public KaldiServe::Service {
public:
  explicit KaldiServeImpl() {
  }

  Status Recognize(ServerContext* context, const RecognizeRequest* recognizeRequest,
                   RecognizeResponse* recognizeResponse) override {
    RecognitionConfig config;
    RecognitionAudio audio;
    string uuid;

    config = recognizeRequest->config();
    audio  = recognizeRequest->audio();
    uuid   = recognizeRequest->uuid();

    transcribe(&config, &audio, uuid, recognizeResponse);

    return Status::OK;
  }
};


void run_server() {
  std::string server_address("0.0.0.0:5016");
  KaldiServeImpl service;

  ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);
  std::unique_ptr<Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;
  server->Wait();
}

int main(int argc, char* argv[]) {
  CLI::App app{"Kaldi gRPC server"};

  std::string model_dir;
  app.add_option("model-dir", model_dir, "Model root directory. This is a temporary API for testing.")
    ->required()
    ->check(CLI::ExistingDirectory);

  CLI11_PARSE(app, argc, argv);

  std::cout << ":: Loading model from " << model_dir << std::endl;

  // TODO: Better organize a kaldi model for distribution
  std::string hclg_filepath = model_dir + "/tree_a_sp/graph/HCLG.fst";
  std::string words_filepath = model_dir + "/tree_a_sp/graph/words.txt";
  std::string model_filepath = model_dir + "/tdnn1g_sp_online/final.mdl";
  std::string mfcc_conf_filepath = model_dir + "/tdnn1g_sp_online/conf/mfcc.conf";
  std::string ivec_conf_filepath = model_dir + "/tdnn1g_sp_online/conf/ivector_extractor.conf";

  DecoderFactory decoder_factory(hclg_filepath);
  Decoder* decoder = decoder_factory(13.0, 7000, 200, 6.0, 1.0, 3,
                                     words_filepath, model_filepath,
                                     mfcc_conf_filepath, ivec_conf_filepath);

  run_server();
  return 0;
}
