// Configuration options.
#pragma once

// stl includes
#include <iostream>
#include <string>
#include <utility>

// kaldi includes
#include "base/kaldi-types.h"

// definitions
#define VERSION "0.4.3"
#define ENDL '\n'
// #define DEBUG false


namespace kaldiserve {

static bool DEBUG = false;

// a pair of model_name and language_code
using model_id_t = std::pair<std::string, std::string>;

// Model Specification for Kaldi ASR
// contains model config for a particular model.
struct ModelSpec {
    std::string name;
    std::string language_code;
    std::string path;
    std::size_t n_decoders = 1;

    // decoding parameters
    kaldi::BaseFloat beam = 16.0;
    std::size_t min_active = 200;
    std::size_t max_active = 7000;
    kaldi::BaseFloat lattice_beam = 6.0;
    kaldi::BaseFloat acoustic_scale = 1.0;
    std::size_t frame_subsampling_factor = 3;
    kaldi::BaseFloat silence_weight = 1.0;
    // rnnlm config
    kaldi::int32 max_ngram_order = 3;
    kaldi::BaseFloat rnnlm_weight = 0.5;
    std::string bos_index = "1";
    std::string eos_index = "2";
};

// prints library version
void print_version();

// returns current timestamp
std::string timestamp_now();

} // namespace kaldiserve