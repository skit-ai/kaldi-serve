// Configuration options.
#pragma once

// stl includes
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "config.hpp"


namespace kaldiserve {

// Model Specification for Kaldi ASR
// contains model config for a particular model.
struct ModelSpec {
    std::string name;
    std::string language_code;
    std::string path;
    int n_decoders = 1;

    // decoding parameters
    int min_active = 200;
    int max_active = 7000;
    int frame_subsampling_factor = 3;
    float beam = 16.0;
    float lattice_beam = 6.0;
    float acoustic_scale = 1.0;
    float silence_weight = 1.0;
    
    // rnnlm config
    int max_ngram_order = 3;
    float rnnlm_weight = 0.5;
    std::string bos_index = "1";
    std::string eos_index = "2";
};

struct Word {
    float start_time, end_time, confidence;
    std::string word;
};

// An alternative defines a single hypothesis and certain details about the
// parse (only scores for now).
struct Alternative {
    std::string transcript;
    double confidence;
    float am_score, lm_score;
    std::vector<Word> words;
};

// Options for decoder
struct DecoderOptions {
    bool enable_word_level;
    bool enable_rnnlm;
};

// Result for one continuous utterance
using utterance_results_t = std::vector<Alternative>;

// a pair of model_name and language_code
using model_id_t = std::pair<std::string, std::string>;

} // namespace kaldiserve