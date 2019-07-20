#pragma once

#define DEBUG true
#define ENDL '\n'

#if DEBUG
#define NUM_DECODERS 5
#else
#define NUM_DECODERS 20
#endif

#include <iostream>
#include <string>

// Model Specification for Kaldi ASR
// contains model config for a particular model.
struct ModelSpec {
    std::string name;
    std::string language_code;
    std::string path;
    std::size_t n_decoders = 1;
};

// A pair of model_name and language_code
using model_id_t = std::pair<std::string, std::string>;

#include <boost/functional/hash.hpp>

// custom hash function for model id type (pair of strings) for use as key in unordered_map
struct model_id_hash {
    std::size_t operator () (const model_id_t &id) const {
        std::size_t seed = 0;
        boost::hash_combine(seed, id.first);
        boost::hash_combine(seed, id.second);
        return seed;
    }
};