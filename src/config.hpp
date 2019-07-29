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

#include <boost/version.hpp>

constexpr bool BOOST_1_67_X = ((BOOST_VERSION / 100000) >= 1) && (((BOOST_VERSION / 100) % 1000) >= 67);

// boost headers for hash functions was changed after version 1_67_X
#if BOOST_1_67_X
#include <boost/container_hash/hash.hpp>
#else
#include <boost/functional/hash.hpp>
#endif

// custom hash function for model id type (pair of strings) for use as key in unordered_map
struct model_id_hash {
    std::size_t operator () (const model_id_t &id) const {
        std::size_t seed = 0;
        boost::hash_combine(seed, id.first);
        boost::hash_combine(seed, id.second);
        return seed;
    }
};