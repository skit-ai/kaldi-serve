// Configuration options.
#pragma once

#define DEBUG false
#define ENDL '\n'

// stl includes
#include <iostream>
#include <string>

// kaldi includes
#include "base/kaldi-types.h"

// Model Specification for Kaldi ASR
// contains model config for a particular model.
struct ModelSpec {
  std::string name;
  std::string language_code;
  std::string path;
  std::size_t n_decoders = 1;

  // decoding parameters
  kaldi::BaseFloat beam = 13.0;
  std::size_t min_active = 200;
  std::size_t max_active = 7000;
  kaldi::BaseFloat lattice_beam = 6.0;
  kaldi::BaseFloat acoustic_scale = 1.0;
  std::size_t frame_subsampling_factor = 3;
};

// a pair of model_name and language_code
using model_id_t = std::pair<std::string, std::string>;

#include <boost/version.hpp>

constexpr bool BOOST_1_67_X = ((BOOST_VERSION / 100000) >= 1) && (((BOOST_VERSION / 100) % 1000) >= 67);

// boost headers for hash functions was changed after version 1_67_X
#if BOOST_1_67_X
#include <boost/container_hash/hash.hpp>
#else
#include <boost/functional/hash.hpp>
#endif

// hash function for model id type (pair of strings) for unordered_map hashing fn
struct model_id_hash {
    std::size_t operator () (const model_id_t &id) const {
        std::size_t seed = 0;
        boost::hash_combine(seed, id.first);
        boost::hash_combine(seed, id.second);
        return seed;
    }
};
