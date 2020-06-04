// Configuration options.
#pragma once

// #define DEBUG false
#define ENDL '\n'

// stl includes
#include <iostream>
#include <string>
#include <ctime>
#include <chrono>

// kaldi includes
#include "base/kaldi-types.h"



#if __cplusplus == 201103L
  //make_unique() implementation for C++11 & 14 from https://stackoverflow.com/questions/17902405/how-to-implement-make-unique-function-in-c11/17902439#17902439
  #include <cstddef>
  #include <memory>
  #include <type_traits>
  #include <utility>

  namespace std {
    template<class T> struct _Unique_if {
        typedef unique_ptr<T> _Single_object;
    };

    template<class T> struct _Unique_if<T[]> {
        typedef unique_ptr<T[]> _Unknown_bound;
    };

    template<class T, size_t N> struct _Unique_if<T[N]> {
        typedef void _Known_bound;
    };

    template<class T, class... Args>
        typename _Unique_if<T>::_Single_object
        make_unique(Args&&... args) {
            return unique_ptr<T>(new T(std::forward<Args>(args)...));
        }

    template<class T>
        typename _Unique_if<T>::_Unknown_bound
        make_unique(size_t n) {
            typedef typename remove_extent<T>::type U;
            return unique_ptr<T>(new U[n]());
        }

    template<class T, class... Args>
        typename _Unique_if<T>::_Known_bound
        make_unique(Args&&...) = delete;
  }
#endif

bool DEBUG = false;

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

std::string timestamp_now() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::string now_str = std::string(std::asctime(std::localtime(&now_time)));
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    std::stringstream time_string;
    time_string << now_str.substr(0, now_str.size() - 6) << "." << (millis % 1000);
    return time_string.str();
}
