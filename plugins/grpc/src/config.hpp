#pragma once

// lib includes
#include <kaldiserve/config.hpp>
#include <boost/version.hpp>

using namespace kaldiserve;


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