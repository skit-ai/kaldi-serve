// decoder-factory.cpp - Decoder Factory Implementation

// local includes
#include "config.hpp"
#include "types.hpp"
#include "model.hpp"
#include "decoder.hpp"


namespace kaldiserve {

DecoderFactory::DecoderFactory(const ModelSpec &model_spec) : model_spec(model_spec) {
    model_ = make_uniq<ChainModel>(model_spec);
}

} // namespace kaldiserve