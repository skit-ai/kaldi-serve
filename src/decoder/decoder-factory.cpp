// decoder-factory.cpp - Decoder Factory Implementation

// local includes
#include "types.hpp"
#include "model.hpp"
#include "decoder.hpp"


namespace kaldiserve {

DecoderFactory::DecoderFactory(const ModelSpec &model_spec) : model_spec(model_spec) {
    model_ = std::make_unique<ChainModel>(model_spec);
}

} // namespace kaldiserve