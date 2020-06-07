// decoder-factory.cpp - Decoder Factory Implementation

// local includes
#include "decoder.hpp"
#include "model.hpp"


namespace kaldiserve {

DecoderFactory::DecoderFactory(const ModelSpec &model_spec) : model_spec(model_spec) {
    model_ = std::make_unique<ChainModel>(model_spec);
}

} // namespace kaldiserve