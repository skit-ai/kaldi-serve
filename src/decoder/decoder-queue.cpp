// decoder-queue.cpp - Decoder Queue Implementation

// local includes
#include "config.hpp"
#include "decoder.hpp"
#include "types.hpp"


namespace kaldiserve {

DecoderQueue::DecoderQueue(const ModelSpec &model_spec) {
    std::cout << ":: Loading model from " << model_spec.path << ENDL;

    decoder_factory_ = make_uniq<DecoderFactory>(model_spec);
    for (size_t i = 0; i < model_spec.n_decoders; i++) {
        queue_.push(decoder_factory_->produce());
    }
}

DecoderQueue::~DecoderQueue() {
    while (!queue_.empty()) {
        auto decoder = queue_.front();
        queue_.pop();
        delete decoder;
    }
}

void DecoderQueue::push_(Decoder *const item) {
    std::unique_lock<std::mutex> mlock(mutex_);
    queue_.push(item);
    mlock.unlock();
    cond_.notify_one(); // condition var notifies another suspended thread (help up in `pop`)
}

Decoder *DecoderQueue::pop_() {
    std::unique_lock<std::mutex> mlock(mutex_);
    // waits until a decoder object is available
    while (queue_.empty()) {
        // suspends current thread execution and awaits condition notification
        cond_.wait(mlock);
    }
    auto item = queue_.front();
    queue_.pop();
    return item;
}

} // namespace kaldiserve