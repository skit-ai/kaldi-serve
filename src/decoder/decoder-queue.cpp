// decoder-queue.cpp - Decoder Queue Implementation

// local includes
#include "decoder.hpp"


namespace kaldiserve {

DecoderQueue::DecoderQueue(const ModelSpec &model_spec) {
    std::cout << ":: Loading model from " << model_spec.path << ENDL;

    std::chrono::system_clock::time_point start_time;
    if (DEBUG) {
        // LOG MODELS LOAD TIME --> START
        start_time = std::chrono::system_clock::now();
    }
    decoder_factory_ = std::unique_ptr<DecoderFactory>(new DecoderFactory(model_spec));
    for (size_t i = 0; i < model_spec.n_decoders; i++) {
        queue_.push(decoder_factory_->produce());
    }

    if (DEBUG) {
        std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
        // LOG MODELS LOAD TIME --> END
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << ":: Decoder models concurrent queue init in: " << ms.count() << "ms" << ENDL;
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