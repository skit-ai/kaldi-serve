// Configuration options.
#pragma once

// stl includes
#include <string>
#include <memory>

// definitions
#define VERSION "1.0.0"
#define ENDL '\n'
// #define DEBUG false


namespace kaldiserve {

template<typename T, typename... Args>
std::unique_ptr<T> make_uniq(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

static bool DEBUG = false;

// prints library version
void print_version();

// returns current timestamp
std::string timestamp_now();

} // namespace kaldiserve