// Configuration options.
#pragma once

// stl includes
#include <string>

// definitions
#define VERSION "0.4.3"
#define ENDL '\n'
// #define DEBUG false


namespace kaldiserve {

static bool DEBUG = false;

// prints library version
void print_version();

// returns current timestamp
std::string timestamp_now();

} // namespace kaldiserve