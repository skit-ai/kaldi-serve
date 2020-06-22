// config.cpp - Configuration Implementation

// stl includes
#include <iostream>
#include <ctime>
#include <string>
#include <chrono>
#include <sstream>

// local includes
#include "config.hpp"


namespace kaldiserve {

void print_version() {
    std::cout << VERSION << std::endl;
    exit(EXIT_SUCCESS);
}

std::string timestamp_now() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::string now_str = std::string(std::asctime(std::localtime(&now_time)));
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    std::stringstream time_string;
    time_string << now_str.substr(0, now_str.size() - 6) << "." << (millis % 1000);
    return time_string.str();
}

} // namespace kaldiserve