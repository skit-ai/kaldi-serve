// Utility functions.
#pragma once

// stl includes
#include <string>
#include <vector>

// local includes
#include "config.hpp"
#include "types.hpp"


namespace kaldiserve {

// If the provided path is relative, expand by prefixing the root_path
std::string expand_relative_path(std::string path, std::string root_path);

std::string join_path(std::string a, std::string b);

bool exists(std::string path);

// Fills a list of model specifications from the config
void parse_model_specs(const std::string &toml_path, std::vector<ModelSpec> &model_specs);

// Joins vector of strings together using a separator token
void string_join(const std::vector<std::string> &strings, std::string separator, std::string &output);

} // namespace kaldiserve