/*
 * Utility functions.
 */

// Include guard
#pragma once

#include "config.hpp"

// C++ stl includes
#include <filesystem>
#include <string>
#include <vector>

// Vendor includes
#include "vendor/cpptoml.h"

// If the provided path is relative, expand by prefixing the root_path
std::string expand_relative_path(std::string path, std::string root_path) {
  std::filesystem::path fs_path(path);
  if (fs_path.is_absolute()) {
    return path;
  } else {
    std::filesystem::path fs_root_path(root_path);
    return (fs_root_path / fs_path).string();
  }
}

std::string join_path(std::string a, std::string b) {
  std::filesystem::path fs_a(a);
  std::filesystem::path (b);
  return (fs_a / fs_b).string();
}

// Fills a list of model specifications from the config
void parse_model_specs(const std::string &toml_path, std::vector<ModelSpec> &model_specs) {
    auto config = cpptoml::parse_file(toml_path);
    auto models = config->get_table_array("model");

    ModelSpec spec;
    for (const auto &model : *models) {
        auto maybe_path = model->get_as<std::string>("path");
        auto maybe_name = model->get_as<std::string>("name");
        auto maybe_language_code = model->get_as<std::string>("language_code");
        auto maybe_n_decoders = model->get_as<std::size_t>("n_decoders");

        // TODO: Throw error in case of invalid toml
        spec.path = *maybe_path;
        spec.name = *maybe_name;
        spec.language_code = *maybe_language_code;

        if (maybe_n_decoders) {
            spec.n_decoders = *maybe_n_decoders;
        } else {
            spec.n_decoders = 1;
        }
        model_specs.push_back(spec);
    }
}

// Joins vector of strings together using a separator token
void string_join(const std::vector<std::string> &strings, std::string separator, std::string &output) {
    output.clear();

    for (auto i = 0; i < strings.size(); i++) {
        output += strings[i];

        if (i != strings.size() - 1) {
            output += separator;
        }
    }
}
