/*
 * Utility functions.
 */

// Include guard
#pragma once

// C++ stl includes
#include <string>
#include <vector>

// Vendor includes
#include "vendor/cpptoml.h"

// Fills a list of model specifications from the config
void parse_model_specs(std::string &toml_path, std::vector<ModelSpec> &model_specs) {
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

void string_join(const std::vector<std::string> &strings, std::string separator, std::string &output) {
    output.clear();

    for (auto i = 0; i < strings.size(); i++) {
        output += strings[i];

        if (i != strings.size() - 1) {
            output += separator;
        }
    }
}
