#include "config.hpp"

// C++ stl includes
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// Local includes
#include "decoder.hpp"
#include "server.hpp"

// Vendor includes
#include "vendor/CLI11.hpp"
#include "vendor/cpptoml.h"

// Return a list of model specifications from the config
std::vector<ModelSpec> parse_model_specs(std::string &toml_path) {
    auto config = cpptoml::parse_file(toml_path);
    auto models = config->get_table_array("model");
    std::vector<ModelSpec> model_specs;

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

    return model_specs;
}

int main(int argc, char *argv[]) {
    CLI::App app{"Kaldi gRPC server"};

    std::string model_spec_toml;
    app.add_option("model_spec_toml", model_spec_toml, "Path to toml specifying models to load.")
        ->required()
        ->check(CLI::ExistingFile);

    CLI11_PARSE(app, argc, argv);

    std::vector<ModelSpec> model_specs = parse_model_specs(model_spec_toml);
    if (model_specs.size() == 0) {
        std::cout << ":: No model found in toml for loading" << std::endl;
        return 1;
    } else {
        std::cout << ":: Loading " << model_specs.size() << " models" << std::endl;
        for (auto const &model_spec : model_specs) {
            std::cout << "::   - " << model_spec.name + " (" + model_spec.language_code + ")" << std::endl;
        }
        run_server(model_specs);
        return 0;
    }
}
