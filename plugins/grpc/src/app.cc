// app.cc - gRPC Application Entry

// stl includes
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <stdlib.h>

// lib includes
#include <kaldiserve/decoder.hpp>
#include <kaldiserve/utils.hpp>

// local includes
#include "config.hpp"
#include "server.hpp"

// vendor includes
#include "vendor/CLI11.hpp"

using namespace kaldiserve;


int main(int argc, char *argv[]) {
    CLI::App app{"Kaldi gRPC server"};

    std::string model_spec_toml;
    app.add_option("model_spec_toml", model_spec_toml, "Path to toml specifying models to load")
      ->required()
      ->check(CLI::ExistingFile);

    app.add_flag("-d,--debug", DEBUG, "Flag to enable debug mode");

    app.add_flag_callback("-v,--version", print_version, "Show program version and exit");

    CLI11_PARSE(app, argc, argv);

    std::vector<ModelSpec> model_specs;
    parse_model_specs(model_spec_toml, model_specs);

    if (model_specs.size() == 0) {
        std::cout << ":: No model found in toml for loading" << ENDL;
        return 1;
    }

    std::cout << ":: Loading " << model_specs.size() << " models" << ENDL;
    for (auto const &model_spec : model_specs) {
        std::cout << "::   - " << model_spec.name + " (" + model_spec.language_code + ")" << ENDL;
    }

    run_server(model_specs);

    return 0;
}
