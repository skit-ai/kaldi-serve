#include "config.hpp"

// C++ stl includes
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// Local includes
#include "decoder.hpp"
#include "server.hpp"
#include "utils.hpp"

// Vendor includes
#include "vendor/CLI11.hpp"
#include "vendor/cpptoml.h"

int main(int argc, char *argv[]) {
    CLI::App app{"Kaldi gRPC server"};

    std::string model_spec_toml;
    app.add_option("model_spec_toml", model_spec_toml, "Path to toml specifying models to load.")
        ->required()
        ->check(CLI::ExistingFile);

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
