#include "config.hpp"

// C++ stl includes
#include <iostream>
#include <memory>
#include <string>

// Local includes
#include "decoder.hpp"
#include "server.hpp"

// Vendor includes
#include "vendor/CLI11.hpp"

int main(int argc, char *argv[]) {
    CLI::App app{"Kaldi gRPC server"};

    std::string model_dir;
    app.add_option("-m,--model-dir", model_dir, "Model root directory. This is a temporary API for testing.")
        ->required()
        ->check(CLI::ExistingDirectory);

    int num_decoders = NUM_DECODERS;
    app.add_option("-n,--num-decoders", num_decoders, "Number of decoders to initialize in the concurrent queue.")
        ->check(CLI::Number);

    CLI11_PARSE(app, argc, argv);

    // runs the server
    run_server(model_dir, num_decoders);

    return 0;
}