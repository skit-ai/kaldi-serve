// C++ stl includes
#include <chrono>
#include <iostream>
#include <memory>
#include <string>

// Vendor includes
#include "vendor/CLI11.hpp"

// Local includes
#include "decoder.hpp"
#include "server.hpp"

int main(int argc, char *argv[]) {
    CLI::App app{"Kaldi gRPC server"};

    std::string model_dir;
    app.add_option("model-dir", model_dir, "Model root directory. This is a temporary API for testing.")
        ->required()
        ->check(CLI::ExistingDirectory);

    CLI11_PARSE(app, argc, argv);

    std::cout << ":: Loading model from " << model_dir << std::endl;

    // TODO: Better organize a kaldi model for distribution
    std::string hclg_filepath = model_dir + "/tree_a_sp/graph/HCLG.fst";
    std::string words_filepath = model_dir + "/tree_a_sp/graph/words.txt";
    std::string model_filepath = model_dir + "/tdnn1g_sp_online/final.mdl";
    std::string mfcc_conf_filepath = model_dir + "/tdnn1g_sp_online/conf/mfcc.conf";
    std::string ivec_conf_filepath = model_dir + "/tdnn1g_sp_online/conf/ivector_extractor.conf";

    // LOG MODEL LOAD TIME --> START
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    start = std::chrono::high_resolution_clock::now();

    DecoderFactory decoder_factory(hclg_filepath);
    std::shared_ptr<Decoder> decoder = decoder_factory(13.0, 7000, 200, 6.0, 1.0, 3,
                                                  words_filepath, model_filepath,
                                                  mfcc_conf_filepath, ivec_conf_filepath);

    end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> load_duration = end - start;

    std::cout << "decoder model loaded in: " << load_duration.count() << 's' << std::endl;
    // LOG MODEL LOAD TIME --> END

    // runs the server using the loaded decoder model
    run_server(decoder);

    return 0;
}