// batched-gpu-decoder.cc - Batch Decoding Binary (GPU)

// stl includes
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cstring>
#include <string.h>
#include <stdlib.h>

// lib includes
#include <kaldiserve/decoder.hpp>
#include <kaldiserve/types.hpp>
#include <kaldiserve/utils.hpp>

// vendor includes
#include "vendor/CLI11.hpp"

namespace ks = kaldiserve;


int main(int argc, char const *argv[]) {
    CLI::App app{"Kaldi Batch Decoding binary (GPU)"};

    std::string model_spec_toml, audio_paths_file, output_dir;
    app.add_option("model_spec_toml", model_spec_toml, "Path to toml specifying models to load")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("audio_paths_file", audio_paths_file, "Path to txt file specifying key and path to audio files")
        ->required()
        ->check(CLI::ExistingFile);
    // app.add_option("output_dir", output_dir, "Path to dir where the transcription results should be written")
    //     ->required()
    //     ->check(CLI::ExistingDirectory);
    CLI11_PARSE(app, argc, argv);

    // parse model spec
    std::vector<ks::ModelSpec> model_specs;
    ks::parse_model_specs(model_spec_toml, model_specs);
    if (model_specs.size() == 0) {
        std::cout << ":: No model found in toml for loading" << std::endl;
        return 1;
    }

    // read audio paths
    std::unordered_map<std::string, std::string> audios;

    std::ifstream audio_paths(audio_paths_file);
    std::string line;
    while (std::getline(audio_paths, line)) {
        const char *delim = std::string(",").c_str();

        // Returns first token
        char *line_c = new char[line.length() + 1];
        strcpy(line_c, line.c_str());

        std::string key, value;
        char *token = strtok(line_c, delim);
        key = std::string(token);
        token = strtok(NULL, delim);
        value = std::string(token);

        audios[key] = value;

        delete[] line_c;
    }

    auto model = std::unique_ptr<ks::ChainModel>(new ks::ChainModel(model_specs[0]));
    auto batch_decoder = std::unique_ptr<ks::BatchDecoder>(new ks::BatchDecoder(model.get()));

    batch_decoder->start_decoding();

    for (auto const &audio : audios) {
        std::string key = audio.first;

        // read audio bytes into stream
        std::ifstream wav_stream(audio.second, std::ifstream::binary);

        std::function<void(const std::vector<ks::Alternative>&results)> lambda = [&key, &output_dir](const std::vector<ks::Alternative>&results) {
            // write results to file
            std::cout << key << ":" << std::endl;
            for (auto const &alt : results) {
                std::cout << alt.transcript << std::endl;
            }
            std::cout << std::endl;
        };

        // kick off decoding
        batch_decoder->decode_with_callback(
            wav_stream, 10, false, key, lambda);
    }

    batch_decoder->wait_for_tasks();
    batch_decoder->free_decoder();

    return 0;
}
