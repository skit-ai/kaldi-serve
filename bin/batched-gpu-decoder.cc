// batched-gpu-decoder.cc - Batch Decoding Binary (GPU)

// stl includes
#include <iostream>
#include <memory>
#include <string>
#include <vector>
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

    std::string model_spec_toml, audio_paths, output_dir;
    app.add_option("model_spec_toml", model_spec_toml, "Path to toml specifying models to load")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("audio_paths", audio_paths, "Path to txt file specifying key and path to audio files")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("output_dir", output_dir, "Path to dir where the transcription results should be written")
        ->required()
        ->check(CLI::ExistingDirectory);
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


    auto model = std::unique_ptr<ks::ChainModel>(new ks::ChainModel(model_specs[0]));
    auto batch_decoder = std::unique_ptr<ks::BatchDecoder>(new ks::BatchDecoder(model.get()));

    batch_decoder->start_decoding();

    for (auto const &audio : audios) {
        std::string key = audio.first;

        // read audio bytes into stream
        std::ifstream wav_stream(audio.second, std::ifstream::binary);

        // kick off decoding
        batch_decoder->decode_with_callback(
            wav_stream, 10, false, key, [&key, &output_dir](const ks::utterance_results_t &results) {
            // write results to file
            
        });
    }

    batch_decoder->wait_for_tasks();
    batch_decoder->free_decoder();

    return 0;
}
