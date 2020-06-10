// utils-io.cpp - I/O Utilities Implementation

// lib includes
#include <boost/filesystem.hpp>

// vendor includes
#include "vendor/cpptoml.h"

// local includes
#include "utils.hpp"
#include "types.hpp"


namespace kaldiserve {

std::string expand_relative_path(std::string path, std::string root_path) {
  boost::filesystem::path fs_path(path);
  if (fs_path.is_absolute()) {
    return path;
  } else {
    boost::filesystem::path fs_root_path(root_path);
    return (fs_root_path / fs_path).string();
  }
}

std::string join_path(std::string a, std::string b) {
  boost::filesystem::path fs_a(a);
  boost::filesystem::path fs_b(b);
  return (fs_a / fs_b).string();
}

bool exists(std::string path) {
  boost::filesystem::path fs_path(path);
  return boost::filesystem::exists(fs_path);
}

void parse_model_specs(const std::string &toml_path, std::vector<ModelSpec> &model_specs) {
    auto config = cpptoml::parse_file(toml_path);
    auto models = config->get_table_array("model");

    ModelSpec spec;
    for (const auto &model : *models) {
        auto maybe_path = model->get_as<std::string>("path");
        auto maybe_name = model->get_as<std::string>("name");
        auto maybe_language_code = model->get_as<std::string>("language_code");
        auto maybe_n_decoders = model->get_as<int>("n_decoders");

        auto maybe_min_active = model->get_as<int>("min_active");
        auto maybe_max_active = model->get_as<int>("max_active");
        auto maybe_frame_subsampling_factor = model->get_as<int>("frame_subsampling_factor");
        auto maybe_beam = model->get_as<double>("beam");
        auto maybe_lattice_beam = model->get_as<double>("lattice_beam");
        auto maybe_acoustic_scale = model->get_as<double>("acoustic_scale");
        auto maybe_silence_weight = model->get_as<double>("silence_weight");
        auto maybe_max_ngram_order = model->get_as<int>("max_ngram_order");
        auto maybe_rnnlm_weight = model->get_as<double>("rnnlm_weight");
        auto maybe_bos_index = model->get_as<std::string>("bos_index");
        auto maybe_eos_index = model->get_as<std::string>("eos_index");

        // TODO: Throw error in case of invalid toml
        spec.path = *maybe_path;
        spec.name = *maybe_name;
        spec.language_code = *maybe_language_code;

        if (maybe_n_decoders) spec.n_decoders = *maybe_n_decoders;
        if (maybe_beam) spec.beam = *maybe_beam;
        if (maybe_min_active) spec.min_active = *maybe_min_active;
        if (maybe_max_active) spec.max_active = *maybe_max_active;
        if (maybe_lattice_beam) spec.lattice_beam = *maybe_lattice_beam;
        if (maybe_acoustic_scale) spec.acoustic_scale = *maybe_acoustic_scale;
        if (maybe_frame_subsampling_factor) spec.frame_subsampling_factor = *maybe_frame_subsampling_factor;
        if (maybe_silence_weight) spec.silence_weight = *maybe_silence_weight;
        if (maybe_max_ngram_order) spec.max_ngram_order = *maybe_max_ngram_order;
        if (maybe_rnnlm_weight) spec.rnnlm_weight = *maybe_rnnlm_weight;
        if (maybe_bos_index) spec.bos_index = *maybe_bos_index;
        if (maybe_eos_index) spec.eos_index = *maybe_eos_index;

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

} // namespace kaldiserve