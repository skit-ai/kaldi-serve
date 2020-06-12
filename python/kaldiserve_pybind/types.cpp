// stl includes
#include <vector>
#include <string>

// kaldiserve_pybind includes
#include "kaldiserve_pybind/kaldiserve_pybind.h"

// kaldiserve includes
#include "kaldiserve/types.hpp"


namespace kaldiserve {

void pybind_types(py::module &m) {

    py::bind_vector<std::vector<ModelSpec>>(m, "_ModelSpecList");

    // kaldiserve.ModelSpec
    py::class_<ModelSpec>(m, "ModelSpec", "Model Specification struct.")
        .def(py::init<>())
        .def_readonly("name", &ModelSpec::name)
        .def_readonly("language_code", &ModelSpec::language_code)
        .def_readonly("path", &ModelSpec::path)
        .def_readonly("n_decoders", &ModelSpec::n_decoders)
        .def_readonly("min_active", &ModelSpec::min_active)
        .def_readonly("max_active", &ModelSpec::max_active)
        .def_readonly("frame_subsampling_factor", &ModelSpec::frame_subsampling_factor)
        .def_readonly("beam", &ModelSpec::beam)
        .def_readonly("lattice_beam", &ModelSpec::lattice_beam)
        .def_readonly("acoustic_scale", &ModelSpec::acoustic_scale)
        .def_readonly("silence_weight", &ModelSpec::silence_weight)
        .def_readonly("max_ngram_order", &ModelSpec::max_ngram_order)
        .def_readonly("rnnlm_weight", &ModelSpec::rnnlm_weight)
        .def_readonly("bos_index", &ModelSpec::bos_index)
        .def_readonly("eos_index", &ModelSpec::eos_index)
        .def("__repr__", [](const ModelSpec &ms) {
            return "<kaldiserve.ModelSpec {name: '" + ms.name +
                   "', language_code: '" + ms.language_code +
                   "', path: '" + ms.path + "'}>";
        });
        // .def(py::init<const std::string &, const std::string &, const std::string &,
        //               const int &, const int &, const int &, const int &,
        //               const float &, const float &, const float &, const float &,
        //               const int &, const float &, const std::string &, const std::string &>(),
        //               py::arg("name"), py::arg("language_code"), py::arg("path"),
        //               py::arg("n_decoders") = 1, py::arg("min_active") = 200,
        //               py::arg("max_active") = 7000, py::arg("frame_subsampling_factor") = 3,
        //               py::arg("beam") = 16.0, py::arg("lattice_beam") = 6.0, py::arg("acoustic_scale") = 1.0,
        //               py::arg("silence_weight") = 1.0, py::arg("max_ngram_order") = 3,
        //               py::arg("rnnlm_weight") = 0.5, py::arg("bos_index") = "1", py::arg("eos_index") = "2");

    py::bind_vector<std::vector<Word>>(m, "_WordList");

    // kaldiserve.Word
    py::class_<Word>(m, "Word", "Word struct.")
        .def(py::init<>())
        .def_readonly("start_time", &Word::start_time)
        .def_readonly("end_time", &Word::end_time)
        .def_readonly("confidence", &Word::confidence)
        .def_readonly("word", &Word::word)
        .def("__repr__", [](const Word &w) {
            return "<kaldiserve.Word {word: '" + w.word +
                   "', confidence: '" + std::to_string(w.confidence) +
                   "', start_time: '" + std::to_string(w.start_time) +
                   "', end_time: '" + std::to_string(w.end_time) + "'}>";
        });
        // .def(py::init<const float &, const float &, const float &, const std::string &>(),
        //      py::arg("start_time"), py::arg("end_time"), py::arg("confidence"), py::arg("word"))

    py::bind_vector<std::vector<Alternative>>(m, "_AlternativeList");

    // kaldiserve.Alternative
    py::class_<Alternative>(m, "Alternative", "Alternative struct.")
        .def(py::init<>())
        .def_readonly("transcript", &Alternative::transcript)
        .def_readonly("confidence", &Alternative::confidence)
        .def_readonly("am_score", &Alternative::am_score)
        .def_readonly("lm_score", &Alternative::lm_score)
        .def_readonly("words", &Alternative::words)
        .def("__repr__", [](const Alternative &alt) {
            return "<kaldiserve.Alternative {transcript: '" + alt.transcript +
                    "', confidence: '" + std::to_string(alt.confidence) +
                    "', am_score: '" + std::to_string(alt.am_score) +
                    "', lm_score: '" + std::to_string(alt.lm_score) + "'}>";
            //    "', words: '" + std::string(alt.words.begin(), alt.words.end()) + "'}>";
        });
        // .def(py::init<const std::string &, const double &, const float &, const float &, std::vector<Word>>(),
        //      py::arg("transcript"), py::arg("confidence"), py::arg("am_score"), py::arg("lm_score"), py::arg("words"))
}

} // namespace kaldiserve