// stl includes
#include <iostream>
#include <string>
#include <sstream>
#include <istream>
#include <vector>

// pybind includes
#include <pybind11/stl.h>

// kaldiserve_pybind includes
#include "kaldiserve_pybind/kaldiserve_pybind.h"

// kaldiserve includes
#include "kaldiserve/model.hpp"
#include "kaldiserve/decoder.hpp"
#include "kaldiserve/types.hpp"


namespace kaldiserve {

void pybind_decoder(py::module &m) {
    // kaldiserve.Decoder
    py::class_<Decoder>(m, "Decoder", "Decoder class.")
        .def(py::init<ChainModel *const>())
        .def("start_decoding", &Decoder::start_decoding)
        .def("free_decoder", &Decoder::free_decoder)
        // wav stream chunk
        .def("decode_stream_wav_chunk", [](Decoder &self, py::bytes &wav_bytes) {
            std::string wav_bytes_str(wav_bytes);
            {
                py::gil_scoped_release release;
                std::istringstream wav_stream(wav_bytes_str);
                self.decode_stream_wav_chunk(wav_stream);
            }
            
        })
        // raw wav stream chunk
        .def("decode_stream_raw_wav_chunk", [](Decoder &self, py::bytes &wav_bytes,
                                               const float &samp_freq, const int &data_bytes) {
            std::string wav_bytes_str(wav_bytes);
            {
                py::gil_scoped_release release;
                std::istringstream wav_stream(wav_bytes_str);
                self.decode_stream_raw_wav_chunk(wav_stream, samp_freq, data_bytes);
            }
        })
        // wav audio
        .def("decode_wav_audio", [](Decoder &self, py::bytes &wav_bytes, const float &chunk_size) {
            std::string wav_bytes_str(wav_bytes);
            {
                py::gil_scoped_release release;
                std::istringstream wav_stream(wav_bytes_str);
                self.decode_wav_audio(wav_stream, chunk_size);
            }
        }, py::arg("wav_bytes"), py::arg("chunk_size") = 1.0)
        // raw wav audio
        .def("decode_raw_wav_audio", [](Decoder &self, py::bytes &wav_bytes, const float &samp_freq,
                                        const int &data_bytes, const float &chunk_size) {
            std::string wav_bytes_str(wav_bytes);
            {
                py::gil_scoped_release release;
                std::istringstream wav_stream(wav_bytes_str);
                self.decode_raw_wav_audio(wav_stream, samp_freq, data_bytes, chunk_size);
            }
        }, py::arg("wav_bytes"), py::arg("samp_freq"),
           py::arg("data_bytes"), py::arg("chunk_size") = 1.0)
        // get decoding results -> list[Alternative]
        .def("get_decoded_results", [](Decoder &self, const int &n_best,
                                       const bool &word_level, const bool &bidi_streaming) {
            std::vector<Alternative> alts;
            {
                py::gil_scoped_release release;
                self.get_decoded_results(n_best, alts, word_level, bidi_streaming);
            }
            py::list py_alts = py::cast(alts);
            return py_alts;
        }, py::arg("n_best"),
           py::arg("word_level") = false,
           py::arg("bidi_streaming") = false);

    // kaldiserve.DecoderFactory
    py::class_<DecoderFactory>(m, "DecoderFactory", "Decoder Factory class.")
        .def(py::init<const ModelSpec &>())
        .def("produce", &DecoderFactory::produce, py::call_guard<py::gil_scoped_release>(), py::return_value_policy::reference);

    // kaldiserve.DecoderQueue
    py::class_<DecoderQueue>(m, "DecoderQueue", "Decoder Queue class.")
        .def(py::init<const ModelSpec &>())
        .def("acquire", &DecoderQueue::acquire, py::call_guard<py::gil_scoped_release>(), py::return_value_policy::reference)
        .def("release", &DecoderQueue::release);//, py::call_guard<py::gil_scoped_release>());
}

} // namespace kaldiserve