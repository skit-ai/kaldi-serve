// kaldiserve_pybind includes
#include "kaldiserve_pybind/kaldiserve_pybind.h"

// kaldiserve includes
#include "kaldiserve/model.hpp"
#include "kaldiserve/decoder.hpp"


namespace kaldiserve {

void pybind_decoder(py::module &m) {
    // kaldiserve.Decoder
    py::class_<Decoder>(m, "Decoder", "Decoder class.")
        .def(py::init<ChainModel *const>())
        .def("start_decoding", &Decoder::start_decoding)
        .def("free_decoder", &Decoder::free_decoder)
        .def("decode_stream_wav_chunk", &Decoder::decode_stream_wav_chunk)
        .def("decode_stream_raw_wav_chunk", &Decoder::decode_stream_raw_wav_chunk)
        .def("decode_wav_audio", &Decoder::decode_wav_audio)
        .def("decode_raw_wav_audio", &Decoder::decode_raw_wav_audio)
        .def("get_decoded_results", &Decoder::get_decoded_results);
}

} // namespace kaldiserve