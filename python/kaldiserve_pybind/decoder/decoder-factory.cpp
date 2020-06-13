#include "kaldiserve_pybind/kaldiserve_pybind.h"

#include "kaldiserve/decoder.hpp"


namespace kaldiserve {

void pybind_decoderfactory(py::module &m) {
    // kaldiserve.DecoderFactory
    py::class_<DecoderFactory>(m, "DecoderFactory", "Decoder Factory class.")
        .def(py::init<const ModelSpec &>())
        .def("produce", &DecoderFactory::produce, py::call_guard<py::gil_scoped_release>());
}

} // namespace kaldiserve