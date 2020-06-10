#include "kaldiserve_pybind/kaldiserve_pybind.h"

#include "kaldiserve/decoder.hpp"


namespace kaldiserve {

void pybind_decoderqueue(py::module &m) {
    // kaldiserve.DecoderQueue
    py::class_<DecoderQueue>(m, "DecoderQueue", "Decoder Queue class.")
        .def(py::init<const ModelSpec &>())
        .def("acquire", &DecoderQueue::acquire)
        .def("release", &DecoderQueue::release);
}

} // namespace kaldiserve