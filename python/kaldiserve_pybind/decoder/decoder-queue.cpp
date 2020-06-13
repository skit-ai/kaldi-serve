#include "kaldiserve_pybind/kaldiserve_pybind.h"

#include "kaldiserve/decoder.hpp"


namespace kaldiserve {

void pybind_decoderqueue(py::module &m) {
    // kaldiserve.DecoderQueue
    py::class_<DecoderQueue>(m, "DecoderQueue", "Decoder Queue class.")
        .def(py::init<const ModelSpec &>())
        .def("acquire", &DecoderQueue::acquire, py::call_guard<py::gil_scoped_release>())
        .def("release", &DecoderQueue::release, py::call_guard<py::gil_scoped_release>());
}

} // namespace kaldiserve