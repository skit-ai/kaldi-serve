// kaldiserve_pybind includes
#include "kaldiserve_pybind/kaldiserve_pybind.h"

// kaldiserve includes
#include "kaldiserve/utils.hpp"


namespace kaldiserve {

void pybind_utilsio(py::module &m) {
    m.def("parse_model_specs", &parse_model_specs, "Parses model specifications.");
}

} // namespace kaldiserve