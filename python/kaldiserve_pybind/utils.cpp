// stl includes
#include <vector>
#include <string>

// pybind includes
#include <pybind11/stl.h>

// kaldiserve_pybind includes
#include "kaldiserve_pybind/kaldiserve_pybind.h"

// kaldiserve includes
#include "kaldiserve/utils.hpp"
#include "kaldiserve/types.hpp"


namespace kaldiserve {

void pybind_utils(py::module &m) {
    m.def("parse_model_specs", [](const std::string &toml_path) {
        std::vector<ModelSpec> model_specs;
        parse_model_specs(toml_path, model_specs);
        py::list py_model_specs = py::cast(model_specs);
        return py_model_specs;
    });
}

} // namespace kaldiserve