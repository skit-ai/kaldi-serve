// kaldiserve_pybind includes
#include "kaldiserve_pybind/kaldiserve_pybind.h"

// kaldiserve includes
#include "kaldiserve/model.hpp"
#include "kaldiserve/types.hpp"


namespace kaldiserve {

void pybind_model(py::module &m) {
    // kaldiserve.ChainModel
    py::class_<ChainModel>(m, "ChainModel", "Chain model class.")
        .def(py::init<const ModelSpec &>());
}

} // namespace kaldiserve