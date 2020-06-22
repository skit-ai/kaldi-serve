#include "kaldiserve_pybind/kaldiserve_pybind.h"


namespace kaldiserve {

PYBIND11_MODULE(kaldiserve_pybind, m) {
    m.doc() = "Python binding of kaldiserve";

    // types bindings
    pybind_types(m);
    // model bindings
    pybind_model(m);
    // decoder bindings
    pybind_decoder(m);
    // utils bindings
    pybind_utils(m);
}

} // namespace kaldiserve