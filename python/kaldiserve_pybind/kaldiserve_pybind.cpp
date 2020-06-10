#include "kaldiserve_pybind/kaldiserve_pybind.h"


namespace kaldiserve {

PYBIND11_MODULE(kaldiserve_pybind, m) {
    m.doc() = "Python binding of kaldiserve";

    // types bindings
    pybind_types(m);

    // model bindings
    pybind_modelchain(m);

    // decoder bindings
    pybind_decoder(m);
    // pybind_decoderbatch(m);
    pybind_decoderfactory(m);
    pybind_decoderqueue(m);

    // utils bindings
    pybind_utilsio(m);
}

} // namespace kaldiserve