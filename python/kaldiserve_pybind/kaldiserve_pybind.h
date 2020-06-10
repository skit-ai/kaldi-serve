#pragma once

// stl includes
#include <vector>

// pybind includes
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>

// kaldiserve includes
#include "kaldiserve/types.hpp"

namespace py = pybind11;
using namespace py::literals;


PYBIND11_MAKE_OPAQUE(std::vector<kaldiserve::ModelSpec>);
PYBIND11_MAKE_OPAQUE(std::vector<kaldiserve::Word>);
PYBIND11_MAKE_OPAQUE(std::vector<kaldiserve::Alternative>);

namespace kaldiserve {

// types
void pybind_types(py::module &m);

// model
void pybind_modelchain(py::module &m);

// decoder
void pybind_decoder(py::module &m);
void pybind_decoderbatch(py::module &m);
void pybind_decoderfactory(py::module &m);
void pybind_decoderqueue(py::module &m);

// utils
void pybind_utilsio(py::module &m);
}