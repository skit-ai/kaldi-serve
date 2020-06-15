find_package(Python REQUIRED)

# add pybind module
file(GLOB_RECURSE PY_ALL_SOURCE_FILES "kaldiserve_pybind/*.cpp")
pybind11_add_module(kaldiserve_pybind ${PY_ALL_SOURCE_FILES})

# suppress pybind11 warnings
target_include_directories(kaldiserve_pybind SYSTEM PRIVATE ${PYBIND11_INCLUDE_DIR})

# kaldiserve_pybind, kaldiserve, kaldi, openfst includes
target_include_directories(kaldiserve_pybind PRIVATE .)
target_include_directories(kaldiserve_pybind PRIVATE ../include)
target_include_directories(kaldiserve_pybind PRIVATE ${KALDI_ROOT}/src ${KALDI_ROOT}/tools/openfst/include)

# kaldiserve lib
target_link_libraries(kaldiserve_pybind PRIVATE kaldiserve)