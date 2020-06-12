cmake_minimum_required(VERSION 3.12.0)
# CMake 3.12+ is required to allow linking with OBJECT libraries
# and to prevent erroneous -gencode option deduplication with CUDA.

# Project settings
project(kaldiserve VERSION 1.0.0 LANGUAGES C CXX)

# Config options
option(BUILD_SHARED_LIB          "Build shared library"                     ON)
option(ENABLE_GPU                "Enable GPU build"                         OFF) # TODO(1)
option(BUILD_PYBIND11            "Build the pybind11 module"                ON)
option(BUILD_PYTHON_MODULE       "Build the python module"                  ON)

# CXX compiler options
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# kaldi config
set(KALDI_ROOT /opt/kaldi CACHE STRING "Path to Kaldi root directory")
add_definitions(-DKALDI_DOUBLEPRECISION=0)

if(ENABLE_GPU)
    add_definitions(-DHAVE_CUDA)
endif()

# Build shared library
if(BUILD_SHARED_LIB)
    find_package(Boost REQUIRED)
    include_directories(${Boost_INCLUDE_DIRS})

    add_subdirectory(src)
endif()

# Build python port
if (BUILD_PYTHON_MODULE)

    # Pybind11
    if (BUILD_PYBIND11)
        set(BUILD_PYBIND11 ON)
        add_subdirectory(3rdparty/pybind11)
    else()
        find_package(pybind11 REQUIRED)
    endif()

    add_subdirectory(python)
endif()
