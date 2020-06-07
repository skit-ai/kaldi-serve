cmake_minimum_required(VERSION 3.12.0)
# CMake 3.12+ is required to allow linking with OBJECT libraries
# and to prevent erroneous -gencode option deduplication with CUDA.

# Project settings
project(kaldiserve VERSION 1.0.0 LANGUAGES C CXX)

# Config options
option(BUILD_SHARED_LIB          "Build shared library"                     ON)
option(ENABLE_GPU                "Enable GPU build"                         OFF) # TODO(1)
option(BUILD_PYTHON_MODULE       "Build the python module"                  OFF) # TODO(2)

# CXX compiler options
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_SO_NO_EXE OFF)

# Build shared library
if(BUILD_SHARED_LIB)
    set(KALDI_ROOT /opt/kaldi CACHE STRING "Path to Kaldi root directory")
    add_definitions(-DKALDI_DOUBLEPRECISION=0)

    find_package(Boost REQUIRED)
    include_directories(${Boost_INCLUDE_DIRS})

    add_subdirectory(src)

    # install(TARGETS kaldiserve DESTINATION /usr/local/lib)
    # install(DIRECTORY include/kaldiserve DESTINATION /usr/include)
endif()

# Build python port
if (BUILD_PYTHON_MODULE)
    find_package(pybind11 REQUIRED)

    add_subdirectory(python)
endif()
