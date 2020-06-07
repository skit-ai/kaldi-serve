include_directories(${KALDI_ROOT}/src ${KALDI_ROOT}/tools/openfst/include)
include_directories(../../include/kaldiserve)

file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")

add_library(kaldiserve-decoder OBJECT ${ALL_SOURCE_FILES})