include_directories(${KALDI_ROOT}/src ${KALDI_ROOT}/tools/openfst/include)
include_directories(../include/kaldiserve)

add_subdirectory(model)
add_subdirectory(decoder)
add_subdirectory(utils)

add_library(kaldiserve SHARED
    config.cpp
    $<TARGET_OBJECTS:kaldiserve-decoder>
    $<TARGET_OBJECTS:kaldiserve-model>
    $<TARGET_OBJECTS:kaldiserve-utils>
)

target_link_directories(kaldiserve
    PUBLIC "${KALDI_ROOT}/src/lib"
    PUBLIC "${KALDI_ROOT}/tools/openfst/lib"
)
target_link_libraries(kaldiserve 
    kaldi-decoder
    kaldi-lat
    kaldi-fstext
    kaldi-hmm
    kaldi-feat
    kaldi-transform
    kaldi-gmm
    kaldi-tree
    kaldi-util
    kaldi-matrix
    kaldi-cudamatrix
    kaldi-base
    kaldi-nnet3
    kaldi-online2
    kaldi-ivector
    kaldi-lm
    kaldi-rnnlm
    boost_filesystem
)

set_target_properties(kaldiserve PROPERTIES LINKER_LANGUAGE CXX)