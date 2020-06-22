include_directories(${KALDI_ROOT}/src ${KALDI_ROOT}/tools/openfst/include)
include_directories(${CUDA_TK_ROOT}/include)
include_directories(../include/kaldiserve)

file(GLOB_RECURSE ALL_SOURCE_FILES "*.cpp")

add_library(kaldiserve SHARED ${ALL_SOURCE_FILES})

target_link_options(kaldiserve PUBLIC "-L/usr/local/lib -Wl,--no-as-needed -Wl,--as-needed -ldl '-Wl,-rpath,$$ORIGIN/../lib' -rdynamic")
target_link_directories(kaldiserve
    PUBLIC "${KALDI_ROOT}/src/lib"
    PUBLIC "${KALDI_ROOT}/tools/openfst/lib"
)
target_link_libraries(kaldiserve 
    # openfst
    fst
    # kaldi
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
    kaldi-chain
    kaldi-online2
    kaldi-ivector
    kaldi-lm
    kaldi-rnnlm
    # boost
    boost_filesystem
    -static-libstdc++
)

set_target_properties(kaldiserve PROPERTIES LINKER_LANGUAGE CXX)