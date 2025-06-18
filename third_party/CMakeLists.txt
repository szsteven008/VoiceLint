include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/kaldi-native-fbank
  ${CMAKE_CURRENT_SOURCE_DIR}/yaml-cpp/include
)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)

set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "Build Python extension" FORCE)
set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
add_subdirectory(kaldi-native-fbank)

set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "Build contrib" FORCE)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "Build tools" FORCE)
add_subdirectory(yaml-cpp)
