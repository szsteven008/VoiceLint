cmake_minimum_required(VERSION 3.10)

project(VoiceLint)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(
  third_party/kaldi-native-fbank
  third_party/yaml-cpp/include
  third_party/cpp-httplib
  third_party/openai.cpp/include
  third_party/imgui
  third_party/imgui/backends
  third_party/stb
)

add_subdirectory(third_party)
add_subdirectory(src)