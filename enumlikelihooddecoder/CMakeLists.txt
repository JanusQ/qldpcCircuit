cmake_minimum_required(VERSION 3.12)
project(enumlikelydecoder)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)


# Add the library
add_library(enumlikelydecoder SHARED
    qldpc_decoder.cpp
    pybind_wrapper.cpp
)

target_link_libraries(enumlikelydecoder PRIVATE pybind11::module) 