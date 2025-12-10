cmake_minimum_required(VERSION 3.18)
project(error_align LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG REQUIRED)

# Build the module
pybind11_add_module(_cpp_edit_distance cpp/edit_distance.cpp)
pybind11_add_module(_cpp_beam_search cpp/beam_search.cpp)

# Install it under the Python package directory (src/error_align)
install(
    TARGETS
        _cpp_edit_distance
        _cpp_beam_search
    LIBRARY DESTINATION error_align
    RUNTIME DESTINATION error_align
)
