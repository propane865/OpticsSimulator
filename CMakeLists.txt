cmake_minimum_required(VERSION 3.24)
project(optics_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# optics_core is an OBJECT library linked into the _native shared module; its
# objects must be PIC-compiled on ELF platforms or the Linux link fails with
# "relocation R_X86_64_PC32 ... can not be used when making a shared object".
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Core numerical kernels (no Python): compiled for the extension and the
# native test target. ``-ffp-contract=off`` keeps evaluation order faithful to
# the NumPy reference paths (no FMA contraction).
add_library(optics_core OBJECT
  native/expression/parser.cpp
  native/expression/evaluate.cpp
  native/geometry/surface.cpp
  native/intersection/plane.cpp
  native/intersection/mesh.cpp
  native/intersection/newton.cpp
  native/intersection/intersect.cpp
  native/intersection/invert.cpp
  native/tracing/direction.cpp
  native/tracing/element.cpp
  native/tracing/stack.cpp
  native/measurement/measurement.cpp
  native/measurement/reconstruct.cpp
  native/optimizing/fit.cpp
  native/optimizing/optimizer.cpp)
target_include_directories(optics_core PUBLIC native/include)
target_compile_options(optics_core PUBLIC
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-ffp-contract=off>)

option(OPTICS_BUILD_TESTS "Build native unit tests" OFF)
if(OPTICS_BUILD_TESTS)
  enable_testing()
  add_executable(optics_native_tests
 native/tests/test_main.cpp
 native/tests/test_expression.cpp
 native/tests/test_kernels.cpp
 native/tests/test_surface.cpp
 native/tests/test_element.cpp
    native/tests/test_stack.cpp
    native/tests/test_measurement.cpp
    native/tests/test_optimizer.cpp)
  target_link_libraries(optics_native_tests PRIVATE optics_core)
  add_test(NAME native_tests COMMAND optics_native_tests)
endif()

option(OPTICS_BUILD_MODULE "Build the Python extension module" ON)
if(NOT OPTICS_BUILD_MODULE)
  return()
endif()

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python_EXECUTABLE}" "-m" "pybind11" "--cmakedir"
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE PYBIND11_RC)
if(PYBIND11_RC)
  message(FATAL_ERROR "The pybind11 Python package is required (pip install pybind11)")
endif()
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" "-c" "import numpy; print(numpy.get_include(), end='')"
  OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
  RESULT_VARIABLE NUMPY_RC)
if(NUMPY_RC)
  message(FATAL_ERROR "The numpy Python package is required (pip install numpy)")
endif()

pybind11_add_module(_native
  native/bindings/module.cpp
  native/bindings/builders.cpp
  native/bindings/kernels.cpp
  native/bindings/tracing.cpp
  native/bindings/measurement.cpp
  native/bindings/optimizer.cpp)
target_link_libraries(_native PRIVATE optics_core)
target_include_directories(_native SYSTEM PRIVATE "${NUMPY_INCLUDE_DIR}")
# Pin the module output so multi-config generators (Visual Studio) do not
# scatter it into per-config subdirectories: editable installers resolve the
# wheel-relative ``optics/_native.*.pyd`` under the build directory.
if(DEFINED OPTICS_PYD_OUT)
  set(_native_out "${OPTICS_PYD_OUT}")
else()
  set(_native_out "${CMAKE_BINARY_DIR}/optics")
endif()
set_target_properties(_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${_native_out}"
  RUNTIME_OUTPUT_DIRECTORY "${_native_out}")
foreach(cfg DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set_target_properties(_native PROPERTIES
    "LIBRARY_OUTPUT_DIRECTORY_${cfg}" "${_native_out}"
    "RUNTIME_OUTPUT_DIRECTORY_${cfg}" "${_native_out}")
endforeach()
if(MINGW)
  # Ship a self-contained module next to a MinGW-built checkout: the
  # posix-thread MinGW runtime pulls in libwinpthread-1.dll unless the
  # whole native link is pinned to static import libraries.
  target_link_options(_native PRIVATE -static -static-libgcc -static-libstdc++)
endif()

# Dev flow (``pythonpath=.`` checkouts): keep a copy of the module next to
# the Python package it backs.
add_custom_command(TARGET _native POST_BUILD
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "$<TARGET_FILE:_native>" "${CMAKE_SOURCE_DIR}/optics/")

install(TARGETS _native DESTINATION optics)

# Pure-Python package data (mirrors the former setuptools package-data).
install(FILES ui/main.ui DESTINATION ui)
install(DIRECTORY optics/examples/ DESTINATION optics/examples FILES_MATCHING PATTERN "*.json")
