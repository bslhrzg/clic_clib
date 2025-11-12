cmake_minimum_required(VERSION 3.18)
project(clic_clib_backend LANGUAGES CXX)

# --- Options ---
option(BUILD_WITH_OPENMP "Enable OpenMP parallelism" ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Find Dependencies ---
find_package(pybind11 CONFIG REQUIRED)
if(BUILD_WITH_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

# --- 1. Define the pure C++ core library (no Python bindings) ---
# This is good practice. It separates your C++ logic from the binding code.
add_library(clic_core STATIC
#  cpp_src/ci_core.cpp
  cpp_src/determinants.cpp 
  cpp_src/wavefunction.cpp
  cpp_src/nbody.cpp
  cpp_src/slater_condon.cpp
  cpp_src/hamiltonian.cpp
  cpp_src/ed_tools.cpp
  cpp_src/applyH.cpp
)
# Tell this target where to find its own headers
target_include_directories(clic_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp_src
)

# --- 2. Define the pybind11 extension module ---
# It only contains the binding code itself.
pybind11_add_module(clic_clib bindings/bindings.cpp)

# --- 3. Link the C++ core library to the Python module ---
target_link_libraries(clic_clib PRIVATE clic_core)

# --- 4. Tell the Python module where to find headers ---
# It needs to see the headers from cpp_src.
target_include_directories(clic_clib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp_src
)

# --- 5. Set the CORRECT output directory for the compiled library ---
# The .so file must go inside the Python package directory we just created.
set_target_properties(clic_clib PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/clic_clib"
)

# --- Link external libraries and set compile flags ---
if(BUILD_WITH_OPENMP)
  target_link_libraries(clic_clib PRIVATE OpenMP::OpenMP_CXX)
endif()

if (MSVC)
  target_compile_options(clic_clib PRIVATE /O2 /DNOMINMAX)
else()
  target_compile_options(clic_clib PRIVATE -O3 -march=native -mtune=native)
endif()

message(STATUS "BUILD_WITH_OPENMP=${BUILD_WITH_OPENMP}")
