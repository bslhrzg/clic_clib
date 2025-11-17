// bindings/bindings.cpp
#include "bind_common.hpp"

PYBIND11_MODULE(clic_clib, m) {
    m.doc() = R"doc(
        High-performance C++ backend for Configuration Interaction (CI) calculations.

        This library provides core data structures for representing Slater determinants
        and wavefunction, efficient C++ kernels for applying operators and building
        Hamiltonian matrices using Slater-Condon rules, and tools for dynamic
        operator application.
    )doc";

    bind_enums_and_slater(m);
    bind_wavefunction(m);
    bind_operators_and_connections(m);
    bind_screened_hamiltonian(m);
    bind_ed_tools_and_matrices(m);
}