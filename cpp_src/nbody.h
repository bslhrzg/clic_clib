#pragma once
// nbody.h
// -----------------------------------------------------------------------------
// Functions for applying one- and two-body operators (e.g., a Hamiltonian)
// to wavefunction, and for generating connectivity graphs.
// -----------------------------------------------------------------------------
#include "wavefunction.h"
#include <vector>
#include <complex>

namespace ci {

using Coeff = std::complex<double>;

struct OneBodyTerm {
    size_t i, j;
    Spin spin_i, spin_j;
    Coeff value;
};

struct TwoBodyTerm {
    size_t i, j, k, l;
    Spin spin_i, spin_j, spin_k, spin_l;
    Coeff value;
};

Wavefunction apply_one_body_operator(const Wavefunction& wf,
                                     const std::vector<OneBodyTerm>& terms);

Wavefunction apply_two_body_operator(const Wavefunction& wf,
                                     const std::vector<TwoBodyTerm>& terms);

std::vector<SlaterDeterminant>
get_connections_one_body(const std::vector<SlaterDeterminant>& basis,
                         const std::vector<OneBodyTerm>& terms);

std::vector<SlaterDeterminant>
get_connections_two_body(const std::vector<SlaterDeterminant>& basis,
                         const std::vector<TwoBodyTerm>& terms);

} // namespace ci