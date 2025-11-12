// cpp_src/applyH.h
#pragma once

//#include "ci_core.h"
#include "determinants.h"
#include "wavefunction.h"
#include "slater_condon.h"
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace ci {

// Corresponds to Python's Sh0: map from annihilated orbital `r` to created orbitals `p`.
using TableSh0 = std::unordered_map<uint32_t, std::vector<uint32_t>>;

// Corresponds to Python's SU: map `r` -> `p` -> {spectators `s`}.
using TableSU = std::unordered_map<uint32_t, std::unordered_map<uint32_t, std::vector<uint32_t>>>;

// Corresponds to Python's D: map {`r`,`s`} -> {{`p`,`q`}}.
// Key: r | (s << 32) where r < s.
// Value: p | (q << 32).
using TableD = std::unordered_map<uint64_t, std::vector<std::pair<uint32_t, uint32_t>>>;

struct ScreenedHamiltonian {
    size_t n_spin_orbitals = 0;
    TableSh0 sh0;
    TableSU su;
    TableD d;
};

// Build the screening tables from integral views.
ScreenedHamiltonian build_screened_hamiltonian(
    size_t K, const H1View& H, const ERIView& V, double tol
);

// Apply the screened Hamiltonian to a wavefunction
Wavefunction apply_hamiltonian(
    const Wavefunction& psi,
    const ScreenedHamiltonian& screened_H,
    const H1View& H,
    const ERIView& V,
    double tol_element
);

// Same logic to get the connected basis only
std::vector<SlaterDeterminant> get_connected_basis(
    const Wavefunction& psi,
    const ScreenedHamiltonian& sh
);

// -------- Fixed basis variant --------

// Filter an existing full screened Hamiltonian to a fixed determinant basis.
// The filter keeps only indices that belong to the union of occupied spatial orbitals
// present anywhere in `basis` (lifted to both spins).
// `M` is the number of spatial orbitals for the determinants in `basis`.
ScreenedHamiltonian build_fixed_basis_tables(
    const ScreenedHamiltonian& sh_full,
    const std::vector<SlaterDeterminant>& basis,
    size_t M
);

// Apply H but restrict outputs to the provided fixed determinant set.
// Only determinants in `basis` are produced, and only moves allowed by
// `sh_fixed_basis` are considered.
Wavefunction apply_hamiltonian_fixed_basis(
    const Wavefunction& psi,
    const ScreenedHamiltonian& sh_fixed_basis,
    const std::vector<SlaterDeterminant>& basis,
    const H1View& H,
    const ERIView& V,
    double tol_element
);


} // namespace ci