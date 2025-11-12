#pragma once
// wavefunction.h
// -----------------------------------------------------------------------------
// A class representing a CI wavefunction as a linear combination of
// Slater determinants.
// -----------------------------------------------------------------------------

#include "determinants.h"
#include <vector>
#include <complex>
#include <unordered_map>

namespace ci {

// -----------------------------------------------------------------------------
// Wavefunction over Slater determinants
// -----------------------------------------------------------------------------
class Wavefunction {
public:
    using Coeff = std::complex<double>;
    using Data  = std::unordered_map<SlaterDeterminant, Coeff>;

    explicit Wavefunction(std::size_t n_spatial);

    Wavefunction(std::size_t n_spatial, const Data& init);


    Wavefunction(std::size_t n_spatial,
        const std::vector<SlaterDeterminant>& basis,
        const std::vector<Coeff>& coeffs,
        bool keep_zeros = true);

    std::size_t  num_spatial_orbitals() const noexcept;
    const Data&  data() const noexcept;

    void add_term(const SlaterDeterminant& det, Coeff c, double tol=0);
    void normalize(double tol=1e-15);
    void prune(double threshold);

    Coeff dot(const Wavefunction& other) const;

    void add_wavefunction(const Wavefunction& other, Coeff scale = {1.0, 0.0});


    std::vector<SlaterDeterminant> basis_sorted() const;
    std::vector<Coeff>             coeffs_sorted(const std::vector<SlaterDeterminant>& basis) const;


private:
    std::size_t n_spatial_{0};
    Data        data_;
};

// Wavefunction operator application (on Slater basis)
Wavefunction apply_creation   (const Wavefunction& wf, std::size_t i0, Spin spin,
                               SpinOrbitalOrder order = SpinOrbitalOrder::AlphaFirst);

Wavefunction apply_annihilation(const Wavefunction& wf, std::size_t i0, Spin spin,
                               SpinOrbitalOrder order = SpinOrbitalOrder::AlphaFirst);

} // namespace ci