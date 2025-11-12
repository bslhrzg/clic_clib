import numpy as np
import h5py
from itertools import combinations
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

import clic_clib as qc
from test_utils import *

def calculate_hf_energy_python(h0, U, Ne, Enuc):
    """Calculates HF energy from raw interleaved integrals, proven to be correct."""
    occupied_indices = range(Ne)
    e1 = sum(h0[i, i] for i in occupied_indices)
    e2 = sum(
        (U[i, j, i, j] - U[i, j, j, i])
        for i in occupied_indices
        for j in occupied_indices
    )
    return e1 + 0.5 * e2 + Enuc

def get_rhf_determinant(Nelec, M):
    """
    Returns the RHF/ROHF Slater determinant in a spin-blocked MO basis.
    """
    if Nelec % 2 == 0:
    
        n_occ = Nelec // 2
        occ_hf = list(range(n_occ))

        return [cc.SlaterDeterminant(M, occ_hf, occ_hf)]
    else :
        n_occ = Nelec // 2
        occ_m = list(range(n_occ))
        occ_p = list(range(n_occ+1))

        b1 = cc.SlaterDeterminant(M, occ_m, occ_p)
        b2 = cc.SlaterDeterminant(M, occ_p, occ_m)

        return [b1,b2]



def test_h2o_fci_energy():
    print("--- Testing FCI Ground State Energy of H2O/STO-3G ---")
    
    # 1. Load data
    h0_interleaved, U_interleaved, Ne, Enuc, M, K = load_integrals_from_h5("h2o_h0U.h5")
    
    # 2. Transform integrals to AlphaFirst basis
    print("\nTransforming integrals to AlphaFirst basis...")
    h0_alphafirst_raw, U_alphafirst_raw = qc.transform_integrals_interleaved_to_alphafirst(
        h0_interleaved, U_interleaved, M
    )
    
    # 3. CRITICAL: Ensure arrays are C-contiguous and have the correct dtype
    h0_alphafirst = np.ascontiguousarray(h0_alphafirst_raw, dtype=np.complex128)
    U_alphafirst = np.ascontiguousarray(U_alphafirst_raw, dtype=np.complex128)


    # 4. Start from HF
    print("\nGenerating HF basis...")
    Nelec=10
    basis = get_rhf_determinant(Ne, M)
    psi = qc.Wavefunction(M,basis,[1.0])

    maxiter=600
    toltables = 0
    tables = qc.build_screened_hamiltonian(h0_alphafirst,U_alphafirst,toltables)
    tol_el = 0
    tol_prune = 0
    for _ in range(maxiter) :
        psi.normalize()
        Hpsi = qc.apply_hamiltonian(psi, tables, h0_alphafirst,U_alphafirst,tol_el)
        electronic_gs_energy = Hpsi.dot(psi)
        psi = Hpsi
        psi.prune(tol_prune)
        basis_ = psi.get_basis()
        print(f"iter {_}: E0 = {electronic_gs_energy}, len(basis) = {len(basis_)}")
    
    # 7. Calculate total energy and validate
    total_gs_energy = electronic_gs_energy + Enuc
    print(f"\n  Lowest electronic energy = {electronic_gs_energy:.8f} Hartree")
    print(f"  Total ground state energy (Electronic + Nuclear) = {total_gs_energy:.8f} Hartree")
    
    # Reference value for FCI/STO-3G water with these integrals
    reference_fci_energy = -75.0232909847239
    print(f"  Reference FCI energy = {reference_fci_energy:.8f} Hartree")
    
    np.testing.assert_allclose(total_gs_energy, reference_fci_energy, atol=1e-6)
    print("\n✅✅✅ FINAL SUCCESS: Calculated FCI energy matches the reference value.")

if __name__ == "__main__":
    # You can keep the old test function if you want, but this is the main one
    test_h2o_fci_energy()

