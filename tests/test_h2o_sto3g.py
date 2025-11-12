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


    # 4. Generate the FCI basis
    print("\nGenerating FCI basis...")
    fci_basis = get_fci_basis(M, Ne)
    print(f"  FCI basis size for ({M}o, {Ne}e) = {len(fci_basis)} determinants.")
    
    # 5. Build the full FCI Hamiltonian matrix
    print("\nBuilding FCI Hamiltonian matrix with C++ kernel...")
    H_sparse = qc.build_hamiltonian_openmp(fci_basis, h0_alphafirst, U_alphafirst)
    print(f"  Hamiltonian construction complete. Matrix shape: {H_sparse.shape}")
    
    # 6. Diagonalize to find the lowest eigenvalue
    print("\nDiagonalizing FCI Hamiltonian...")
    electronic_eigenvalues, _ = eigsh(H_sparse, k=4, which='SA')
    #electronic_eigenvalues, _ = eigh(H_sparse.toarray())
    electronic_gs_energy = electronic_eigenvalues[0]
    
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

