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

    fci_basis = get_fci_basis(M, Ne)
    H_sparse = qc.build_hamiltonian_openmp(fci_basis, h0_alphafirst, U_alphafirst)

    e0,_ = eigsh(H_sparse, k=1)
    print(f"e0 = {e0+Enuc}")

    # applyH test 
    size = len(fci_basis)
    #randib = np.random.choice(len(fci_basis),size=size,replace=False)

    fbasis = fci_basis#[fci_basis[i] for i in randib]
    fH = qc.build_hamiltonian_openmp(fbasis, h0_alphafirst, U_alphafirst).toarray()

    famps = np.random.randn(size) 
    famps /= np.linalg.norm(famps)
    print(f"famps[:10] = {famps[:10]}")

    psi = qc.Wavefunction(M, fbasis, famps)

    true_Hpsi = fH @ famps 
    true_Hpsi = true_Hpsi / np.linalg.norm(true_Hpsi)
    

    toltables = 0
    tables = qc.build_hamiltonian_tables(h0_alphafirst,U_alphafirst,toltables)
    tol_el = 0

    ftables = qc.build_fixed_basis_tables(tables,fbasis,M)
    test_Hpsi = qc.apply_hamiltonian(psi, tables, h0_alphafirst,U_alphafirst,tol_el)

    test_Hpsi.normalize()
    test_Hpsi = test_Hpsi.get_amplitudes() 

    print(f"ApplyH TEST : <true_Hpsi | test_Hpsi> = {np.dot(test_Hpsi,true_Hpsi)}")

    for (i,e) in enumerate(fbasis[:10]):
        print(f"state {e}: true : {true_Hpsi[i]}, test: {test_Hpsi[i]}")


    # applyH in fixed basis test
    size = 100
    randib = np.random.choice(len(fci_basis),size=size,replace=False)
    #randib = np.sort(randib)

    fbasis = [fci_basis[i] for i in randib]
    fH = qc.build_hamiltonian_openmp(fbasis, h0_alphafirst, U_alphafirst).toarray()

    H_fb = qc.build_hamiltonian_matrix_fixed_basis(
        ftables, fbasis, h0_alphafirst, U_alphafirst, 0.0
    )
    H_ref = qc.build_hamiltonian_openmp(
        fbasis, h0_alphafirst, U_alphafirst
    ).toarray()

    print("max |ΔH| =", np.max(np.abs(H_fb.toarray() - H_ref)))

    assert 1==0
    famps = np.random.randn(size) 
    famps /= np.linalg.norm(famps)
    print(f"famps[:10] = {famps[:10]}")

    psi = qc.Wavefunction(M, fbasis, famps)

    true_Hpsi = fH @ famps 
    true_Hpsi = true_Hpsi / np.linalg.norm(true_Hpsi)

    true_Hpsi = qc.Wavefunction(M,fbasis, true_Hpsi)
    

    toltables = 0
    tables = qc.build_hamiltonian_tables(h0_alphafirst,U_alphafirst,toltables)
    tol_el = 0

    ftables = qc.build_fixed_basis_tables(tables,fbasis,M)
    test_Hpsi = qc.apply_hamiltonian_fixed_basis(psi,ftables,fbasis, h0_alphafirst,U_alphafirst,tol_el)
    
    test_Hpsi.normalize()
    #test_Hpsi = test_Hpsi.get_amplitudes() 

    print(f"Fixed Basis applyH TEST : <true_Hpsi | test_Hpsi> = {test_Hpsi.dot(true_Hpsi)}")

    #for (i,e) in enumerate(fbasis[:10]):
    #    print(f"state {e}: true : {true_Hpsi[i]}, test: {test_Hpsi[i]}")

   

if __name__ == "__main__":
    # You can keep the old test function if you want, but this is the main one
    test_h2o_fci_energy()

