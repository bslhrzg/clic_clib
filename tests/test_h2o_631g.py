import numpy as np
import h5py
from itertools import combinations
from scipy.sparse.linalg import eigsh
from clic_clib import *
from clic import *
import time
from test_utils import *



# --- Main Test Function ---

def test_h2o_iterative_ci():
    print("--- Testing Iterative CI for H2O/6-31G ---")
    
    # --- 1. Load and Transform Integrals ---
    h_core, ee_mo, Enuc, M = load_spatial_integrals("my_data.hdf5")
    
    print("\nConverting spatial integrals to spin-orbital form (AlphaFirst)...")
    h0 = double_h(h_core, M)
    # Since C++ code and Julia code use same formula, and Julia uses physicist notation,
    # we just need to spin-adapt the physicist's notation integrals.
    U_phys = umo2so(ee_mo, M)
    
    h0_clean = np.ascontiguousarray(h0, dtype=np.complex128)
    U_clean = np.ascontiguousarray(U_phys, dtype=np.complex128)


    
    # We still need operator terms for the dynamic part.
    # U_phys is <ij|V|kl>. The C++ expects V[i,j,k,l] to be passed to get_connections
    # for the operator c_i† c_j† c_l c_k. This matches.
    one_body_terms = get_one_body_terms(h0_clean, M)
    two_body_terms = get_two_body_terms(U_clean, M)
    
    # --- 2. Define HF and run calculations at each CI level ---
    Ne = 10
    hf_det = SlaterDeterminant(M, list(range(Ne//2)), list(range(Ne//2)))
    save_integrals_to_h5("h2o_631g_alphafirst.h5", h0_clean, U_clean, Ne, Enuc)

    ref_energies = {
        "HF": -75.98394138177851,
        "CISD": -76.11534669560683,
        "CISDT": -76.122,
        "CISDTQ": -76.12236,
    }

    # --- Iteration Loop ---
    current_basis = sorted([hf_det])

    psi = Wavefunction(M,current_basis,[1.0])
    toltables = 1e-12
    tables = build_hamiltonian_tables(h0_clean,U_clean,toltables)
    
    for level in ["HF", "CISD", "CISDTQ"]:
        print(f"\n--- Calculating {level} Energy ---")
        
        if level != "HF":
            print(f"Expanding basis from {len(current_basis)} determinants...")
            t_start = time.time()
            #connected_by_H1 = get_connections_one_body(current_basis, one_body_terms)
            #connected_by_H2 = get_connections_two_body(current_basis, two_body_terms)
            
            #new_basis_set = set(current_basis) | set(connected_by_H1) | set(connected_by_H2)
            #current_basis = sorted(list(new_basis_set))


            connected_basis = get_connected_basis(psi,tables)
            new_basis_set = set(current_basis) | set(connected_basis)
            current_basis = sorted(list(new_basis_set))

            t_end = time.time()
            print(f"  New basis size = {len(current_basis)} (generated in {t_end - t_start:.2f}s)")

        print(f"Building {level} Hamiltonian ({len(current_basis)}x{len(current_basis)})...")
        t_start = time.time()
        H_sparse = build_hamiltonian_openmp(current_basis, h0_clean, U_clean)
        t_end = time.time()
        print(f"  Hamiltonian built in {t_end - t_start:.2f}s")
        
        print("Diagonalizing...")
        t_start = time.time()
        # CORRECTED EIGENSOLVER HANDLING
        if len(current_basis) == 1:
            electronic_gs_energy = H_sparse[0, 0]
        else:
            eigvals, eigvecs = eigsh(H_sparse, k=1, which='SA')
            electronic_gs_energy = eigvals[0]
            psi = Wavefunction(M,current_basis,eigvecs[:,0])
            psi.prune(1e-6)

        total_gs_energy = electronic_gs_energy + Enuc
        t_end = time.time()
        print(f"  Diagonalized in {t_end - t_start:.2f}s")


        
        print(f"  {level} Total Energy = {np.real(total_gs_energy):.8f}")
        print(f"  Reference Energy   = {ref_energies[level]:.8f}")
        np.testing.assert_allclose(total_gs_energy, ref_energies[level], atol=1e-3)
        print(f"  ✅ {level} energy is correct.")

if __name__ == "__main__":
    test_h2o_iterative_ci()
