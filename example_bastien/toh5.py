import numpy as np
from time import time
from scipy.sparse.linalg import eigsh
from pyscf import gto, scf, ao2mo

import clic_clib as cc


# --- build molecule and SCF ----
mol = gto.Mole(basis="6-31g").fromfile("./h2o.xyz")
mf = scf.RHF(mol).run()

# --- number of spatial orbitals
M = mf.mo_coeff.shape[1]

# --- one-electron integrals in MO (spatial) basis
h0_ = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff              # shape (norb, norb)

# --- two-electron integrals in MO basis, chemists' notation (pq|rs)
# ao2mo.kernel returns a packed vector; restore(1) expands to full 4-index tensor
eri_packed = ao2mo.kernel(mol, mf.mo_coeff)                    # packed (pq|rs)
U_ = ao2mo.restore(1, eri_packed, M)                         # shape (norb,norb,norb,norb)

# We need U in a different order
U_ = np.ascontiguousarray(U_.transpose(0,2,1,3))

# We need to double h0 and U on the whole spinorbitals indexes
h0 = cc.double_h(h0_,M)
U = cc.umo2so(U_,M)



# --- nuclear repulsion (useful to store alongside)
Enuc = float(mol.energy_nuc())



# --- 2. Define HF and run calculations at each CI level ---
Ne = 10
hf_det = cc.SlaterDeterminant(M, list(range(Ne//2)), list(range(Ne//2)))


# --- Iteration Loop ---
current_basis = sorted([hf_det])

psi = cc.Wavefunction(M,current_basis,[1.0])
toltables = 1e-12
tables = cc.build_screened_hamiltonian(h0,U,toltables)

for level in ["HF", "CISD", "CISDTQ"]:
    print(f"\n--- Calculating {level} Energy ---")
    
    if level != "HF":
        print(f"Expanding basis from {len(current_basis)} determinants...")
        t_start = time()
        #connected_by_H1 = get_connections_one_body(current_basis, one_body_terms)
        #connected_by_H2 = get_connections_two_body(current_basis, two_body_terms)
        
        #new_basis_set = set(current_basis) | set(connected_by_H1) | set(connected_by_H2)
        #current_basis = sorted(list(new_basis_set))


        connected_basis = cc.get_connected_basis(psi,tables)
        new_basis_set = set(current_basis) | set(connected_basis)
        current_basis = sorted(list(new_basis_set))

        t_end = time()
        print(f"  New basis size = {len(current_basis)} (generated in {t_end - t_start:.2f}s)")

    print(f"Building {level} Hamiltonian ({len(current_basis)}x{len(current_basis)})...")
    t_start = time()
    H_sparse = cc.build_hamiltonian_openmp(current_basis, h0, U)
    t_end = time()
    print(f"  Hamiltonian built in {t_end - t_start:.2f}s")
    
    print("Diagonalizing...")
    t_start = time()
    # CORRECTED EIGENSOLVER HANDLING
    if len(current_basis) == 1:
        electronic_gs_energy = H_sparse[0, 0]
    else:
        eigvals, eigvecs = eigsh(H_sparse, k=1, which='SA')
        electronic_gs_energy = eigvals[0]
        psi = cc.Wavefunction(M,current_basis,eigvecs[:,0])
        psi.prune(1e-6)

    total_gs_energy = electronic_gs_energy + Enuc
    t_end = time()
    print(f"  Diagonalized in {t_end - t_start:.2f}s")
    
    print(f"  {level} Total Energy = {np.real(total_gs_energy):.8f}")



