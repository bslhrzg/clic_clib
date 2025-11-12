import numpy as np
from time import time
from scipy.sparse.linalg import eigsh
from pyscf import gto, scf, ao2mo

import clic_clib as cc

############ GET INTEGRALS AND STUFF FROM PYSCF ############
# --- build molecule and SCF ----
mol = gto.Mole(basis="6-31g").fromfile("./h2o.xyz")
mf = scf.RHF(mol).run()

# --- number of spatial orbitals
M = mf.mo_coeff.shape[1]
print(f"M = {M} spatial orbitals")
# --- one-electron integrals in MO (spatial) basis
h0_ = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff  # shape (norb, norb)

# --- two-electron integrals in MO basis, chemists' notation (pq|rs)
# ao2mo.kernel returns a packed vector; restore(1) expands to full 4-index tensor
eri_packed = ao2mo.kernel(mol, mf.mo_coeff)         # packed (pq|rs)
U_ = ao2mo.restore(1, eri_packed, M)                # shape (norb,norb,norb,norb)

# --- nuclear repulsion (useful to store alongside)
Enuc = float(mol.energy_nuc())

############ TRANSFORM TO CLIC_CLIB FORM ############
# We need U in a different order
U_ = np.ascontiguousarray(U_.transpose(0,2,1,3))

# We need to double h0 and U on the whole spinorbitals indexes
h0 = cc.double_h(h0_,M)
U = cc.umo2so(U_,M)

h0 = np.ascontiguousarray(h0, dtype=np.complex128)
U = np.ascontiguousarray(U, dtype=np.complex128)
#####################################################


############ MAIN ####################################


# --- 2. Define HF and run calculations at each CI level ---
Ne = 10

# SlaterDeterminant(spatial_orbital_number, list_of_alpha_occupied, list_of_beta_occupied)
# Ici --> (13, [0,1,2,3,4], [0,1,2,3,4])
hf_det = cc.SlaterDeterminant(M, list(range(Ne//2)), list(range(Ne//2)))


# --- Iteration Loop ---
current_basis = sorted([hf_det])

# Wavefunction(spatial_orbital_number, list of SlaterDeterminants, vector of amplitudes)
# Ici, uniquement le HF state
psi = cc.Wavefunction(M,current_basis,[1.0])


toltables = 1e-3
# les tables définissent des sauts autorisés par le hamiltonien 
# h0[j,i] > toltables ajoute le saut i -> j 
# U[j,p,p,i] > toltable ajoute le saut i -> j, si p est occupé
# U[k,l,i,j] > toltable ajoute le saut i,j -> k,l, 
tables = cc.build_screened_hamiltonian(h0,U,toltables)

def expand_basis(psi,tables,tol):

    # psi.get_basis() return the basis as a list of SlaterDeterminants
    current_basis = psi.get_basis()
    # We copy psi to psi_to_expand
    psi_to_expand = cc.Wavefunction(psi)  # This calls the copy constructor
    # coeff c with |c| < tol are pruned with their basis element
    psi_to_expand.prune(tol)
    #######
    # This uses tables to get the new basis
    connected_basis = cc.get_connected_basis(psi_to_expand,tables)
    ######
    
    new_basis_set = set(current_basis) | set(connected_basis)
    new_basis_set = sorted(list(new_basis_set))    
    return new_basis_set



tolprune = 1e-3
maxiter=5
for iter in range(maxiter):
    
    print(f"\n--- Iter {iter} ---")
    
    if iter != 0:
        print(f"Expanding basis from {len(current_basis)} determinants...")
        t_start = time()
        current_basis = expand_basis(psi,tables,tolprune)
        t_end = time()
        print(f"  New basis size = {len(current_basis)} (generated in {t_end - t_start:.2f}s)")

    print(f"Building Hamiltonian ({len(current_basis)}x{len(current_basis)})...")
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
    
    print(f"Total Energy = {np.real(total_gs_energy):.8f}")



