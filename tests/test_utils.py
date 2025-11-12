import numpy as np 
from itertools import combinations
import clic_clib as cc
import h5py
import scipy 

# Some utilitaries that would be needed in general 
# --- Integral Transformation Functions ---
def load_integrals_from_h5(filename):
    with h5py.File(filename, 'r') as f:
        h0 = f['h0'][:]
        U = f['U'][:]
        Ne = int(f.attrs['Ne'])
        Enuc = float(f.attrs['Enuc'])
        K = h0.shape[0]
        M = K // 2
    return h0, U, Ne, Enuc, M, K

def load_spatial_integrals(filename):
    """Loads spatial integrals and metadata from the HDF5 file."""
    with h5py.File(filename, "r") as f:
        hcore = f["h0"][:]
        ee = f["U"][:]
    Enuc = 9.003584105404158

    M = hcore.shape[0]
    print(f"Loaded data from {filename}:")
    print(f"  Spatial Orbitals (M) = {M}, Nuclear Repulsion (Enuc) = {Enuc:.8f}")
    return hcore, ee, Enuc, M

def save_integrals_to_h5(filename, h0, U, Ne, Enuc):
    """Save one- and two-body integrals and metadata to an HDF5 file."""
    with h5py.File(filename, 'w') as f:
        f.create_dataset('h0', data=np.asarray(h0))
        f.create_dataset('U', data=np.asarray(U))
        f.attrs['Ne'] = int(Ne)
        f.attrs['Enuc'] = float(Enuc)
    print(f"Saved integrals to {filename}")


# Just to run tests
def get_fci_basis(num_spatial, num_electrons):
    """
    Return the Full Configuration Interaction (fci) basis 
    for N_electrons among M spatial orbitals, as a list of 
    SlaterDeterminants objects 
    """
    num_spin_orbitals = 2 * num_spatial
    basis_dets = []
    for occupied_indices in combinations(range(num_spin_orbitals), num_electrons):
        occ_a = [i for i in occupied_indices if i < num_spatial]
        occ_b = [i - num_spatial for i in occupied_indices if i >= num_spatial]
        det = cc.SlaterDeterminant(num_spatial, occ_a, occ_b)
        basis_dets.append(det)
    return sorted(basis_dets)

# --- Integral Generation for Anderson Impurity Model ---
def get_impurity_integrals(M, u, e_bath, V_bath, mu):
    """
    Builds the h0 and U integrals for the Anderson Impurity Model.
    """
    K = 2 * M
    h_spatial = np.zeros((M, M))
    diagonal_elements = np.concatenate(([-mu], e_bath))
    np.fill_diagonal(h_spatial, diagonal_elements)
    h_spatial[0, 1:] = V_bath
    h_spatial[1:, 0] = np.conj(V_bath)

    h0 = np.zeros((K, K))
    h0[0:M, 0:M] = h_spatial
    h0[M:K, M:K] = h_spatial
    
    U = np.zeros((K, K, K, K))
    imp_alpha_idx, imp_beta_idx = 0, M
    U[imp_alpha_idx, imp_beta_idx, imp_alpha_idx, imp_beta_idx] = u
    U[imp_beta_idx, imp_alpha_idx, imp_beta_idx, imp_alpha_idx] = u
    return h0, U

# --- Helpers for Operator Terms ---
def get_one_body_terms(h1, M):
    terms = []
    for i in range(2*M):
        for j in range(2*M):
            if abs(h1[i, j]) > 1e-14:
                si = cc.Spin.Alpha if i < M else cc.Spin.Beta
                sj = cc.Spin.Alpha if j < M else cc.Spin.Beta
                oi, oj = (i % M), (j % M)
                terms.append((oi, oj, si, sj, complex(h1[i, j])))
    return terms

def get_two_body_terms(U, M):
    terms = []
    for i,j,k,l in np.argwhere(np.abs(U) > 1e-14):
        spins = [cc.Spin.Alpha if idx < M else cc.Spin.Beta for idx in [i,j,k,l]]
        orbs  = [idx % M for idx in [i,j,k,l]]
        terms.append((orbs[0],orbs[1],orbs[2],orbs[3],
                      spins[0],spins[1],spins[2],spins[3], complex(0.5*U[i,j,k,l])))
    return terms
    
# --- The Matrix-Free Hamiltonian Operator ---
def make_H_on_psi(one_body_terms, two_body_terms, prune_thresh=1e-6):
    """Factory function to create the H|psi> operator."""
    def H_on_psi(psi_in):
        psi_out1 = cc.apply_one_body_operator(psi_in, one_body_terms)
        psi_out2 = cc.apply_two_body_operator(psi_in, two_body_terms)
        psi_res = psi_out1 + psi_out2
        psi_res.prune(prune_thresh)
        return psi_res
    return H_on_psi


def get_scalar_lanczos_wf(H_op, v_init, L):
    """
    Robust scalar Lanczos for a single Wavefunction vector.
    """
    alphas, betas = [], []
    v_list = []

    # Normalize initial vector
    norm_v = np.sqrt(abs(v_init.dot(v_init)))
    if norm_v < 1e-14:
        return np.array(alphas), np.array(betas)
    
    v_list.append((1.0 / norm_v) * v_init)

    for j in range(L):
        q = v_list[j]
        w = H_op(q)
        
        # Re-orthogonalize against previous two vectors (maintains orthogonality)
        if j > 0:
            w = w - betas[j-1] * v_list[j-1]
        
        alpha = q.dot(w)
        alphas.append(alpha)
        
        w = w - alpha * q
        
        beta = np.sqrt(abs(w.dot(w)))
        
        if beta < 1e-12:
            break # Breakdown
        
        betas.append(beta)
        v_list.append((1.0 / beta) * w)

    return np.array(alphas), np.array(betas)

def green_function_scalar_lanczos_wf(H_op, M, psi0_wf, e0, L, ws, eta, impurity_indices):
    """
    Calculates DIAGONAL elements G_ii(ω) using a corrected scalar Lanczos.
    """
    Norb = 2 * M
    Nw = len(ws)
    G_all = np.zeros((Nw, Norb, Norb), dtype=np.complex128)

    for i in impurity_indices:
        # --- Greater Green's function (particle addition) ---
        seed_g = cc.apply_creation(psi0_wf, i % M, cc.Spin.Alpha if i < M else cc.Spin.Beta)
        norm_g_sq = abs(seed_g.dot(seed_g))
        
        if norm_g_sq > 1e-12:
            a_g, b_g = get_scalar_lanczos_wf(H_op, seed_g, L)
            z = (ws + e0) + 1j * eta
            
            # CORRECTED Continued Fraction
            g_g = np.zeros_like(z)
            for k in range(len(a_g) - 1, -1, -1):
                b2 = b_g[k]**2 if k < len(b_g) else 0.0
                g_g = 1.0 / (z - a_g[k] - b2 * g_g)
            
            G_all[:, i, i] += norm_g_sq * g_g

        # --- Lesser Green's function (particle removal) ---
        seed_l = cc.apply_annihilation(psi0_wf, i % M, cc.Spin.Alpha if i < M else cc.Spin.Beta)
        norm_l_sq = abs(seed_l.dot(seed_l))
        
        if norm_l_sq > 1e-12:
            a_l, b_l = get_scalar_lanczos_wf(H_op, seed_l, L)
            z = (-ws + e0) - 1j * eta
            
            # CORRECTED Continued Fraction
            g_l = np.zeros_like(z)
            for k in range(len(a_l) - 1, -1, -1):
                b2 = b_l[k]**2 if k < len(b_l) else 0.0
                g_l = 1.0 / (z - a_l[k] - b2 * g_l)
            
            G_all[:, i, i] -= norm_l_sq * g_l
            
    return G_all


def create_hubbard_V(M, U_val):
    """
    Creates the Hubbard U tensor <pq|V|rs> (Physicist's notation) for the 
    AlphaFirst spin-orbital ordering.
    """
    K = 2 * M
    V = np.zeros((K, K, K, K), dtype=np.complex128)
    for i in range(M): # Loop over spatial sites
        alpha_i = i
        beta_i  = i + M
        
        # The general 2e Hamiltonian has a 1/2 prefactor. To get a final energy of U,
        # the integral <iα, iβ | V | iα, iβ> must be 2*U.
        # This corresponds to V_pqrs with p=iα, q=iβ, r=iα, s=iβ
        V[alpha_i, beta_i, alpha_i, beta_i] = 2.0 * U_val
        
    return V

def get_hubbard_dimer_ed_ref(t, U, M):
    K = 2 * M
    c_dag = [cc.get_creation_operator(K, i + 1) for i in range(K)]
    c = [cc.get_annihilation_operator(K, i + 1) for i in range(K)]
    H = scipy.sparse.csr_matrix((2**K, 2**K), dtype=np.complex128)
    if M == 2:
        H += -t * (c_dag[0] @ c[1] + c_dag[1] @ c[0])
        H += -t * (c_dag[0+M] @ c[1+M] + c_dag[1+M] @ c[0+M])
    for i in range(M):
        n_up = c_dag[i] @ c[i]
        n_down = c_dag[i+M] @ c[i+M]
        H += U * (n_up @ n_down)
    return H
####################################################################
