import numpy as np 
from itertools import combinations
import clic_clib as cc

# Some utilitaries that would be needed in general 

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

####################################################################

# If U comes from chemistry program, we expect a different order
def U_chem_to_phys(U):
    U_phys = np.ascontiguousarray(U.transpose(0,2,1,3))
    return U_phys

# If clic_clib is used on spin symmetric models
# We need to expand the hamiltonian to include spin 
def double_h(h_core, M):
    """Converts spatial one-electron integrals to spin-orbital form (AlphaFirst)."""
    K = 2 * M
    h0 = np.zeros((K, K))
    for p in range(M):
        for q in range(M):
            # Alpha-alpha block
            h0[p, q] = h_core[p, q]
            # Beta-beta block
            h0[p + M, q + M] = h_core[p, q]
    return h0

def umo2so(U_mo, M):
    """
    Converts spatial physicist's integrals <pq|V|rs> to spin-orbital
    physicist's integrals <ij|V|kl> in AlphaFirst ordering.
    """
    K = 2 * M
    U_so = np.zeros((K, K, K, K))
    # U_mo[p,q,r,s] = <pq|V|rs>
    for p in range(M):
        for q in range(M):
            for r in range(M):
                for s in range(M):
                    val = U_mo[p, q, r, s]
                    if abs(val) > 1e-12:
                        p_a, p_b = p, p + M
                        q_a, q_b = q, q + M
                        r_a, r_b = r, r + M
                        s_a, s_b = s, s + M
                        
                        # αααα
                        U_so[p_a, q_a, r_a, s_a] = val
                        # ββββ
                        U_so[p_b, q_b, r_b, s_b] = val
                        # αβαβ
                        U_so[p_a, q_b, r_a, s_b] = val
                        # βαβα
                        U_so[p_b, q_a, r_b, s_a] = val
    return U_so


def transform_integrals_interleaved_to_alphafirst(h0_int, U_int=None, M=None):
    """
    Transforms integrals from spin-interleaved to AlphaFirst (spin-blocked) ordering

    Args:
        h0_int (np.ndarray): One-body integrals in interleaved basis.
        U_int (np.ndarray): Two-body integrals in interleaved basis.
        M (int): Number of spatial orbitals.

    Returns:
        tuple[np.ndarray, np.ndarray]: h0_af, U_af in AlphaFirst basis.
    """
    if M is None : 
        M = np.shape(h0_int)[0] // 2
    
    K = 2 * M
    
    # Create the mapping from the new "AlphaFirst" index to the old "interleaved" index.
    # This is the permutation vector.
    # af_map[i_alphafirst] = i_interleaved
    af_map = np.zeros(K, dtype=int)
    
    # First M indices in the new basis are the alpha spins.
    # Their corresponding indices in the old basis are 0, 2, 4, ...
    af_map[:M] = np.arange(0, K, 2)
    
    # Last M indices in the new basis are the beta spins.
    # Their corresponding indices in the old basis are 1, 3, 5, ...
    af_map[M:] = np.arange(1, K, 2)

    # Use np.ix_ to create indexers that permute the axes of the arrays.
    # This is equivalent to the loops but is executed in highly optimized C/Fortran code.
    # h0_af[i, j] = h0_int[af_map[i], af_map[j]]
    h0_af = h0_int[np.ix_(af_map, af_map)]
    h0_af = np.ascontiguousarray(h0_af, dtype=np.complex128)

    if U_int is not None:
        # For the 4D tensor, we permute all four axes using the same map.
        # U_af[i,j,k,l] = U_int[af_map[i], af_map[j], af_map[k], af_map[l]]
        U_af = U_int[np.ix_(af_map, af_map, af_map, af_map)]
    
        U_af = np.ascontiguousarray(U_af, dtype=np.complex128)
        
        return h0_af, U_af
    
    else:
        return h0_af

def transform_h0_alphafirst_to_interleaved(h0_af, M=None):
    """
    Transforms one particle h0 from AlphaFirst to spin-interleaved ordering
    """
    if M is None:
        M = np.shape(h0_af)[0] // 2

    K = 2 * M
    
    # Create the mapping from the new "AlphaFirst" index to the "interleaved" index.
    af_map = np.zeros(K, dtype=int)
    af_map[:M] = np.arange(0, K, 2)
    af_map[M:] = np.arange(1, K, 2)

    inverse_map = np.argsort(af_map)

    h0_ab = h0_af[np.ix_(inverse_map, inverse_map)]

    
    return h0_ab

