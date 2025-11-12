import numpy as np
import scipy.sparse
from scipy.linalg import eigh
from itertools import combinations
import clic_clib as qc
from test_utils import *
# --- (Helper functions get_fci_basis, create_hubbard_V, get_hubbard_dimer_ed_ref remain the same) ---





# --- Main Test Function (Modified for Debugging) ---
def test_hubbard_comparison():
    print("--- DEBUGGING Hubbard Dimer (N=2, full subspace) ---")
    np.set_printoptions(linewidth=200, precision=4, suppress=True)
    
    # Model parameters
    M = 2
    K = 2 * M
    Nelec = 2
    t = 1.0
    U = 4.0 # Using U=4.0 for a more pronounced interaction effect

    # --- Part 1: ED Tools Reference ---
    print("\n--- Method 1: ED Tools ---")
    H_ed_full = get_hubbard_dimer_ed_ref(t, U, M)
    states_2e_indices_ed = [i for i in range(2**K) if bin(i).count('1') == Nelec]
    H_ed_2e = H_ed_full[np.ix_(states_2e_indices_ed, states_2e_indices_ed)].toarray()
    
    print("ED Tools Basis (integer representation):")
    # This is the order of rows/cols for the matrix below
    print(states_2e_indices_ed)
    # Expected: [3, 5, 6, 9, 10, 12]
    
    print("\nED Tools Hamiltonian Matrix (N=2 subspace):")
    print(np.real(H_ed_2e))

    # --- Part 2: Slater-Condon Builder ---
    print("\n--- Method 2: Slater-Condon (Naive Builder) ---")
    basis = get_fci_basis(M, Nelec)
    
    # Let's print our basis to be sure
    print("Slater-Condon Basis (alpha | beta occupations):")
    for i, det in enumerate(basis):
        print(f"{i}: {det.alpha_occupied_indices()} | {det.beta_occupied_indices()}")
        
    H1 = np.zeros((K, K), dtype=np.complex128)
    H1[0, 1] = H1[1, 0] = -t
    H1[2, 3] = H1[3, 2] = -t
    V = create_hubbard_V(M, U)
    
    # Use the naive builder for a clear 1-to-1 comparison
    H_sc = qc.build_hamiltonian_naive(basis, H1, V).toarray()
    print("\nSlater-Condon Hamiltonian Matrix:")
    print(np.real(H_sc))

    # --- Part 3: Explicit KL calls for specific matrix elements ---
    print("\n--- Method 3: Manual KL Calls ---")
    
    # Let's check a diagonal element: <D_4 | H | D_4>
    # D_4 has occ_a=[1], occ_b=[1]. This is |α₁, β₁>, doubly occupied site 1.
    # Expected value: U
    det4 = basis[4]
    occ_a4 = det4.alpha_occupied_indices()
    occ_b4 = det4.beta_occupied_indices()
    combined_occ4 = occ_a4 + [i + M for i in occ_b4] # AlphaFirst: [1, 3]
    val_diag = qc.KL(combined_occ4, combined_occ4, K, H1, V)
    print(f"Manual KL for <D_4|H|D_4> (expected {U:.1f}): {np.real(val_diag):.4f}")

    # Let's check an off-diagonal hopping term: <D_0 | H | D_1>
    # D_0: occ_a=[], occ_b=[0,1] -> |β₀, β₁>
    # D_1: occ_a=[0], occ_b=[0]   -> |α₀, β₀>
    # These are not connected by H. Hamming distance is 2, but not a simple excitation.
    # Expected value: 0
    det0, det1 = basis[0], basis[1]
    combined_occ0 = det0.alpha_occupied_indices() + [i + M for i in det0.beta_occupied_indices()]
    combined_occ1 = det1.alpha_occupied_indices() + [i + M for i in det1.beta_occupied_indices()]
    val_offdiag1 = qc.KL(combined_occ0, combined_occ1, K, H1, V)
    print(f"Manual KL for <D_0|H|D_1> (expected 0.0): {np.real(val_offdiag1):.4f}")

    # Let's check a proper off-diagonal hopping term: <D_5 | H | D_3>
    # D_5: occ_a=[0,1], occ_b=[]   -> |α₀, α₁>
    # D_3: occ_a=[0],   occ_b=[1]   -> |α₀, β₁>
    # These are not connected by H.
    
    # Let's check < D_sz=0 | H | D_sz=0 > hopping: <|α₀β₀| H |α₁β₀|>
    # D_1: |α₀β₀>, occ_a=[0], occ_b=[0]. combined=[0, 2]
    # D_2: |α₁β₀>, occ_a=[1], occ_b=[0]. combined=[1, 2]
    # This is a single excitation α₀ -> α₁. Expected value: -t
    det1, det2 = basis[1], basis[2]
    combined_occ1 = det1.alpha_occupied_indices() + [i + M for i in det1.beta_occupied_indices()]
    combined_occ2 = det2.alpha_occupied_indices() + [i + M for i in det2.beta_occupied_indices()]
    val_offdiag2 = qc.KL(combined_occ1, combined_occ2, K, H1, V)
    print(f"Manual KL for <D_1|H|D_2> (expected {-t:.1f}): {np.real(val_offdiag2):.4f}")


if __name__ == "__main__":
    test_hubbard_comparison()