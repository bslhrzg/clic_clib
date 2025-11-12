import numpy as np
import h5py
import clic_clib as qc
from test_utils import *

def test_oo_kernel_consistency():
    print("--- Isolating and Testing the C++ OO Kernel ---")
    
    # 1. Load and transform the integrals to get our test data
    h0_interleaved, U_interleaved, Ne, Enuc, M, K = load_integrals_from_h5("h2o_h0U.h5")
    h0_af, U_af = qc.transform_integrals_interleaved_to_alphafirst(
        h0_interleaved, U_interleaved, M
    )
    
    # 2. Define the occupied orbitals for the HF state in the AlphaFirst basis
    occupied_af = list(range(Ne // 2)) + [i + M for i in range(Ne // 2)]
    print(f"\nOccupied orbitals in AlphaFirst basis: {occupied_af}")

    # --- Method 1: Pure Python calculation on the transformed arrays (Our Reference) ---
    print("\nCalculating HF energy in Python using AlphaFirst arrays...")
    e1_py = sum(h0_af[i, i] for i in occupied_af)
    e2_py = sum((U_af[i, j, i, j] - U_af[i, j, j, i]) for i in occupied_af for j in occupied_af)
    electronic_hf_py = e1_py + 0.5 * e2_py
    print(f"  Python electronic energy = {electronic_hf_py:.12f}")

    # --- Method 2: C++ calculation ---
    print("\nCalculating HF energy in C++...")
    
    # CRITICAL STEP: Ensure arrays are C-contiguous before passing to C++
    h0_af_clean = np.ascontiguousarray(h0_af, dtype=np.complex128)
    U_af_clean = np.ascontiguousarray(U_af, dtype=np.complex128)

    # Call the C++ kernel ONCE with the cleaned arrays
    electronic_hf_cpp = qc.KL(occupied_af, occupied_af, K, h0_af_clean, U_af_clean)

    print(f"  C++ electronic energy    = {np.real(electronic_hf_cpp):.12f}")

    # --- 3. Compare ---
    print("\nComparing Python vs C++ results...")
    np.testing.assert_allclose(electronic_hf_py, np.real(electronic_hf_cpp), atol=1e-9)
    assert np.isclose(np.imag(electronic_hf_cpp), 0.0, atol=1e-9)
    print("✅ SUCCESS: The C++ OO kernel calculates the same value as the Python loop.")

if __name__ == "__main__":
    test_oo_kernel_consistency()
