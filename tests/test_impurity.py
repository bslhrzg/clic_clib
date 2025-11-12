import numpy as np
from scipy.sparse.linalg import eigsh
import time
import matplotlib.pyplot as plt
from test_utils import *

# --- Main Test Execution ---
if __name__ == "__main__":
    # --- System Setup ---
    nb = 7
    M = 1 + nb
    u = 1
    mu = u/2
    Nelec = M
    e_bath = np.linspace(-2.0, 2.0, nb)
    if nb == 1 :
        e_bath = [0.0]
    print("e_bath = ",e_bath)
    V_bath = np.full(nb, 0.1)
    
    h0, U_mat = get_impurity_integrals(M, u, e_bath, V_bath, mu)
    h0_clean = np.ascontiguousarray(h0, dtype=np.complex128)
    U_clean = np.ascontiguousarray(U_mat, dtype=np.complex128)

    print("h0 = ")
    print(h0)

    # --- Find Ground State ---
    basis = get_fci_basis(M, Nelec)
    print(f"basis size = {len(basis)}")
    H_sparse = cc.build_hamiltonian_openmp(basis, h0_clean, U_clean)
    eigvals, eigvecs = eigsh(H_sparse, k=1, which='SA')
    e0 = eigvals[0]
    psi0_wf = cc.Wavefunction(M, basis, eigvecs[:, 0])
    
    print(f"Impurity Model Ground State Energy: {e0:.6f}")

    # --- Matrix-Free Lanczos Calculation ---
    one_body_terms = get_one_body_terms(h0_clean, M)
    two_body_terms = get_two_body_terms(U_clean, M)
    H_op = make_H_on_psi(one_body_terms, two_body_terms)
    
    ws = np.linspace(-6, 6, 1001)
    eta = 0.02
    L = 100
    
    # Call the SCALAR version
    impurity_indices = [0, M]  # same as before
    print(f"\nRunning EFFICIENT SCALAR Lanczos for impurity orbitals {impurity_indices}...")
    t_start = time.time()
    G_scalar = green_function_scalar_lanczos_wf(H_op, M, psi0_wf, e0, L, ws, eta, impurity_indices)
    t_end = time.time()
    print(f"  Calculation finished in {t_end-t_start:.3f}s")
    A_scalar = -(1/np.pi) * np.imag(G_scalar)
    
    # --- Verify they give the same diagonal elements ---
    #np.testing.assert_allclose(A_block, A_scalar, atol=1e-6)
    #print("\n✅ SUCCESS: Scalar Lanczos results match Block Lanczos diagonal.")
    
    A_mat_lanc_wf = -(1/np.pi) * np.imag(G_scalar)
    
    # --- Plotting to Compare ---
    # Plot the impurity spectral functions (G_00_αα and G_00_ββ)
    plt.figure(figsize=(8, 4))
    for (i,ii) in enumerate(impurity_indices):
        plt.plot(ws, i*5+(A_mat_lanc_wf[:, ii, ii]), label="A_sc_"+str(i)+"(ω)")
    plt.title("Impurity Spectral Function for Anderson Model (Matrix-Free Lanczos)")
    plt.xlabel("Frequency (ω)")
    plt.ylabel("A(ω)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("\nScript finished. Check the plot for the spectral function.")