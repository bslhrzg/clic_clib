// bindings/bind_ed_tools.cpp
#include "bind_common.hpp"

static py::object get_annihilation_operator_py(int num_spin_orbitals,
                                               int orbital_index_1based)
{
    int i0 = orbital_index_1based - 1;
    auto triplets = get_annihilation_operator_sparse(num_spin_orbitals, i0);
    int dim = 1 << num_spin_orbitals;
    return scipy_csr_from_triplets(dim, triplets);
}

static py::object get_creation_operator_py(int num_spin_orbitals,
                                           int orbital_index_1based)
{
    int i0 = orbital_index_1based - 1;
    auto triplets = get_creation_operator_sparse(num_spin_orbitals, i0);
    int dim = 1 << num_spin_orbitals;
    return scipy_csr_from_triplets(dim, triplets);
}

static py::object build_ham_naive_py(const std::vector<SlaterDeterminant>& basis,
                                     py::array H, py::array V, double tol)
{
    if (basis.empty()) return py::none();
    CSR csr = build_hamiltonian_naive(basis, make_H1(H), make_ERI(V), tol);
    return scipy_csr_from_CSR(csr);
}

static py::object build_ham_openmp_py(const std::vector<SlaterDeterminant>& basis,
                                      py::array H, py::array V,
                                      bool enable_magnetic, double tol)
{
    if (basis.empty()) return py::none();
    CSR csr = build_hamiltonian_openmp(
        basis, make_H1(H), make_ERI(V), enable_magnetic, tol
    );
    return scipy_csr_from_CSR(csr);
}

void bind_ed_tools_and_matrices(py::module_ &m)
{
    m.def("get_annihilation_operator", &get_annihilation_operator_py,
          py::arg("num_spin_orbitals"), py::arg("orbital_index_1based"));

    m.def("get_creation_operator", &get_creation_operator_py,
          py::arg("num_spin_orbitals"), py::arg("orbital_index_1based"));

    m.def("build_hamiltonian_naive", &build_ham_naive_py,
          py::arg("basis"), py::arg("H"), py::arg("V"),
          py::arg("tol") = 1e-16);

    m.def("build_hamiltonian_openmp", &build_ham_openmp_py,
          py::arg("basis"), py::arg("H"), py::arg("V"),
          py::arg("enable_magnetic") = true, py::arg("tol") = 1e-16);

    m.def("KL",
          [](const std::vector<int>& occ1, const std::vector<int>& occ2,
             size_t K, py::array H, py::array V) {
              Determinant D1(K, occ1);
              Determinant D2(K, occ2);
              return KL(D1, D2, make_H1(H), make_ERI(V));
          },
          py::arg("occ1"), py::arg("occ2"), py::arg("K"),
          py::arg("H"), py::arg("V"));
}