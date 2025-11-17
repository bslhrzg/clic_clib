// bindings/bind_screened_hamiltonian.cpp
#include "bind_common.hpp"

void bind_screened_hamiltonian(py::module_ &m)
{
    py::class_<HamiltonianTables>(m, "HamiltonianTables",
        "Pre-computed tables for efficient H|psi> application.")
        .def(py::init<>())
        .def("__repr__", [](const HamiltonianTables& sh) {
            return "<HamiltonianTables for K=" +
                   std::to_string(sh.n_spin_orbitals) + " spin-orbitals>";
        });

    m.def("build_hamiltonian_tables",
        [](py::array H, py::array V, double tol) {
            size_t K = H.request().shape[0];
            return build_hamiltonian_tables(K, make_H1(H), make_ERI(V), tol);
        },
        py::arg("H"), py::arg("V"), py::arg("tol"),
        R"doc(
        Builds pre-computed screening tables for fast Hamiltonian application based on integral values.
        )doc");

    m.def("apply_hamiltonian",
        [](const Wavefunction& psi, const HamiltonianTables& sh,
           py::array H, py::array V, double tol_element) {
            return apply_hamiltonian(psi, sh, make_H1(H), make_ERI(V), tol_element);
        },
        py::arg("psi"), py::arg("screened_h0U"),
        py::arg("H"), py::arg("V"), py::arg("tol_element"),
        R"doc(
        Applies the Hamiltonian to a wavefunction using pre-built screening tables.
        )doc");

    m.def("get_connected_basis",
        [](const Wavefunction& psi, const HamiltonianTables& sh) {
            return get_connected_basis(psi, sh);
        },
        py::arg("psi"), py::arg("screened_h0U"),
        R"doc(
        Using pre-built screening tables of h0U, get the connected basis to psi given the Hamiltonian.
        )doc");

    m.def("build_fixed_basis_tables",
        [](const HamiltonianTables& sh_full,
           const std::vector<SlaterDeterminant>& basis,
           size_t M) {
            return build_fixed_basis_tables(sh_full, basis, M);
        },
        py::arg("screened_full"),
        py::arg("basis"),
        py::arg("M"));

    m.def("apply_hamiltonian_fixed_basis",
        [](const Wavefunction& psi,
           const HamiltonianTables& sh_fb,
           const std::vector<SlaterDeterminant>& basis,
           py::array H,
           py::array V,
           double tol_element) {
            return apply_hamiltonian_fixed_basis(
                psi, sh_fb, basis,
                make_H1(H), make_ERI(V), tol_element
            );
        },
        py::arg("psi"),
        py::arg("screened_H_fixed_basis"),
        py::arg("basis"),
        py::arg("H"),
        py::arg("V"),
        py::arg("tol_element"));


    m.def("build_hamiltonian_matrix_fixed_basis",
        [](const HamiltonianTables& sh_fb,
           const std::vector<SlaterDeterminant>& basis,
           py::array H,
           py::array V,
           double tol_element) -> py::object {
            CSR csr = build_hamiltonian_matrix_fixed_basis(
                sh_fb, basis, make_H1(H), make_ERI(V), tol_element
            );
            return scipy_csr_from_CSR(csr);
        },
        py::arg("screened_H_fixed_basis"),
        py::arg("basis"),
        py::arg("H"),
        py::arg("V"),
        py::arg("tol_element"),
        R"doc(
        Build the Hamiltonian matrix in the given fixed determinant basis,
        using pre-built screening tables restricted to that basis.

        Returns
        -------
        scipy.sparse.csr_matrix
            Hamiltonian in the fixed basis.
        )doc");
}
