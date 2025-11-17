// bindings/bind_common.hpp
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/operators.h>

#include <complex>
#include <vector>
#include <tuple>

#include "determinants.h"
#include "wavefunction.h"
#include "nbody.h"
#include "slater_condon.h"
#include "hamiltonian.h"
#include "ed_tools.h"
#include "applyH.h"

namespace py = pybind11;
using cx = std::complex<double>;
using namespace ci;

// ---- shared helpers --------------------------------------------------------

inline py::object scipy_csr_from_CSR(const CSR& csr)
{
    py::module_ sp_sparse = py::module_::import("scipy.sparse");
    auto np_data   = py::array_t<cx>(csr.data.size(), csr.data.data());
    auto np_indices= py::array_t<int32_t>(csr.indices.size(), csr.indices.data());
    auto np_indptr = py::array_t<int64_t>(csr.indptr.size(), csr.indptr.data());
    py::tuple shape = py::make_tuple((py::int_)csr.n_rows, (py::int_)csr.n_cols);
    return sp_sparse.attr("csr_matrix")(py::make_tuple(np_data, np_indices, np_indptr),
                                        py::arg("shape")=shape);
}

inline py::object scipy_csr_from_triplets(
    int dim, const std::vector<Triplet>& triplets)
{
    py::module_ sp_sparse = py::module_::import("scipy.sparse");
    std::vector<int> rows, cols;
    std::vector<cx> vals;
    rows.reserve(triplets.size());
    cols.reserve(triplets.size());
    vals.reserve(triplets.size());
    for (const auto& t : triplets) {
        rows.push_back(t.row);
        cols.push_back(t.col);
        vals.push_back(t.value);
    }
    auto np_data = py::array_t<cx>(vals.size(), vals.data());
    auto np_rows = py::array_t<int>(rows.size(), rows.data());
    auto np_cols = py::array_t<int>(cols.size(), cols.data());
    py::tuple ij = py::make_tuple(np_rows, np_cols);
    py::tuple data_ij = py::make_tuple(np_data, ij);
    py::tuple shape = py::make_tuple(dim, dim);
    return sp_sparse.attr("csr_matrix")(data_ij, py::arg("shape")=shape);
}

inline H1View make_H1(py::array H) {
    py::buffer_info buf = H.request();
    if (buf.ndim != 2 || buf.shape[0] != buf.shape[1])
        throw std::runtime_error("H must be a square 2D NumPy array.");
    return H1View{reinterpret_cast<const cx*>(buf.ptr), (size_t)buf.shape[0]};
}

inline ERIView make_ERI(py::array V) {
    py::buffer_info buf = V.request();
    if (buf.ndim != 4 || buf.shape[0] != buf.shape[1] ||
        buf.shape[0] != buf.shape[2] || buf.shape[0] != buf.shape[3])
        throw std::runtime_error("V must be a 4D NumPy array of shape (K,K,K,K).");
    return ERIView{reinterpret_cast<const cx*>(buf.ptr), (size_t)buf.shape[0]};
}

// ---- forward declarations of binder functions ------------------------------

void bind_enums_and_slater(py::module_ &m);
void bind_wavefunction(py::module_ &m);
void bind_operators_and_connections(py::module_ &m);
void bind_screened_hamiltonian(py::module_ &m);
void bind_ed_tools_and_matrices(py::module_ &m);