// bindings/bind_wavefunction.cpp
#include "bind_common.hpp"

void bind_wavefunction(py::module_ &m)
{
    using Coeff = Wavefunction::Coeff;

    auto wf_cls = py::class_<Wavefunction>(m, "Wavefunction")
        .def(py::init<std::size_t>(), py::arg("n_spatial"))
        .def(py::init<const Wavefunction &>())
        .def(py::init<std::size_t, const Wavefunction::Data&>(),
             py::arg("n_spatial"), py::arg("data"))
        .def(py::init<std::size_t,
                      const std::vector<SlaterDeterminant>&,
                      const std::vector<Coeff>&,
                      bool>(),
             py::arg("n_spatial"), py::arg("basis"),
             py::arg("amplitudes"), py::arg("keep_zeros") = true)
        .def("add_term", &Wavefunction::add_term,
             py::arg("det"), py::arg("coeff"), py::arg("tol") = 0)
        .def("add_wavefunction", &Wavefunction::add_wavefunction,
             py::arg("other"), py::arg("scale") = cx{1.0, 0.0},
             "Adds another wavefunction scaled by a coefficient, modifying this one in-place.")
        .def("data", &Wavefunction::data, py::return_value_policy::reference_internal)
        .def("prune", &Wavefunction::prune, py::arg("threshold"))
        .def("normalize", &Wavefunction::normalize, py::arg("tol") = 1e-15)
        .def("dot", &Wavefunction::dot, py::arg("other"),
             "Computes dot product <self|other>")
        .def("__iadd__", [](Wavefunction &wf, const Wavefunction &other) {
                wf.add_wavefunction(other);
                return wf;
             }, py::is_operator())
        .def("__add__", [](const Wavefunction &wf, const Wavefunction &other) {
                Wavefunction result = wf;
                result.add_wavefunction(other);
                return result;
             }, py::is_operator())
        .def("__sub__", [](const Wavefunction &wf, const Wavefunction &other) {
                Wavefunction result = wf;
                result.add_wavefunction(other, {-1.0, 0.0});
                return result;
             }, py::is_operator())
        .def("__mul__", [](const Wavefunction &wf, cx scale) {
                Wavefunction result(wf.num_spatial_orbitals());
                for (const auto& [det, coeff] : wf.data())
                    result.add_term(det, coeff * scale);
                return result;
             }, py::is_operator())
        .def("__rmul__", [](const Wavefunction &wf, cx scale) {
                Wavefunction result(wf.num_spatial_orbitals());
                for (const auto& [det, coeff] : wf.data())
                    result.add_term(det, coeff * scale);
                return result;
             }, py::is_operator())
        .def("get_basis", &Wavefunction::basis_sorted,
             "Returns a sorted list of SlaterDeterminant objects in the wavefunction.")
        .def("get_amplitudes", [](Wavefunction& wf) {
            auto basis = wf.basis_sorted();
            auto coeffs_vec = wf.coeffs_sorted(basis);
            return py::array_t<cx>(coeffs_vec.size(), coeffs_vec.data());
        }, "Returns a NumPy array of amplitudes, sorted according to the basis.")
        .def_property_readonly("n_spatial", &Wavefunction::num_spatial_orbitals)
        .def("__repr__", [](const Wavefunction& wf) {
            return "<Wavefunction with " +
                   std::to_string(wf.data().size()) + " terms>";
        });

    // extra methods that you previously added via m.attr("Wavefunction")
    wf_cls
        .def("amplitude", [](const Wavefunction& wf, const SlaterDeterminant& det) {
            const auto& M = wf.data();
            auto it = M.find(det);
            return (it == M.end()) ? cx{0.0, 0.0} : it->second;
        }, py::arg("det"), R"doc(Returns ⟨det|ψ⟩, or 0 if absent.)doc")
        .def("__contains__", [](const Wavefunction& wf, const SlaterDeterminant& det) {
            return wf.data().find(det) != wf.data().end();
        }, py::arg("det"));
}