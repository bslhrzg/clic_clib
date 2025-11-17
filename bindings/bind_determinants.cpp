// bindings/bind_slater.cpp
#include "bind_common.hpp"

void bind_enums_and_slater(py::module_ &m)
{
    py::enum_<Spin>(m, "Spin")
        .value("Alpha", Spin::Alpha)
        .value("Beta",  Spin::Beta)
        .export_values();

    py::enum_<SpinOrbitalOrder>(m, "SpinOrbitalOrder")
        .value("AlphaFirst", SpinOrbitalOrder::AlphaFirst)
        .value("Interleaved", SpinOrbitalOrder::Interleaved)
        .export_values();

    py::class_<SlaterDeterminant::OpResult>(m, "SlaterDeterminantOpResult")
        .def_readonly("det",  &SlaterDeterminant::OpResult::det)
        .def_readonly("sign", &SlaterDeterminant::OpResult::sign)
        .def("__repr__", [](const SlaterDeterminant::OpResult &res) {
            return "<SlaterDeterminantOpResult: sign=" + std::to_string(res.sign) + ">";
        });

    py::class_<SlaterDeterminant>(m, "SlaterDeterminant",
        "Represents a Slater determinant with separate alpha and beta spin strings.")
        .def(py::init<std::size_t, const std::vector<int>&, const std::vector<int>&>(),
             py::arg("n_spatial"), py::arg("occ_alpha0"), py::arg("occ_beta0"),
             "Constructs a determinant for M spatial orbitals with given occupations.")
        .def("alpha_occupied_indices", [](const SlaterDeterminant& det) {
            std::vector<int> occ;
            for (auto it = det.alpha().begin_occ(); it != det.alpha().end_occ(); ++it)
                occ.push_back(*it);
            return occ;
        })
        .def("beta_occupied_indices", [](const SlaterDeterminant& det) {
            std::vector<int> occ;
            for (auto it = det.beta().begin_occ(); it != det.beta().end_occ(); ++it)
                occ.push_back(*it);
            return occ;
        })
        .def_static("create",     &SlaterDeterminant::create,
                    py::arg("s"), py::arg("i0"), py::arg("spin"),
                    py::arg("order") = SpinOrbitalOrder::AlphaFirst)
        .def_static("annihilate", &SlaterDeterminant::annihilate,
                    py::arg("s"), py::arg("i0"), py::arg("spin"),
                    py::arg("order") = SpinOrbitalOrder::AlphaFirst)
        .def_static("apply_excitation_single", &SlaterDeterminant::apply_excitation_single)
        .def_static("apply_excitation_double", &SlaterDeterminant::apply_excitation_double)
        .def_property_readonly("n_spatial", &SlaterDeterminant::num_spatial_orbitals)
        .def("get_occupied_spin_orbitals", [](const SlaterDeterminant& det) {
            std::vector<int> occ;
            const auto M = det.num_spatial_orbitals();
            for (auto it = det.alpha().begin_occ(); it != det.alpha().end_occ(); ++it)
                occ.push_back(*it);
            for (auto it = det.beta().begin_occ(); it != det.beta().end_occ(); ++it)
                occ.push_back(*it + static_cast<int>(M));
            std::sort(occ.begin(), occ.end());
            return occ;
        }, "Returns the combined list of occupied spin-orbital indices (AlphaFirst ordering).")
        .def(py::self < py::self)
        .def(py::self == py::self)
        .def(py::hash(py::self))
        .def("__repr__", [](const SlaterDeterminant& det) {
            py::list alpha, beta;
            for (auto it = det.alpha().begin_occ(); it != det.alpha().end_occ(); ++it)
                alpha.append(*it);
            for (auto it = det.beta().begin_occ(); it != det.beta().end_occ(); ++it)
                beta.append(*it);
            return "<SlaterDeterminant α=" + std::string(py::str(alpha)) +
                   " β=" + std::string(py::str(beta)) + ">";
        });
}