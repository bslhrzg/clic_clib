// bindings/bind_operators.cpp
#include "bind_common.hpp"

void bind_operators_and_connections(py::module_ &m)
{
    m.def("apply_creation", &apply_creation,
          py::arg("wf"), py::arg("i0"), py::arg("spin"),
          py::arg("order") = SpinOrbitalOrder::AlphaFirst);

    m.def("apply_annihilation", &apply_annihilation,
          py::arg("wf"), py::arg("i0"), py::arg("spin"),
          py::arg("order") = SpinOrbitalOrder::AlphaFirst);

    // apply_one_body_operator
    m.def("apply_one_body_operator",
          [](const Wavefunction& wf,
             const std::vector<std::tuple<size_t, size_t, Spin, Spin, Wavefunction::Coeff>>& term_tuples) {
              std::vector<OneBodyTerm> terms;
              terms.reserve(term_tuples.size());
              for (const auto& t : term_tuples) {
                  terms.emplace_back(OneBodyTerm{
                      std::get<0>(t), std::get<1>(t),
                      std::get<2>(t), std::get<3>(t),
                      std::get<4>(t)
                  });
              }
              return apply_one_body_operator(wf, terms);
          },
          py::arg("wf"), py::arg("terms"));

    // apply_two_body_operator
    m.def("apply_two_body_operator",
          [](const Wavefunction& wf,
             const std::vector<std::tuple<size_t, size_t, size_t, size_t,
                                          Spin, Spin, Spin, Spin,
                                          Wavefunction::Coeff>>& term_tuples) {
              std::vector<TwoBodyTerm> terms;
              terms.reserve(term_tuples.size());
              for (const auto& t : term_tuples) {
                  terms.emplace_back(TwoBodyTerm{
                      std::get<0>(t), std::get<1>(t),
                      std::get<2>(t), std::get<3>(t),
                      std::get<4>(t), std::get<5>(t),
                      std::get<6>(t), std::get<7>(t),
                      std::get<8>(t)
                  });
              }
              return apply_two_body_operator(wf, terms);
          },
          py::arg("wf"), py::arg("terms"));

    // connectivity
    m.def("get_connections_one_body",
          [](const std::vector<SlaterDeterminant>& basis,
             const std::vector<std::tuple<size_t, size_t, Spin, Spin, Wavefunction::Coeff>>& term_tuples) {
              std::vector<OneBodyTerm> terms;
              terms.reserve(term_tuples.size());
              for (const auto& t : term_tuples) {
                  terms.emplace_back(OneBodyTerm{
                      std::get<0>(t), std::get<1>(t),
                      std::get<2>(t), std::get<3>(t),
                      std::get<4>(t)
                  });
              }
              return get_connections_one_body(basis, terms);
          },
          py::arg("basis"), py::arg("terms"));

    m.def("get_connections_two_body",
          [](const std::vector<SlaterDeterminant>& basis,
             const std::vector<std::tuple<size_t, size_t, size_t, size_t,
                                          Spin, Spin, Spin, Spin,
                                          Wavefunction::Coeff>>& term_tuples) {
              std::vector<TwoBodyTerm> terms;
              terms.reserve(term_tuples.size());
              for (const auto& t : term_tuples) {
                  terms.emplace_back(TwoBodyTerm{
                      std::get<0>(t), std::get<1>(t),
                      std::get<2>(t), std::get<3>(t),
                      std::get<4>(t), std::get<5>(t),
                      std::get<6>(t), std::get<7>(t),
                      std::get<8>(t)
                  });
              }
              return get_connections_two_body(basis, terms);
          },
          py::arg("basis"), py::arg("terms"));
}