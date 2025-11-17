// cpp_src/applyH.cpp
#include "applyH.h"
#include "hamiltonian.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <omp.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>




namespace ci {

static inline Determinant combined_from(const SlaterDeterminant& sd) {
    const size_t M = sd.num_spatial_orbitals();
    Determinant D(2 * M);
    for (auto it = sd.alpha().begin_occ(); it != sd.alpha().end_occ(); ++it)
        D.set(static_cast<size_t>(*it));
    for (auto it = sd.beta().begin_occ(); it != sd.beta().end_occ(); ++it)
        D.set(static_cast<size_t>(M + *it));
    return D;
}


HamiltonianTables build_hamiltonian_tables(
    size_t K, const H1View& H, const ERIView& V, double tol)
{
    HamiltonianTables sh;
    sh.n_spin_orbitals = K;

    // --- TableSh0 (build_Sh0) ---
    for (uint32_t r = 0; r < K; ++r) {
        std::vector<uint32_t> targets;
        for (uint32_t p = 0; p < K; ++p) {
            if (p == r) continue;
            if (std::abs(H(p, r)) >= tol) {
                targets.push_back(p);
            }
        }
        if (!targets.empty()) {
            sh.sh0[r] = std::move(targets);
        }
    }

    // --- TableSU (build_SU detailed=True) ---
    for (uint32_t i = 0; i < K; ++i) {
        for (uint32_t j = 0; j < K; ++j) {
            if (i == j) continue;
            std::vector<uint32_t> spectators;
            for (uint32_t p = 0; p < K; ++p) {
                if (std::abs(V(j, p, p, i)) > tol ||
                    std::abs(V(j, p, i, p)) > tol ||
                    std::abs(V(p, j, p, i)) > tol ||
                    std::abs(V(p, j, i, p)) > tol) {
                    spectators.push_back(p);
                }
            }
            if (!spectators.empty()) {
                sh.su[i][j] = std::move(spectators);
            }
        }
    }

    // --- TableD (build_D) ---
    for (uint32_t i = 0; i < K; ++i) {
        for (uint32_t j = i + 1; j < K; ++j) {
            uint64_t key_ij = i | (static_cast<uint64_t>(j) << 32);
            std::vector<std::pair<uint32_t, uint32_t>> targets;
            for (uint32_t k = 0; k < K; ++k) {
                for (uint32_t l = 0; l < K; ++l) {
                    if (k == l || k == i || k == j || l == i || l == j) continue;
                    if (std::abs(V(k, l, i, j)) > tol ||
                        std::abs(V(l, k, i, j)) > tol ||
                        std::abs(V(k, l, j, i)) > tol ||
                        std::abs(V(l, k, j, i)) > tol) {
                        targets.emplace_back(k, l);
                    }
                }
            }
            if (!targets.empty()) {
                sh.d[key_ij] = std::move(targets);
            }
        }
    }
    return sh;
}


Wavefunction apply_hamiltonian(
    const Wavefunction& psi,
    const HamiltonianTables& sh,
    const H1View& H,
    const ERIView& V,
    double tol_element)
{
    using Coeff = Wavefunction::Coeff;
    using Map = std::unordered_map<SlaterDeterminant, Coeff>;

    const size_t M = psi.num_spatial_orbitals();
    const size_t K = sh.n_spin_orbitals;

    std::vector<std::pair<SlaterDeterminant, Coeff>> items;
    items.reserve(psi.data().size());
    for (const auto& kv : psi.data()) {
        items.emplace_back(kv.first, kv.second);
    }

    int T = 1;
    #ifdef _OPENMP
    T = omp_get_max_threads();
    #endif
    std::vector<Map> local_maps(T);

    #pragma omp parallel for schedule(dynamic)
    for (size_t item_idx = 0; item_idx < items.size(); ++item_idx) {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif
        auto& acc = local_maps[tid];
        const auto& [det_in, coeff] = items[item_idx];

        std::unordered_set<SlaterDeterminant> seen;
        
        std::vector<int> occ_so_vec;
        occ_so_vec.reserve(K);
        for (auto it = det_in.alpha().begin_occ(); it != det_in.alpha().end_occ(); ++it) occ_so_vec.push_back(*it);
        for (auto it = det_in.beta().begin_occ(); it != det_in.beta().end_occ(); ++it) occ_so_vec.push_back(*it + static_cast<int>(M));
        std::sort(occ_so_vec.begin(), occ_so_vec.end());

        std::unordered_set<int> occ_so_set(occ_so_vec.begin(), occ_so_vec.end());
        
        Determinant d_in_comb(K, occ_so_vec);
        SlaterDeterminant det_out_scratch(M);
        int8_t sign; 

        // 1) Diagonal
        cx val_diag = KL(d_in_comb, d_in_comb, H, V);
        if (std::abs(val_diag) > 0) acc[det_in] += coeff * val_diag;
        seen.insert(det_in);

        // 2) Singles from Sh0
        for (int r : occ_so_vec) {
            auto it = sh.sh0.find(r);
            if (it == sh.sh0.end()) continue;
            for (uint32_t p : it->second) {
                if (occ_so_set.count(p)) continue; 

                if (SlaterDeterminant::apply_excitation_single_fast(
                        det_in, p % M, r % M,
                        p < M ? Spin::Alpha : Spin::Beta,
                        r < M ? Spin::Alpha : Spin::Beta,
                        det_out_scratch, sign)) { 

                    if (seen.count(det_out_scratch)) continue;
                    seen.insert(det_out_scratch);

                    Determinant d_out_comb = combined_from(det_out_scratch); 
                    cx val = KL(d_in_comb, d_out_comb, H, V);
                    if (std::abs(val) > 0) acc[det_out_scratch] += coeff * val;
                }
            }
        }

        // 3) Singles from SU
        for (int r : occ_so_vec) {
            auto it_r = sh.su.find(r);
            if (it_r == sh.su.end()) continue;
            for (const auto& [p, spectators] : it_r->second) {
                if (occ_so_set.count(p)) continue; 

                bool spectator_ok = false;
                for (uint32_t s : spectators) {
                    if (occ_so_set.count(s)) { spectator_ok = true; break; }
                }
                if (!spectator_ok) continue;

                if (SlaterDeterminant::apply_excitation_single_fast(
                        det_in, p % M, r % M,
                        p < M ? Spin::Alpha : Spin::Beta,
                        r < M ? Spin::Alpha : Spin::Beta,
                        det_out_scratch, sign)) { 
                    if (seen.count(det_out_scratch)) continue;
                    seen.insert(det_out_scratch);

                    Determinant d_out_comb = combined_from(det_out_scratch); 
                    cx val = KL(d_in_comb, d_out_comb, H, V);
                    if (std::abs(val) > 0) acc[det_out_scratch] += coeff * val;
                }
            }
        }

        // 4) Doubles from D
        for (size_t i = 0; i < occ_so_vec.size(); ++i) {
            for (size_t j = i + 1; j < occ_so_vec.size(); ++j) {
                uint32_t r = occ_so_vec[i];
                uint32_t s = occ_so_vec[j];
                uint64_t key_rs = r | (static_cast<uint64_t>(s) << 32);

                auto it = sh.d.find(key_rs);
                if (it == sh.d.end()) continue;

                for (const auto& [p, q] : it->second) {
                    if (occ_so_set.count(p) || occ_so_set.count(q)) continue;

                    if (SlaterDeterminant::apply_excitation_double_fast(
                            det_in, p % M, q % M, r % M, s % M,
                            p < M ? Spin::Alpha : Spin::Beta, q < M ? Spin::Alpha : Spin::Beta,
                            r < M ? Spin::Alpha : Spin::Beta, s < M ? Spin::Alpha : Spin::Beta,
                            det_out_scratch, sign)) { 
                        
                        if (seen.count(det_out_scratch)) continue;
                        seen.insert(det_out_scratch);

                        Determinant d_out_comb = combined_from(det_out_scratch); 
                        cx val = KL(d_in_comb, d_out_comb, H, V);
                        if (std::abs(val) > 0) acc[det_out_scratch] += coeff * val;
                    }
                }
            }
        }
    }

    // --- Merge results ---
    Map final_acc;
    size_t hint = 0;
    for(const auto& m : local_maps) hint += m.size();
    final_acc.reserve(hint);
    for (const auto& m : local_maps) {
        for (const auto& kv : m) {
            final_acc[kv.first] += kv.second;
        }
    }

    Wavefunction out(M);
    for (const auto& kv : final_acc) {
        out.add_term(kv.first, kv.second, tol_element);
    }
    return out;
}

std::vector<SlaterDeterminant> get_connected_basis(
    const Wavefunction& psi,
    const HamiltonianTables& sh)
{
    // We only need a set to store the unique determinants we find.
    using Set = std::unordered_set<SlaterDeterminant>;

    const size_t M = psi.num_spatial_orbitals();
    const size_t K = sh.n_spin_orbitals;

    // Flatten the input wavefunction for parallel processing.
    std::vector<SlaterDeterminant> basis_in;
    basis_in.reserve(psi.data().size());
    for (const auto& kv : psi.data()) {
        basis_in.push_back(kv.first);
    }

    int T = 1;
    #ifdef _OPENMP
    T = omp_get_max_threads();
    #endif
    // Each thread will have its own local set to avoid locking.
    std::vector<Set> local_sets(T);

    #pragma omp parallel for schedule(dynamic)
    for (size_t item_idx = 0; item_idx < basis_in.size(); ++item_idx) {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif
        auto& acc_set = local_sets[tid]; // Accumulate into the thread's local set
        const auto& det_in = basis_in[item_idx];

        // This is still useful to avoid inserting the same new determinant
        // multiple times if it's generated from the same det_in.
        std::unordered_set<SlaterDeterminant> seen;
        
        std::vector<int> occ_so_vec;
        occ_so_vec.reserve(K);
        for (auto it = det_in.alpha().begin_occ(); it != det_in.alpha().end_occ(); ++it) occ_so_vec.push_back(*it);
        for (auto it = det_in.beta().begin_occ(); it != det_in.beta().end_occ(); ++it) occ_so_vec.push_back(*it + static_cast<int>(M));
        std::sort(occ_so_vec.begin(), occ_so_vec.end());
        std::unordered_set<int> occ_so_set(occ_so_vec.begin(), occ_so_vec.end());
        
        SlaterDeterminant det_out_scratch(M);
        int8_t sign; 

        // 1) Diagonal: The determinant is always connected to itself.
        acc_set.insert(det_in);
        seen.insert(det_in);


        // 2) Singles from Sh0
        for (int r : occ_so_vec) {
            auto it = sh.sh0.find(r);
            if (it == sh.sh0.end()) continue;
            for (uint32_t p : it->second) {
                if (occ_so_set.count(p)) continue; 

                if (SlaterDeterminant::apply_excitation_single_fast(
                        det_in, p % M, r % M,
                        p < M ? Spin::Alpha : Spin::Beta,
                        r < M ? Spin::Alpha : Spin::Beta,
                        det_out_scratch, sign)) { 

                    if (seen.count(det_out_scratch)) continue;
                    seen.insert(det_out_scratch);
                    acc_set.insert(det_out_scratch);
                }
            }
        }

        // 3) Singles from SU
        for (int r : occ_so_vec) {
            auto it_r = sh.su.find(r);
            if (it_r == sh.su.end()) continue;
            for (const auto& [p, spectators] : it_r->second) {
                if (occ_so_set.count(p)) continue; 

                bool spectator_ok = false;
                for (uint32_t s : spectators) {
                    if (occ_so_set.count(s)) { spectator_ok = true; break; }
                }
                if (!spectator_ok) continue;

                if (SlaterDeterminant::apply_excitation_single_fast(
                        det_in, p % M, r % M,
                        p < M ? Spin::Alpha : Spin::Beta,
                        r < M ? Spin::Alpha : Spin::Beta,
                        det_out_scratch, sign)) { 
                    if (seen.count(det_out_scratch)) continue;
                    seen.insert(det_out_scratch);
                    acc_set.insert(det_out_scratch);
                }
            }
        }

        // 4) Doubles from D
        for (size_t i = 0; i < occ_so_vec.size(); ++i) {
            for (size_t j = i + 1; j < occ_so_vec.size(); ++j) {
                uint32_t r = occ_so_vec[i];
                uint32_t s = occ_so_vec[j];
                uint64_t key_rs = r | (static_cast<uint64_t>(s) << 32);

                auto it = sh.d.find(key_rs);
                if (it == sh.d.end()) continue;

                for (const auto& [p, q] : it->second) {
                    if (occ_so_set.count(p) || occ_so_set.count(q)) continue;

                    if (SlaterDeterminant::apply_excitation_double_fast(
                            det_in, p % M, q % M, r % M, s % M,
                            p < M ? Spin::Alpha : Spin::Beta, q < M ? Spin::Alpha : Spin::Beta,
                            r < M ? Spin::Alpha : Spin::Beta, s < M ? Spin::Alpha : Spin::Beta,
                            det_out_scratch, sign)) { 
                        
                        if (seen.count(det_out_scratch)) continue;
                        seen.insert(det_out_scratch);
                        acc_set.insert(det_out_scratch);
                    }
                }
            }
        }
    }

    // --- Merge results from all threads ---
    Set final_set;
    size_t hint = 0;
    for(const auto& s : local_sets) hint += s.size();
    final_set.reserve(hint);

    for (const auto& s : local_sets) {
        final_set.insert(s.begin(), s.end());
    }

    // Convert the final set to a sorted vector for a canonical representation.
    std::vector<SlaterDeterminant> connected_basis(final_set.begin(), final_set.end());
    std::sort(connected_basis.begin(), connected_basis.end());
    
    return connected_basis;
}


// ============================
// New fixed basis support
// ============================

struct AllowedSets {
    std::unordered_set<uint32_t> spatial;
    std::unordered_set<uint32_t> spinorbital;
};

static AllowedSets collect_allowed_from_basis(const std::vector<SlaterDeterminant>& basis, size_t M) {
    AllowedSets A;
    A.spatial.reserve(M);
    for (const auto& sd : basis) {
        for (auto it = sd.alpha().begin_occ(); it != sd.alpha().end_occ(); ++it) A.spatial.insert(static_cast<uint32_t>(*it));
        for (auto it = sd.beta().begin_occ();  it != sd.beta().end_occ();  ++it) A.spatial.insert(static_cast<uint32_t>(*it));
    }
    A.spinorbital.reserve(2 * A.spatial.size());
    for (uint32_t p : A.spatial) {
        A.spinorbital.insert(p);
        A.spinorbital.insert(p + static_cast<uint32_t>(M));
    }
    return A;
}

HamiltonianTables build_fixed_basis_tables(
    const HamiltonianTables& sh_full,
    const std::vector<SlaterDeterminant>& basis,
    size_t M)
{
    const auto A = collect_allowed_from_basis(basis, M);
    const uint32_t K = static_cast<uint32_t>(2 * M);

    HamiltonianTables sh_fb;
    sh_fb.n_spin_orbitals = sh_full.n_spin_orbitals;

    // sh0
    for (uint32_t r = 0; r < K; ++r) {
        if (!A.spinorbital.count(r)) continue;
        auto it = sh_full.sh0.find(r);
        if (it == sh_full.sh0.end()) continue;

        std::vector<uint32_t> filtered;
        filtered.reserve(it->second.size());
        for (uint32_t p : it->second) if (A.spinorbital.count(p)) filtered.push_back(p);
        if (!filtered.empty()) sh_fb.sh0[r] = std::move(filtered);
    }

    // su
    for (const auto& rij : sh_full.su) {
        uint32_t i = rij.first;
        if (!A.spinorbital.count(i)) continue;
        for (const auto& pj : rij.second) {
            uint32_t j = pj.first;
            if (!A.spinorbital.count(j)) continue;

            std::vector<uint32_t> spectators_fb;
            spectators_fb.reserve(pj.second.size());
            for (uint32_t s : pj.second) if (A.spinorbital.count(s)) spectators_fb.push_back(s);
            if (!spectators_fb.empty()) sh_fb.su[i][j] = std::move(spectators_fb);
        }
    }

    // d
    for (const auto& kv : sh_full.d) {
        const uint64_t key = kv.first;
        const uint32_t r = static_cast<uint32_t>(key & 0xffffffffULL);
        const uint32_t s = static_cast<uint32_t>((key >> 32) & 0xffffffffULL);
        if (!A.spinorbital.count(r) || !A.spinorbital.count(s)) continue;

        std::vector<std::pair<uint32_t,uint32_t>> filtered;
        filtered.reserve(kv.second.size());
        for (auto [p,q] : kv.second) if (A.spinorbital.count(p) && A.spinorbital.count(q)) filtered.emplace_back(p,q);
        if (!filtered.empty()) sh_fb.d[r | (static_cast<uint64_t>(s) << 32)] = std::move(filtered);
    }

    return sh_fb;
}

Wavefunction apply_hamiltonian_fixed_basis(
    const Wavefunction& psi,
    const HamiltonianTables& sh_fixed_basis,
    const std::vector<SlaterDeterminant>& basis,
    const H1View& H,
    const ERIView& V,
    double tol_element)
{
    using Coeff = Wavefunction::Coeff;
    using Map   = std::unordered_map<SlaterDeterminant, Coeff>;

    const size_t M = psi.num_spatial_orbitals();
    const size_t K = sh_fixed_basis.n_spin_orbitals;

    std::unordered_set<SlaterDeterminant> basis_set;
    basis_set.reserve(basis.size() * 2);
    for (const auto& d : basis) basis_set.insert(d);

    std::vector<std::pair<SlaterDeterminant, Coeff>> items;
    items.reserve(psi.data().size());
    for (const auto& kv : psi.data()) if (basis_set.count(kv.first)) items.emplace_back(kv.first, kv.second);

    int T = 1;
    #ifdef _OPENMP
    T = omp_get_max_threads();
    #endif
    std::vector<Map> local_maps(T);

    #pragma omp parallel for schedule(dynamic)
    for (size_t idx = 0; idx < items.size(); ++idx) {
        int tid = 0;
        #ifdef _OPENMP
        tid = omp_get_thread_num();
        #endif
        auto& acc = local_maps[tid];

        const auto& det_in  = items[idx].first;
        const auto& coeff   = items[idx].second;

        std::unordered_set<SlaterDeterminant> seen;
        seen.reserve(64);

        std::vector<int> occ_so_vec;
        occ_so_vec.reserve(K);
        for (auto it = det_in.alpha().begin_occ(); it != det_in.alpha().end_occ(); ++it) occ_so_vec.push_back(*it);
        for (auto it = det_in.beta().begin_occ(); it != det_in.beta().end_occ();  ++it) occ_so_vec.push_back(*it + static_cast<int>(M));
        std::sort(occ_so_vec.begin(), occ_so_vec.end());
        std::unordered_set<int> occ_so_set(occ_so_vec.begin(), occ_so_vec.end());

        Determinant d_in_comb(K, occ_so_vec);
        SlaterDeterminant det_out_scratch(M);
        int8_t sign = 1;

        // diagonal
        cx val_diag = KL(d_in_comb, d_in_comb, H, V);
        if (std::abs(val_diag) > 0) acc[det_in] += coeff * val_diag;
        seen.insert(det_in);

        // singles from sh0
        for (int r : occ_so_vec) {
            auto it = sh_fixed_basis.sh0.find(static_cast<uint32_t>(r));
            if (it == sh_fixed_basis.sh0.end()) continue;

            for (uint32_t p : it->second) {
                if (occ_so_set.count(static_cast<int>(p))) continue;

                if (SlaterDeterminant::apply_excitation_single_fast(
                        det_in, p % M, r % M,
                        p < M ? Spin::Alpha : Spin::Beta,
                        r < M ? Spin::Alpha : Spin::Beta,
                        det_out_scratch, sign)) {

                    if (!basis_set.count(det_out_scratch)) continue;
                    if (seen.count(det_out_scratch)) continue;
                    seen.insert(det_out_scratch);

                    Determinant d_out_comb = combined_from(det_out_scratch);
                    cx val = KL(d_in_comb, d_out_comb, H, V);
                    if (std::abs(val) > 0) acc[det_out_scratch] += coeff * val;
                }
            }
        }

        // singles from su
        for (int r : occ_so_vec) {
            auto it_r = sh_fixed_basis.su.find(static_cast<uint32_t>(r));
            if (it_r == sh_fixed_basis.su.end()) continue;

            for (const auto& [p, spectators] : it_r->second) {
                if (occ_so_set.count(static_cast<int>(p))) continue;

                bool ok = false;
                for (uint32_t s : spectators) {
                    if (occ_so_set.count(static_cast<int>(s))) { ok = true; break; }
                }
                if (!ok) continue;

                if (SlaterDeterminant::apply_excitation_single_fast(
                        det_in, p % M, r % M,
                        p < M ? Spin::Alpha : Spin::Beta,
                        r < M ? Spin::Alpha : Spin::Beta,
                        det_out_scratch, sign)) {

                    if (!basis_set.count(det_out_scratch)) continue;
                    if (seen.count(det_out_scratch)) continue;
                    seen.insert(det_out_scratch);

                    Determinant d_out_comb = combined_from(det_out_scratch);
                    cx val = KL(d_in_comb, d_out_comb, H, V);
                    if (std::abs(val) > 0) acc[det_out_scratch] += coeff * val;
                }
            }
        }

        // doubles from d
        for (size_t a = 0; a < occ_so_vec.size(); ++a) {
            for (size_t b = a + 1; b < occ_so_vec.size(); ++b) {
                uint32_t r = static_cast<uint32_t>(occ_so_vec[a]);
                uint32_t s = static_cast<uint32_t>(occ_so_vec[b]);
                uint64_t key_rs = r | (static_cast<uint64_t>(s) << 32);

                auto it = sh_fixed_basis.d.find(key_rs);
                if (it == sh_fixed_basis.d.end()) continue;

                for (const auto& [p, q] : it->second) {
                    if (occ_so_set.count(static_cast<int>(p)) || occ_so_set.count(static_cast<int>(q))) continue;

                    if (SlaterDeterminant::apply_excitation_double_fast(
                            det_in, p % M, q % M, r % M, s % M,
                            p < M ? Spin::Alpha : Spin::Beta, q < M ? Spin::Alpha : Spin::Beta,
                            r < M ? Spin::Alpha : Spin::Beta, s < M ? Spin::Alpha : Spin::Beta,
                            det_out_scratch, sign)) {

                        if (!basis_set.count(det_out_scratch)) continue;
                        if (seen.count(det_out_scratch)) continue;
                        seen.insert(det_out_scratch);

                        Determinant d_out_comb = combined_from(det_out_scratch);
                        cx val = KL(d_in_comb, d_out_comb, H, V);
                        if (std::abs(val) > 0) acc[det_out_scratch] += coeff * val;
                    }
                }
            }
        }
    }

    // merge
    std::unordered_map<SlaterDeterminant, Coeff> final_acc;
    size_t hint = 0;
    for (const auto& m : local_maps) hint += m.size();
    final_acc.reserve(hint);
    for (const auto& m : local_maps)
        for (const auto& kv : m) final_acc[kv.first] += kv.second;

    Wavefunction out(M);
    for (const auto& kv : final_acc) out.add_term(kv.first, kv.second, tol_element);
    return out;
}


namespace {

// Local COO helper, same layout as in build_hamiltonian_openmp
struct COO {
    std::vector<int32_t> row;
    std::vector<int32_t> col;
    std::vector<cx>      val;

    void push(int32_t r, int32_t c, cx v, double tol) {
        if (std::abs(v) < tol) return;
        row.push_back(r);
        col.push_back(c);
        val.push_back(v);
    }
};

static CSR coo_to_csr(std::size_t N, std::size_t M, COO&& coo) {
    CSR out;
    out.n_rows = N;
    out.n_cols = M;
    out.indptr.assign(N + 1, 0);

    const std::size_t nnz = coo.val.size();
    // count per row
    for (std::size_t e = 0; e < nnz; ++e) {
        ++out.indptr[static_cast<std::size_t>(coo.row[e]) + 1];
    }
    // prefix sum
    for (std::size_t i = 0; i < N; ++i) out.indptr[i + 1] += out.indptr[i];

    out.indices.resize(nnz);
    out.data.resize(nnz);

    // cursors
    std::vector<int64_t> cursor = out.indptr;

    for (std::size_t e = 0; e < nnz; ++e) {
        const int32_t r  = coo.row[e];
        const int64_t pos = cursor[static_cast<std::size_t>(r)]++;
        out.indices[static_cast<std::size_t>(pos)] = coo.col[e];
        out.data   [static_cast<std::size_t>(pos)] = coo.val[e];
    }

    // no sorting/summing; we generate each (i,j) once

    return out;
}

} // anonymous namespace

CSR build_hamiltonian_matrix_fixed_basis(
    const HamiltonianTables& sh_fixed_basis,
    const std::vector<SlaterDeterminant>& basis,
    const H1View& H,
    const ERIView& V,
    double tol_element)
{
    const std::size_t dim = basis.size();
    if (dim == 0) {
        COO empty;
        return coo_to_csr(0, 0, std::move(empty));
    }

    const std::size_t M = basis.front().num_spatial_orbitals();
    const std::size_t K = sh_fixed_basis.n_spin_orbitals;

    // determinant -> basis index
    std::unordered_map<SlaterDeterminant, std::size_t> index_map;
    index_map.reserve(dim * 2);
    for (std::size_t i = 0; i < dim; ++i) {
        index_map.emplace(basis[i], i);
    }

    int T = 1;
#ifdef _OPENMP
    T = omp_get_max_threads();
#endif
    std::vector<COO> local_coo(T);

#pragma omp parallel for schedule(dynamic)
    for (std::int64_t i64 = 0; i64 < static_cast<std::int64_t>(dim); ++i64) {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        COO& coo = local_coo[tid];

        const std::size_t i = static_cast<std::size_t>(i64);
        const SlaterDeterminant& det_in = basis[i];

        // spin-orbital occupations
        std::vector<int> occ_so_vec;
        occ_so_vec.reserve(K);
        for (auto it = det_in.alpha().begin_occ(); it != det_in.alpha().end_occ(); ++it)
            occ_so_vec.push_back(*it);
        for (auto it = det_in.beta().begin_occ(); it != det_in.beta().end_occ(); ++it)
            occ_so_vec.push_back(*it + static_cast<int>(M));
        std::sort(occ_so_vec.begin(), occ_so_vec.end());
        std::unordered_set<int> occ_so_set(occ_so_vec.begin(), occ_so_vec.end());

        Determinant d_in_comb(K, occ_so_vec);
        SlaterDeterminant det_out_scratch(M);
        int8_t sign = 1;

        // Avoid generating the same det_out multiple times from this det_in
        std::unordered_set<SlaterDeterminant> seen;
        seen.reserve(64);
        seen.insert(det_in);

        // 1) diagonal
        {
            cx val_diag = KL(d_in_comb, d_in_comb, H, V);
            if (std::abs(val_diag) >= tol_element) {
                coo.push(static_cast<int32_t>(i),
                         static_cast<int32_t>(i),
                         val_diag, 0.0); // already thresholded
            }
        }

        // 2) singles from sh0
        for (int r : occ_so_vec) {
            auto it = sh_fixed_basis.sh0.find(static_cast<uint32_t>(r));
            if (it == sh_fixed_basis.sh0.end()) continue;

            for (uint32_t p : it->second) {
                if (occ_so_set.count(static_cast<int>(p))) continue;

                if (SlaterDeterminant::apply_excitation_single_fast(
                        det_in, p % M, r % M,
                        p < M ? Spin::Alpha : Spin::Beta,
                        r < M ? Spin::Alpha : Spin::Beta,
                        det_out_scratch, sign))
                {
                    auto it_idx = index_map.find(det_out_scratch);
                    if (it_idx == index_map.end()) continue;
                    const std::size_t j = it_idx->second;
                    if (j < i) continue; // generate only upper triangle

                    if (seen.count(det_out_scratch)) continue;
                    seen.insert(det_out_scratch);

                    Determinant d_out_comb = combined_from(det_out_scratch);
                    cx val = KL(d_in_comb, d_out_comb, H, V);
                    if (std::abs(val) < tol_element) continue;

                    coo.push(static_cast<int32_t>(i),
                             static_cast<int32_t>(j),
                             val, 0.0);
                    if (j != i) {
                        coo.push(static_cast<int32_t>(j),
                                 static_cast<int32_t>(i),
                                 std::conj(val), 0.0);
                    }
                }
            }
        }

        // 3) singles from su
        for (int r : occ_so_vec) {
            auto it_r = sh_fixed_basis.su.find(static_cast<uint32_t>(r));
            if (it_r == sh_fixed_basis.su.end()) continue;

            for (const auto& [p, spectators] : it_r->second) {
                if (occ_so_set.count(static_cast<int>(p))) continue;

                bool ok = false;
                for (uint32_t s : spectators) {
                    if (occ_so_set.count(static_cast<int>(s))) { ok = true; break; }
                }
                if (!ok) continue;

                if (SlaterDeterminant::apply_excitation_single_fast(
                        det_in, p % M, r % M,
                        p < M ? Spin::Alpha : Spin::Beta,
                        r < M ? Spin::Alpha : Spin::Beta,
                        det_out_scratch, sign))
                {
                    auto it_idx = index_map.find(det_out_scratch);
                    if (it_idx == index_map.end()) continue;
                    const std::size_t j = it_idx->second;
                    if (j < i) continue;

                    if (seen.count(det_out_scratch)) continue;
                    seen.insert(det_out_scratch);

                    Determinant d_out_comb = combined_from(det_out_scratch);
                    cx val = KL(d_in_comb, d_out_comb, H, V);
                    if (std::abs(val) < tol_element) continue;

                    coo.push(static_cast<int32_t>(i),
                             static_cast<int32_t>(j),
                             val, 0.0);
                    if (j != i) {
                        coo.push(static_cast<int32_t>(j),
                                 static_cast<int32_t>(i),
                                 std::conj(val), 0.0);
                    }
                }
            }
        }

        // 4) doubles from d
        for (std::size_t a = 0; a < occ_so_vec.size(); ++a) {
            for (std::size_t b = a + 1; b < occ_so_vec.size(); ++b) {
                uint32_t r = static_cast<uint32_t>(occ_so_vec[a]);
                uint32_t s = static_cast<uint32_t>(occ_so_vec[b]);
                uint64_t key_rs = r | (static_cast<uint64_t>(s) << 32);

                auto it = sh_fixed_basis.d.find(key_rs);
                if (it == sh_fixed_basis.d.end()) continue;

                for (const auto& [p, q] : it->second) {
                    if (occ_so_set.count(static_cast<int>(p)) ||
                        occ_so_set.count(static_cast<int>(q))) continue;

                    if (SlaterDeterminant::apply_excitation_double_fast(
                            det_in, p % M, q % M, r % M, s % M,
                            p < M ? Spin::Alpha : Spin::Beta,
                            q < M ? Spin::Alpha : Spin::Beta,
                            r < M ? Spin::Alpha : Spin::Beta,
                            s < M ? Spin::Alpha : Spin::Beta,
                            det_out_scratch, sign))
                    {
                        auto it_idx = index_map.find(det_out_scratch);
                        if (it_idx == index_map.end()) continue;
                        const std::size_t j = it_idx->second;
                        if (j < i) continue;

                        if (seen.count(det_out_scratch)) continue;
                        seen.insert(det_out_scratch);

                        Determinant d_out_comb = combined_from(det_out_scratch);
                        cx val = KL(d_in_comb, d_out_comb, H, V);
                        if (std::abs(val) < tol_element) continue;

                        coo.push(static_cast<int32_t>(i),
                                 static_cast<int32_t>(j),
                                 val, 0.0);
                        if (j != i) {
                            coo.push(static_cast<int32_t>(j),
                                     static_cast<int32_t>(i),
                                     std::conj(val), 0.0);
                        }
                    }
                }
            }
        }
    } // omp parallel for

    // merge COOs
    COO global;
    std::size_t total_nnz = 0;
    for (const auto& c : local_coo) total_nnz += c.val.size();
    global.row.reserve(total_nnz);
    global.col.reserve(total_nnz);
    global.val.reserve(total_nnz);

    for (auto& c : local_coo) {
        global.row.insert(global.row.end(),
                          std::make_move_iterator(c.row.begin()),
                          std::make_move_iterator(c.row.end()));
        global.col.insert(global.col.end(),
                          std::make_move_iterator(c.col.begin()),
                          std::make_move_iterator(c.col.end()));
        global.val.insert(global.val.end(),
                          std::make_move_iterator(c.val.begin()),
                          std::make_move_iterator(c.val.end()));
    }

    return coo_to_csr(dim, dim, std::move(global));
}



} // namespace ci