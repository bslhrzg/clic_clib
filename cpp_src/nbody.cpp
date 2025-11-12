// nbody.cpp
#include "nbody.h"

#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace ci {

// ====================== Wavefunction Operator Application =======================

// --- Helper for bucketing ---
// Key for one-body term: packs (j, spin_j)
inline uint64_t make_key1(size_t j, Spin spin_j) {
    return (static_cast<uint64_t>(j) << 1) | static_cast<uint64_t>(spin_j);
}

// Key for two-body term: packs (k, l, spin_k, spin_l)
inline uint64_t make_key2(size_t k, size_t l, Spin spin_k, Spin spin_l) {
    // Ensure canonical order (k, sk) <= (l, sl) to halve bucket search space for some cases
    size_t p1 = k; Spin s1 = spin_k;
    size_t p2 = l; Spin s2 = spin_l;
    if (p1 > p2 || (p1 == p2 && static_cast<uint8_t>(s1) > static_cast<uint8_t>(s2))) {
        std::swap(p1, p2);
        std::swap(s1, s2);
    }
    return (static_cast<uint64_t>(p1) << 32) |
           (static_cast<uint64_t>(p2) << 2)  |
           (static_cast<uint64_t>(s1) << 1)  |
           static_cast<uint64_t>(s2);
}

// NOTE: The following four functions with trailing underscores are the original
// serial implementations from your source file. They are included here as they
// were part of the provided code block, but are not exposed in the header.
Wavefunction apply_one_body_operator_(const Wavefunction& wf,
                                     const std::vector<OneBodyTerm>& terms)
{
    Wavefunction out(wf.num_spatial_orbitals());
    if (wf.data().empty()) return out;

    // 1. Build buckets of terms based on annihilated orbital (j, spin_j)
    std::unordered_map<uint64_t, std::vector<const OneBodyTerm*>> buckets;
    for (const auto& term : terms) {
        if (std::abs(term.value) > 0.0) { // Simple screening
            buckets[make_key1(term.j, term.spin_j)].push_back(&term);
        }
    }

    // 2. Iterate through determinants in the wavefunction
    SlaterDeterminant excited_det(wf.num_spatial_orbitals());
    int8_t sign;

    for (const auto& [det, coeff] : wf.data()) {
        // 3. For each determinant, check its occupied alpha orbitals
        for (auto it = det.alpha().begin_occ(); it != det.alpha().end_occ(); ++it) {
            const size_t j = static_cast<size_t>(*it);
            auto bucket_it = buckets.find(make_key1(j, Spin::Alpha));
            if (bucket_it != buckets.end()) {
                // 4. If a bucket exists, apply all excitations from it
                for (const auto* term : bucket_it->second) {
                    if (SlaterDeterminant::apply_excitation_single_fast(det, term->i, term->j, term->spin_i, term->spin_j, excited_det, sign)) {
                        out.add_term(excited_det, coeff * term->value * static_cast<double>(sign));
                    }
                }
            }
        }
        // 5. Repeat for beta orbitals
        for (auto it = det.beta().begin_occ(); it != det.beta().end_occ(); ++it) {
            const size_t j = static_cast<size_t>(*it);
            auto bucket_it = buckets.find(make_key1(j, Spin::Beta));
            if (bucket_it != buckets.end()) {
                for (const auto* term : bucket_it->second) {
                    if (SlaterDeterminant::apply_excitation_single_fast(det, term->i, term->j, term->spin_i, term->spin_j, excited_det, sign)) {
                        out.add_term(excited_det, coeff * term->value * static_cast<double>(sign));
                    }
                }
            }
        }
    }
    return out;
}

Wavefunction apply_two_body_operator_(const Wavefunction& wf,
                                     const std::vector<TwoBodyTerm>& terms)
{
    Wavefunction out(wf.num_spatial_orbitals());
    if (wf.data().empty()) return out;

    // 1. Build buckets for two-body terms
    std::unordered_map<uint64_t, std::vector<const TwoBodyTerm*>> buckets;
    for (const auto& term : terms) {
        if (std::abs(term.value) > 0.0) {
            buckets[make_key2(term.k, term.l, term.spin_k, term.spin_l)].push_back(&term);
        }
    }

    // 2. Iterate through determinants
    SlaterDeterminant excited_det(wf.num_spatial_orbitals());
    int8_t sign;
    std::vector<std::pair<size_t, Spin>> occupied_orbs;

    for (const auto& [det, coeff] : wf.data()) {
        // Collect all occupied spin-orbitals for easier iteration over pairs
        occupied_orbs.clear();
        for (auto it = det.alpha().begin_occ(); it != det.alpha().end_occ(); ++it) occupied_orbs.emplace_back(static_cast<size_t>(*it), Spin::Alpha);
        for (auto it = det.beta().begin_occ();  it != det.beta().end_occ();  ++it) occupied_orbs.emplace_back(static_cast<size_t>(*it), Spin::Beta);

        // 3. Iterate over all PAIRS of occupied orbitals (k, sk) and (l, sl)
        for (size_t idx1 = 0; idx1 < occupied_orbs.size(); ++idx1) {
            for (size_t idx2 = idx1 + 1; idx2 < occupied_orbs.size(); ++idx2) {
                const auto& [k, sk] = occupied_orbs[idx1];
                const auto& [l, sl] = occupied_orbs[idx2];

                auto bucket_it = buckets.find(make_key2(k, l, sk, sl));
                if (bucket_it != buckets.end()) {
                    for (const auto* term : bucket_it->second) {
                        // The key is canonical, but the term may not be. Check both permutations.
                        if ((term->k == k && term->l == l && term->spin_k == sk && term->spin_l == sl) ||
                            (term->k == l && term->l == k && term->spin_k == sl && term->spin_l == sk)) {
                            if (SlaterDeterminant::apply_excitation_double_fast(det, term->i, term->j, term->k, term->l, term->spin_i, term->spin_j, term->spin_k, term->spin_l, excited_det, sign)) {
                                out.add_term(excited_det, coeff * term->value * static_cast<double>(sign));
                            }
                        }
                    }
                }
            }
        }
    }
    return out;
}


// Hash & Eq already defined for SlaterDeterminant in your code
using Coeff = Wavefunction::Coeff;
using Map   = std::unordered_map<SlaterDeterminant, Coeff>;

static inline std::size_t shard_of(const SlaterDeterminant& d, std::size_t S) {
    std::size_t h = std::hash<SlaterDeterminant>{}(d);
    return h & (S - 1); // S must be power of two
}

Wavefunction apply_one_body_operator(const Wavefunction& wf,
                                     const std::vector<OneBodyTerm>& terms)
{
    Wavefunction out(wf.num_spatial_orbitals());
    if (wf.data().empty()) return out;
    const std::size_t M = wf.num_spatial_orbitals();

    // Buckets (read-only)
    std::unordered_map<uint64_t, std::vector<const OneBodyTerm*>> buckets;
    buckets.reserve(terms.size());
    for (const auto& t : terms) if (std::abs(t.value) > 0.0)
        buckets[make_key1(t.j, t.spin_j)].push_back(&t);

    // Flatten wf for parallel iteration
    std::vector<std::pair<SlaterDeterminant, Coeff>> items;
    items.reserve(wf.data().size());
    for (const auto& kv : wf.data()) items.emplace_back(kv.first, kv.second);

    // Local per-thread maps (sums duplicates early)
    int T = 1;
#ifdef _OPENMP
    T = std::max(1, omp_get_max_threads());
#endif
    std::vector<Map> local_maps(T);

    // Generate + accumulate locally
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        auto& acc = local_maps[tid];
        acc.reserve(std::max<std::size_t>(512, items.size()/T));

        SlaterDeterminant excited(M);
        int8_t sign;

#ifdef _OPENMP
        #pragma omp for schedule(static)
#endif
        for (std::ptrdiff_t idx = 0; idx < (std::ptrdiff_t)items.size(); ++idx) {
            const auto& det   = items[(std::size_t)idx].first;
            const auto& coeff = items[(std::size_t)idx].second;

            // α
            for (auto it = det.alpha().begin_occ(); it != det.alpha().end_occ(); ++it) {
                const size_t j = (size_t)*it;
                auto bi = buckets.find(make_key1(j, Spin::Alpha));
                if (bi == buckets.end()) continue;
                for (const auto* t : bi->second) {
                    if (SlaterDeterminant::apply_excitation_single_fast(det, t->i, t->j, t->spin_i, t->spin_j,
                                                                        excited, sign)) {
                        acc[excited] += coeff * t->value * double(sign);
                    }
                }
            }
            // β
            for (auto it = det.beta().begin_occ(); it != det.beta().end_occ(); ++it) {
                const size_t j = (size_t)*it;
                auto bi = buckets.find(make_key1(j, Spin::Beta));
                if (bi == buckets.end()) continue;
                for (const auto* t : bi->second) {
                    if (SlaterDeterminant::apply_excitation_single_fast(det, t->i, t->j, t->spin_i, t->spin_j,
                                                                        excited, sign)) {
                        acc[excited] += coeff * t->value * double(sign);
                    }
                }
            }
        }
    }

    // Sharded global reduce to avoid a single hot hash map
    // Choose power-of-two shard count (e.g. next power of two >= 2*T)
    std::size_t S = 1;
    while (S < (std::size_t)(2 * T)) S <<= 1;
    std::vector<Map> shards(S);

#ifdef _OPENMP
    #pragma omp parallel for schedule(guided)
#endif
    for (int t = 0; t < T; ++t) {
        for (const auto& kv : local_maps[t]) {
            const auto& det = kv.first;
            auto s = shard_of(det, S);
            // To avoid locks, move per-thread into a thread-local buffer first,
            // then we do a second parallel pass to combine shard-by-shard:
#ifdef _OPENMP
            #pragma omp critical  // cheap: S is large; contention low
#endif
            shards[s][det] += kv.second;
        }
    }

    // Finalize Wavefunction
    for (auto& sh : shards)
        for (auto& kv : sh)
            out.add_term(kv.first, kv.second);

    return out;
}



Wavefunction apply_two_body_operator(const Wavefunction& wf,
                                     const std::vector<TwoBodyTerm>& terms)
{
    using Coeff = Wavefunction::Coeff;

    Wavefunction out(wf.num_spatial_orbitals());
    if (wf.data().empty() || terms.empty()) return out;
    const std::size_t M = wf.num_spatial_orbitals();

    // 1) Read-only buckets keyed by canonical annihilated pair (k,sk),(l,sl)
    std::unordered_map<uint64_t, std::vector<const TwoBodyTerm*>> buckets;
    buckets.reserve(terms.size());
    for (const auto& term : terms) {
        if (std::abs(term.value) > 0.0) {
            buckets[make_key2(term.k, term.l, term.spin_k, term.spin_l)].push_back(&term);
        }
    }

    // 2) Flatten wf for parallel iteration
    std::vector<std::pair<SlaterDeterminant, Coeff>> items;
    items.reserve(wf.data().size());
    for (const auto& kv : wf.data()) items.emplace_back(kv.first, kv.second);

    // 3) Per-thread accumulation maps: SlaterDeterminant -> summed Coeff
    using Map = std::unordered_map<SlaterDeterminant, Coeff>;
    int T = 1;
#ifdef _OPENMP
    T = std::max(1, omp_get_max_threads());
#endif
    std::vector<Map> local_maps((std::size_t)T);

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        int tid = 0;
#ifdef _OPENMP
        tid = omp_get_thread_num();
#endif
        auto& acc = local_maps[(std::size_t)tid];
        acc.reserve(std::max<std::size_t>(512, items.size() / std::max(1, T)));

        SlaterDeterminant excited_det(M);
        int8_t sign;

        std::vector<std::pair<size_t, Spin>> occ;
        occ.reserve(2 * M);

#ifdef _OPENMP
        #pragma omp for schedule(guided)
#endif
        for (std::ptrdiff_t idx = 0; idx < static_cast<std::ptrdiff_t>(items.size()); ++idx) {
            const auto& det   = items[(std::size_t)idx].first;
            const auto& coeff = items[(std::size_t)idx].second;

            // Collect occupied spin-orbitals
            occ.clear();
            for (auto it = det.alpha().begin_occ(); it != det.alpha().end_occ(); ++it)
                occ.emplace_back(static_cast<size_t>(*it), Spin::Alpha);
            for (auto it = det.beta().begin_occ();  it != det.beta().end_occ();  ++it)
                occ.emplace_back(static_cast<size_t>(*it), Spin::Beta);

            const std::size_t L = occ.size();
            for (std::size_t a = 0; a + 1 < L; ++a) {
                const auto [k, sk] = occ[a];
                for (std::size_t b = a + 1; b < L; ++b) {
                    const auto [l, sl] = occ[b];

                    auto itB = buckets.find(make_key2(k, l, sk, sl));
                    if (itB == buckets.end()) continue;

                    // Apply all matching two-body terms from the bucket
                    for (const auto* term : itB->second) {
                        const bool match_kl = (term->k == k && term->l == l &&
                                               term->spin_k == sk && term->spin_l == sl);
                        const bool match_lk = (term->k == l && term->l == k &&
                                               term->spin_k == sl && term->spin_l == sk);
                        if (!(match_kl || match_lk)) continue;

                        if (SlaterDeterminant::apply_excitation_double_fast(
                                det, term->i, term->j, term->k, term->l,
                                term->spin_i, term->spin_j, term->spin_k, term->spin_l,
                                excited_det, sign))
                        {
                            acc[excited_det] += coeff * term->value * static_cast<double>(sign);
                        }
                    }
                }
            }
        } // omp for
    } // omp parallel

    // 4) Lock-free sharded merge of the local maps into S shards
    auto shard_of = [](const SlaterDeterminant& d, std::size_t S)->std::size_t {
        std::size_t h = std::hash<SlaterDeterminant>{}(d);
        return h & (S - 1); // S must be a power of two
    };

    std::size_t S = 1;
    while (S < (std::size_t)(2 * T)) S <<= 1; // at least 2*T shards
    std::vector<Map> shards(S);

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int s = 0; s < (int)S; ++s) {
        auto& dst = shards[(std::size_t)s];
        // Rough capacity hint: distribute keys evenly
        std::size_t hint = 0;
        for (const auto& lm : local_maps) hint += lm.size();
        hint = hint / S + 8;
        dst.reserve(hint);

        for (const auto& lm : local_maps) {
            for (const auto& kv : lm) {
                if ((int)shard_of(kv.first, S) == s) {
                    dst[kv.first] += kv.second;
                }
            }
        }
    }

    // 5) Emit to output wavefunction
    for (auto& sh : shards)
        for (auto& kv : sh)
            out.add_term(kv.first, kv.second);

    return out;
}

// ============================ Connectivity Generation ===========================

std::vector<SlaterDeterminant>
get_connections_one_body_(const std::vector<SlaterDeterminant>& basis,
                         const std::vector<OneBodyTerm>& terms)
{
    std::unordered_set<SlaterDeterminant> unique_dets;
    if (basis.empty()) return {};

    const size_t n_spatial = basis[0].num_spatial_orbitals();
    std::unordered_map<uint64_t, std::vector<const OneBodyTerm*>> buckets;
    for (const auto& term : terms) {
        if (std::abs(term.value) > 0.0) {
            buckets[make_key1(term.j, term.spin_j)].push_back(&term);
        }
    }

    SlaterDeterminant excited_det(n_spatial);
    int8_t sign;

    for (const auto& det : basis) {
        // Alpha
        for (auto it = det.alpha().begin_occ(); it != det.alpha().end_occ(); ++it) {
            const size_t j = static_cast<size_t>(*it);
            auto bucket_it = buckets.find(make_key1(j, Spin::Alpha));
            if (bucket_it != buckets.end()) {
                for (const auto* term : bucket_it->second) {
                    if (SlaterDeterminant::apply_excitation_single_fast(det, term->i, term->j, term->spin_i, term->spin_j, excited_det, sign)) {
                        unique_dets.insert(excited_det);
                    }
                }
            }
        }
        // Beta
        for (auto it = det.beta().begin_occ(); it != det.beta().end_occ(); ++it) {
            const size_t j = static_cast<size_t>(*it);
            auto bucket_it = buckets.find(make_key1(j, Spin::Beta));
            if (bucket_it != buckets.end()) {
                for (const auto* term : bucket_it->second) {
                    if (SlaterDeterminant::apply_excitation_single_fast(det, term->i, term->j, term->spin_i, term->spin_j, excited_det, sign)) {
                        unique_dets.insert(excited_det);
                    }
                }
            }
        }
    }

    std::vector<SlaterDeterminant> connected_dets(unique_dets.begin(), unique_dets.end());
    std::sort(connected_dets.begin(), connected_dets.end());
    return connected_dets;
}

std::vector<SlaterDeterminant>
get_connections_two_body_(const std::vector<SlaterDeterminant>& basis,
                         const std::vector<TwoBodyTerm>& terms)
{
    // 1. Handle edge cases
    if (basis.empty() || terms.empty()) {
        return {};
    }

    // 2. Use a hash set for efficient collection of unique determinants
    std::unordered_set<SlaterDeterminant> unique_dets;
    const size_t n_spatial = basis[0].num_spatial_orbitals();

    // 3. Build buckets of terms based on the CANONICAL annihilated pair (k, sk), (l, sl)
    //    This drastically prunes the search space.
    std::unordered_map<uint64_t, std::vector<const TwoBodyTerm*>> buckets;
    for (const auto& term : terms) {
        if (std::abs(term.value) > 0.0) {
            buckets[make_key2(term.k, term.l, term.spin_k, term.spin_l)].push_back(&term);
        }
    }

    // 4. Pre-allocate objects to avoid re-allocation inside the tight loops
    SlaterDeterminant excited_det(n_spatial);
    int8_t sign;
    std::vector<std::pair<size_t, Spin>> occupied_orbs;
    occupied_orbs.reserve(n_spatial * 2);

    // 5. Iterate through each determinant in the starting basis
    for (const auto& det : basis) {
        // Create a flat list of occupied spin-orbitals for easy iteration over pairs
        occupied_orbs.clear();
        for (auto it = det.alpha().begin_occ(); it != det.alpha().end_occ(); ++it) {
            occupied_orbs.emplace_back(static_cast<size_t>(*it), Spin::Alpha);
        }
        for (auto it = det.beta().begin_occ(); it != det.beta().end_occ(); ++it) {
            occupied_orbs.emplace_back(static_cast<size_t>(*it), Spin::Beta);
        }

        // 6. Iterate over all unique PAIRS of occupied orbitals { (k, sk), (l, sl) }
        for (size_t idx1 = 0; idx1 < occupied_orbs.size(); ++idx1) {
            for (size_t idx2 = idx1 + 1; idx2 < occupied_orbs.size(); ++idx2) {
                const auto& [k, sk] = occupied_orbs[idx1];
                const auto& [l, sl] = occupied_orbs[idx2];

                // 7. Find the corresponding bucket for this annihilated pair
                auto bucket_it = buckets.find(make_key2(k, l, sk, sl));
                if (bucket_it != buckets.end()) {

                    // 8. Apply all operator terms from the bucket
                    for (const auto* term : bucket_it->second) {
                        // The bucket key is canonical, but the term itself may be stored as v_ijkl or v_ijlk.
                        // We must ensure the term we apply matches the orbitals we're annihilating.
                        bool term_matches_kl = (term->k == k && term->l == l && term->spin_k == sk && term->spin_l == sl);
                        bool term_matches_lk = (term->k == l && term->l == k && term->spin_k == sl && term->spin_l == sk);

                        if (term_matches_kl || term_matches_lk) {
                            // Apply the excitation operator c†_i c†_j c_l c_k
                            if (SlaterDeterminant::apply_excitation_double_fast(det,
                                                             term->i, term->j, term->k, term->l,
                                                             term->spin_i, term->spin_j, term->spin_k, term->spin_l,
                                                             excited_det, sign))
                            {
                                // If successful, add the new determinant to our set
                                unique_dets.insert(excited_det);
                            }
                        }
                    }
                }
            }
        }
    }

    // 9. Convert the set to a sorted vector and return
    std::vector<SlaterDeterminant> connected_dets(unique_dets.begin(), unique_dets.end());
    std::sort(connected_dets.begin(), connected_dets.end());
    return connected_dets;
}

std::vector<SlaterDeterminant>
get_connections_one_body(const std::vector<SlaterDeterminant>& basis,
                         const std::vector<OneBodyTerm>& terms)
{
    if (basis.empty()) return {};

    const size_t n_spatial = basis[0].num_spatial_orbitals();

    // Build read-only buckets once
    std::unordered_map<uint64_t, std::vector<const OneBodyTerm*>> buckets;
    buckets.reserve(terms.size());
    for (const auto& term : terms) {
        if (std::abs(term.value) > 0.0) {
            buckets[make_key1(term.j, term.spin_j)].push_back(&term);
        }
    }

    // Thread-local accumulation of unique dets; gathered after parallel region
    std::vector<std::unordered_set<SlaterDeterminant>> partials;
    partials.reserve(omp_get_max_threads());

    #pragma omp parallel
    {
        std::unordered_set<SlaterDeterminant> local;
        // Rough reserve to reduce rehashing
        local.reserve(std::max<size_t>(16, basis.size() / std::max(1, omp_get_num_threads())));

        SlaterDeterminant excited_det(n_spatial);
        int8_t sign;

        #pragma omp for schedule(guided)
        for (std::ptrdiff_t idx = 0; idx < static_cast<std::ptrdiff_t>(basis.size()); ++idx) {
            const auto& det = basis[static_cast<size_t>(idx)];

            // Alpha occupied orbitals
            for (auto it = det.alpha().begin_occ(); it != det.alpha().end_occ(); ++it) {
                const size_t j = static_cast<size_t>(*it);
                auto key = make_key1(j, Spin::Alpha);
                auto b = buckets.find(key);
                if (b != buckets.end()) {
                    for (const auto* term : b->second) {
                        if (SlaterDeterminant::apply_excitation_single_fast(
                                det, term->i, term->j, term->spin_i, term->spin_j,
                                excited_det, sign))
                        {
                            local.insert(excited_det);
                        }
                    }
                }
            }
            // Beta occupied orbitals
            for (auto it = det.beta().begin_occ(); it != det.beta().end_occ(); ++it) {
                const size_t j = static_cast<size_t>(*it);
                auto key = make_key1(j, Spin::Beta);
                auto b = buckets.find(key);
                if (b != buckets.end()) {
                    for (const auto* term : b->second) {
                        if (SlaterDeterminant::apply_excitation_single_fast(
                                det, term->i, term->j, term->spin_i, term->spin_j,
                                excited_det, sign))
                        {
                            local.insert(excited_det);
                        }
                    }
                }
            }
        }

        #pragma omp critical
        {
            partials.push_back(std::move(local));
        }
    }

    // Merge thread-local sets
    std::unordered_set<SlaterDeterminant> unique_dets;
    size_t hint = 0; for (auto& s : partials) hint += s.size();
    unique_dets.reserve(hint);
    for (auto& s : partials) {
        unique_dets.insert(s.begin(), s.end());
    }

    std::vector<SlaterDeterminant> connected_dets(unique_dets.begin(), unique_dets.end());
    std::sort(connected_dets.begin(), connected_dets.end());
    return connected_dets;
}


std::vector<SlaterDeterminant>
get_connections_two_body(const std::vector<SlaterDeterminant>& basis,
                         const std::vector<TwoBodyTerm>& terms)
{
    if (basis.empty() || terms.empty()) return {};

    const size_t n_spatial = basis[0].num_spatial_orbitals();

    // Read-only buckets keyed by canonical annihilated pair (k,sk),(l,sl)
    std::unordered_map<uint64_t, std::vector<const TwoBodyTerm*>> buckets;
    buckets.reserve(terms.size());
    for (const auto& term : terms) {
        if (std::abs(term.value) > 0.0) {
            buckets[make_key2(term.k, term.l, term.spin_k, term.spin_l)].push_back(&term);
        }
    }

    // Thread-local accumulation
    std::vector<std::unordered_set<SlaterDeterminant>> partials;
    partials.reserve(omp_get_max_threads());

    #pragma omp parallel
    {
        std::unordered_set<SlaterDeterminant> local;
        local.reserve(std::max<size_t>(16, basis.size() / std::max(1, omp_get_num_threads())));

        SlaterDeterminant excited_det(n_spatial);
        int8_t sign;

        // Per-thread scratch
        std::vector<std::pair<size_t, Spin>> occupied_orbs;
        occupied_orbs.reserve(n_spatial * 2);

        #pragma omp for schedule(guided)
        for (std::ptrdiff_t idx = 0; idx < static_cast<std::ptrdiff_t>(basis.size()); ++idx) {
            const auto& det = basis[static_cast<size_t>(idx)];

            occupied_orbs.clear();
            for (auto it = det.alpha().begin_occ(); it != det.alpha().end_occ(); ++it)
                occupied_orbs.emplace_back(static_cast<size_t>(*it), Spin::Alpha);
            for (auto it = det.beta().begin_occ(); it != det.beta().end_occ(); ++it)
                occupied_orbs.emplace_back(static_cast<size_t>(*it), Spin::Beta);

            const size_t L = occupied_orbs.size();
            for (size_t a = 0; a + 1 < L; ++a) {
                const auto [k, sk] = occupied_orbs[a];
                for (size_t b = a + 1; b < L; ++b) {
                    const auto [l, sl] = occupied_orbs[b];

                    auto key = make_key2(k, l, sk, sl);
                    auto bkt = buckets.find(key);
                    if (bkt == buckets.end()) continue;

                    for (const auto* term : bkt->second) {
                        // ensure the term orientation matches the occupied pair
                        const bool match_kl = (term->k == k && term->l == l &&
                                               term->spin_k == sk && term->spin_l == sl);
                        const bool match_lk = (term->k == l && term->l == k &&
                                               term->spin_k == sl && term->spin_l == sk);
                        if (!(match_kl || match_lk)) continue;

                        if (SlaterDeterminant::apply_excitation_double_fast(
                                det, term->i, term->j, term->k, term->l,
                                term->spin_i, term->spin_j, term->spin_k, term->spin_l,
                                excited_det, sign))
                        {
                            local.insert(excited_det);
                        }
                    }
                }
            }
        }

        #pragma omp critical
        {
            partials.push_back(std::move(local));
        }
    }

    // Merge thread-local sets
    std::unordered_set<SlaterDeterminant> unique_dets;
    size_t hint = 0; for (auto& s : partials) hint += s.size();
    unique_dets.reserve(hint);
    for (auto& s : partials) {
        unique_dets.insert(s.begin(), s.end());
    }

    std::vector<SlaterDeterminant> connected_dets(unique_dets.begin(), unique_dets.end());
    std::sort(connected_dets.begin(), connected_dets.end());
    return connected_dets;
}

} // namespace ci