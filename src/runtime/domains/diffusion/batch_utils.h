#pragma once
// Shared helpers for diffusion batch inference.
//
// `derive_per_sample_seeds(global, N)` turns one global seed into N distinct,
// reproducible per-sample seeds via std::seed_seq{global_lo, global_hi, i}.
// `derive_per_sample_seeds(list, N)` validates and forwards an explicit list.
// `plan_chunks(total, cap)` returns the size of each TRT call when the user's
// requested batch exceeds the engine cap (design doc Decision D).
//
// These utilities are isolated so all three diffusion families can share one
// implementation and one regression test (see Decision D's per-sample seed
// contract; the AnimateDiff #174 lesson is the reason we don't broadcast one
// RNG stream across the batch).

#include <cstdint>
#include <vector>

namespace trtmc::diffusion {

// Derive `count` deterministic per-sample seeds from one `global_seed`.
// Each sample's seed = first u32 of std::seed_seq{global_lo, global_hi, i}.
std::vector<std::uint32_t> derive_per_sample_seeds(std::uint64_t global_seed, int count);

// Validate an explicit list provided by the user (e.g. --seed s0,s1,...).
// Throws std::invalid_argument when `list.size() != count`.
std::vector<std::uint32_t> derive_per_sample_seeds(const std::vector<std::uint64_t>& explicit_list,
                                                   int count);

// Split `total` work into chunks of at most `cap`. The last chunk may be
// smaller. Throws std::invalid_argument for non-positive inputs.
std::vector<int> plan_chunks(int total, int cap);

} // namespace trtmc::diffusion
