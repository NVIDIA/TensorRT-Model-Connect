#pragma once

#include <cstdint>
#include <vector>

namespace trtmc::diffusion {

std::vector<std::int32_t> derive_per_sample_seeds(std::uint64_t global_seed, int count);

std::vector<std::int32_t> derive_per_sample_seeds(
    const std::vector<std::uint64_t>& explicit_list, int count);

std::vector<int> plan_chunks(int total, int cap);

} // namespace trtmc::diffusion
