#include "runtime/domains/diffusion/batch_utils.h"

#include <algorithm>
#include <array>
#include <limits>
#include <random>
#include <stdexcept>

namespace trtmc::diffusion {

namespace {

std::int32_t normalize_public_seed(std::uint32_t seed) {
    return static_cast<std::int32_t>(seed & 0x7FFFFFFFU);
}

} // namespace

std::vector<std::int32_t> derive_per_sample_seeds(std::uint64_t global_seed, int count) {
    if (count < 1)
        throw std::invalid_argument("count must be >= 1");

    std::vector<std::int32_t> out(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        std::seed_seq seq{
            static_cast<std::uint32_t>(global_seed & 0xFFFFFFFFULL),
            static_cast<std::uint32_t>(global_seed >> 32U),
            static_cast<std::uint32_t>(i),
        };
        std::array<std::uint32_t, 1> buf{};
        seq.generate(buf.begin(), buf.end());
        out[static_cast<std::size_t>(i)] = normalize_public_seed(buf[0]);
    }
    return out;
}

std::vector<std::int32_t> derive_per_sample_seeds(
    const std::vector<std::uint64_t>& explicit_list, int count) {
    if (count < 1)
        throw std::invalid_argument("count must be >= 1");
    if (static_cast<int>(explicit_list.size()) != count)
        throw std::invalid_argument("--seed list length must equal total batch count");

    std::vector<std::int32_t> out;
    out.reserve(explicit_list.size());
    for (std::uint64_t seed : explicit_list) {
        if (seed > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()))
            throw std::invalid_argument("--seed values must fit in int32");
        out.push_back(static_cast<std::int32_t>(seed));
    }
    return out;
}

std::vector<int> plan_chunks(int total, int cap) {
    if (total < 1)
        throw std::invalid_argument("total must be >= 1");
    if (cap < 1)
        throw std::invalid_argument("cap must be >= 1");

    std::vector<int> plan;
    while (total > 0) {
        const int n = std::min(total, cap);
        plan.push_back(n);
        total -= n;
    }
    return plan;
}

} // namespace trtmc::diffusion
