#include "runtime/domains/diffusion/batch_utils.h"

#include <algorithm>
#include <array>
#include <random>
#include <stdexcept>

namespace trtmc::diffusion {

namespace {

std::uint32_t low32(std::uint64_t v) {
    return static_cast<std::uint32_t>(v & 0xFFFFFFFFu);
}

std::uint32_t high32(std::uint64_t v) {
    return static_cast<std::uint32_t>((v >> 32) & 0xFFFFFFFFu);
}

} // namespace

std::vector<std::uint32_t> derive_per_sample_seeds(std::uint64_t global_seed, int count) {
    if (count < 1) {
        throw std::invalid_argument("count must be >= 1");
    }
    std::vector<std::uint32_t> out;
    out.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        std::seed_seq seq{
            low32(global_seed),
            high32(global_seed),
            static_cast<std::uint32_t>(i),
        };
        std::array<std::uint32_t, 1> buf{};
        seq.generate(buf.begin(), buf.end());
        out.push_back(buf[0]);
    }
    return out;
}

std::vector<std::uint32_t> derive_per_sample_seeds(const std::vector<std::uint64_t>& explicit_list,
                                                   int count) {
    if (count < 1) {
        throw std::invalid_argument("count must be >= 1");
    }
    if (static_cast<int>(explicit_list.size()) != count) {
        throw std::invalid_argument("explicit seed list length must equal the total batch count");
    }
    std::vector<std::uint32_t> out;
    out.reserve(static_cast<std::size_t>(count));
    for (auto v : explicit_list) {
        out.push_back(static_cast<std::uint32_t>(v));
    }
    return out;
}

std::vector<int> plan_chunks(int total, int cap) {
    if (total < 1) {
        throw std::invalid_argument("total must be >= 1");
    }
    if (cap < 1) {
        throw std::invalid_argument("cap must be >= 1");
    }
    std::vector<int> plan;
    while (total > 0) {
        const int n = std::min(total, cap);
        plan.push_back(n);
        total -= n;
    }
    return plan;
}

} // namespace trtmc::diffusion
