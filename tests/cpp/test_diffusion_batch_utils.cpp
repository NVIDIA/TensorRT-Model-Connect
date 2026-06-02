// Unit test for the shared per-sample seed and chunk-planning helpers used
// by every diffusion pipeline. Per design doc Decision D.

#include "runtime/domains/diffusion/batch_utils.h"

#include <cstdint>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

void test_batch_utils_contracts() {
    // Per-sample seeds from one global are distinct + reproducible.
    auto a = trtmc::diffusion::derive_per_sample_seeds(/*global=*/42, /*count=*/4);
    auto b = trtmc::diffusion::derive_per_sample_seeds(/*global=*/42, /*count=*/4);
    check(a == b && a.size() == 4, "global seed reproduces same sequence of 4");
    std::set<std::uint32_t> unique(a.begin(), a.end());
    check(unique.size() == 4, "per-sample seeds are distinct");

    // Explicit list: forwarded verbatim, length-mismatched list throws.
    auto explicit_seeds = trtmc::diffusion::derive_per_sample_seeds(
        std::vector<std::uint64_t>{10, 20, 30, 40}, /*count=*/4);
    check(explicit_seeds.size() == 4 && explicit_seeds[0] == 10 && explicit_seeds[3] == 40,
          "explicit list forwarded unchanged");
    bool threw = false;
    try {
        (void)trtmc::diffusion::derive_per_sample_seeds(std::vector<std::uint64_t>{1, 2, 3},
                                                        /*count=*/4);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "length-mismatched explicit list throws");

    // Chunk planning: even split + remainder.
    auto even = trtmc::diffusion::plan_chunks(/*total=*/8, /*cap=*/4);
    check(even == std::vector<int>{4, 4}, "8/4 splits evenly");
    auto rem = trtmc::diffusion::plan_chunks(/*total=*/9, /*cap=*/4);
    check(rem == std::vector<int>{4, 4, 1}, "9/4 leaves remainder as final chunk");
}

} // namespace

int main() {
    test_batch_utils_contracts();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All diffusion batch_utils tests passed.\n";
    return 0;
}
