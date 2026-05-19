#include "runtime/domains/diffusion/batch_utils.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static void test_per_sample_seed_from_global() {
    const auto seeds = trtmc::diffusion::derive_per_sample_seeds(42, 4);
    check(seeds.size() == 4U, "seed count");
    check(seeds[1] != seeds[0], "seed 1 differs");
    check(seeds[2] != seeds[0], "seed 2 differs");
    check(seeds == trtmc::diffusion::derive_per_sample_seeds(42, 4),
          "seed derivation deterministic");
}

static void test_explicit_seed_list_must_match_count() {
    bool threw = false;
    try {
        (void)trtmc::diffusion::derive_per_sample_seeds(
            std::vector<std::uint64_t>{1, 2, 3}, 4);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "explicit seed count mismatch throws");
}

static void test_chunking_divides_evenly() {
    const auto plan = trtmc::diffusion::plan_chunks(8, 4);
    check(plan == std::vector<int>({4, 4}), "chunking divides evenly");
}

static void test_chunking_handles_remainder() {
    const auto plan = trtmc::diffusion::plan_chunks(9, 4);
    check(plan == std::vector<int>({4, 4, 1}), "chunking handles remainder");
}

static void test_chunking_rejects_invalid_cap() {
    bool threw = false;
    try {
        (void)trtmc::diffusion::plan_chunks(4, 0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "chunking rejects invalid cap");
}

int main() {
    test_per_sample_seed_from_global();
    test_explicit_seed_list_must_match_count();
    test_chunking_divides_evenly();
    test_chunking_handles_remainder();
    test_chunking_rejects_invalid_cap();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All diffusion_batch_utils tests passed.\n";
    return 0;
}
