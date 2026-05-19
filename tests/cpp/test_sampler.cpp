// =============================================================================
// Sampling parameter and sampler behavior tests
// =============================================================================

#include "trtmc/pipeline.h"
#include "trtmc/runtime/sampler.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#if TRTMC_HAS_LIBTORCH_MULTINOMIAL && TRTMC_HAS_CUDA_KERNELS
#include <cuda_runtime_api.h>
#endif

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static std::unique_ptr<trtmc::ISampler> create_host_sampler(const trtmc::SamplingParams& params) {
    trtmc::SamplerFactoryOptions options;
    options.prefer_torch_cuda_multinomial = false;
    return trtmc::create_sampler(params, options);
}

static void test_sampling_params_from_config() {
    trtmc::GenerateConfig cfg;
    cfg.temperature = 0.6F;
    cfg.top_k = 20;
    cfg.top_p = 0.95F;
    cfg.min_p = 0.1F;
    cfg.seed = 123;
    cfg.eos_token_id = 99;
    auto params = trtmc::sampling_params_from_config(cfg, 42);
    check(params.temperature == 0.6F, "temperature forwarded");
    check(params.top_k == 20, "top_k forwarded");
    check(params.top_p == 0.95F, "top_p forwarded");
    check(params.min_p == 0.1F, "min_p forwarded");
    check(params.seed == 123, "seed forwarded");
    check(params.eos_token_id == 99, "explicit eos forwarded");
}

static void test_create_sampler_greedy_only_when_sampling_disabled() {
    trtmc::SamplingParams params;
    params.top_k = 1;
    params.top_p = 1.0F;
    params.min_p = 0.0F;
    params.seed = -1;
    auto sampler = create_host_sampler(params);
    check(std::string(sampler->sampler_type()) == "greedy", "greedy factory path");

    params.top_p = 0.95F;
    sampler = trtmc::create_sampler(params);
    const std::string sampler_type = sampler->sampler_type();
    check(sampler_type == "top_k" || sampler_type == "torch_multinomial",
          "top_p forces sampling path");
}

static void test_top_p_alone_uses_full_vocab() {
    trtmc::SamplingParams params;
    params.temperature = 1.0F;
    params.top_k = 1;
    params.top_p = 0.95F;
    params.min_p = 0.0F;
    params.seed = 99;
    auto sampler = create_host_sampler(params);

    const std::vector<float> logits = {1.0F, 1.0F, 1.0F, 1.0F, 1.0F};
    std::set<int32_t> seen;
    for (int i = 0; i < 500; ++i) {
        auto result = sampler->sample(logits.data(), static_cast<int32_t>(logits.size()), params);
        seen.insert(result.token_id);
    }
    check(seen.size() >= 2, "top_p alone uses full vocab when top_k default is 1");
}

static void test_top_k_zero_means_no_topk_limit() {
    trtmc::SamplingParams params;
    params.temperature = 1.0F;
    params.top_k = 0;
    params.top_p = 0.95F;
    params.min_p = 0.0F;
    params.seed = 17;
    auto sampler = create_host_sampler(params);

    const std::vector<float> logits = {1.0F, 1.0F, 1.0F, 1.0F, 1.0F};
    std::set<int32_t> seen;
    for (int i = 0; i < 500; ++i) {
        auto result = sampler->sample(logits.data(), static_cast<int32_t>(logits.size()), params);
        seen.insert(result.token_id);
    }
    check(seen.size() >= 2, "top_k zero means no top-k limit");
}

static void test_min_p_without_top_p_keeps_default_top_k() {
    trtmc::SamplingParams params;
    params.temperature = 1.0F;
    params.top_k = 1;
    params.top_p = 1.0F;
    params.min_p = 0.5F;
    params.seed = 33;
    auto sampler = create_host_sampler(params);

    const float logits[] = {5.0F, 4.0F, 4.0F, 4.0F};
    for (int i = 0; i < 64; ++i) {
        auto result = sampler->sample(logits, 4, params);
        check(result.token_id == 0, "min_p without top_p keeps default top_k=1 behavior");
    }
}

static void test_top_p_zero_is_greedy() {
    const float logits[] = {0.1F, 5.0F, 2.3F, 0.7F};
    trtmc::SamplingParams params;
    params.temperature = 1.0F;
    params.top_k = 4;
    params.top_p = 0.0F;
    params.min_p = 0.0F;
    params.seed = 7;
    auto sampler = create_host_sampler(params);
    auto result = sampler->sample(logits, 4, params);
    check(result.token_id == 1, "top_p zero is greedy");
}

static void test_invalid_sampling_values_are_sanitized() {
    const float logits[] = {0.1F, 5.0F, 2.3F, 0.7F};
    trtmc::SamplingParams params;
    params.temperature = -1.0F;
    params.top_k = 4;
    params.top_p = 1.5F;
    params.min_p = -0.2F;
    params.seed = 7;
    auto sampler = create_host_sampler(params);
    auto result = sampler->sample(logits, 4, params);
    check(result.token_id == 1, "invalid sampling values are sanitized");
}

static void test_sampler_reset_replays_seeded_sequence() {
    trtmc::SamplingParams params;
    params.temperature = 1.0F;
    params.top_k = 0;
    params.top_p = 0.95F;
    params.min_p = 0.0F;
    params.seed = 123;
    auto sampler = create_host_sampler(params);

    const std::vector<float> logits = {1.0F, 0.9F, 0.8F, 0.7F, 0.6F};
    std::vector<int32_t> first;
    std::vector<int32_t> second;
    for (int i = 0; i < 32; ++i)
        first.push_back(
            sampler->sample(logits.data(), static_cast<int32_t>(logits.size()), params).token_id);
    sampler->reset();
    for (int i = 0; i < 32; ++i)
        second.push_back(
            sampler->sample(logits.data(), static_cast<int32_t>(logits.size()), params).token_id);

    check(first == second, "sampler reset replays seeded sequence");
}

static void test_create_sampler_can_force_host_topk() {
    trtmc::SamplingParams params;
    params.top_k = 4;
    params.seed = 123;
    trtmc::SamplerFactoryOptions options;
    options.prefer_torch_cuda_multinomial = false;
    auto sampler = trtmc::create_sampler(params, options);
    check(std::string(sampler->sampler_type()) == "top_k", "factory can force host top_k");
}

static void test_top_p_truncates_tail_tokens() {
    const float logits[] = {10.0F, 9.0F, -10.0F};
    trtmc::SamplingParams params;
    params.temperature = 1.0F;
    params.top_k = 3;
    params.top_p = 0.55F;
    params.min_p = 0.0F;
    params.seed = 17;
    auto sampler = create_host_sampler(params);
    auto result = sampler->sample(logits, 3, params);
    check(result.token_id == 0, "top_p keeps only the highest-prob token");
}

static void test_min_p_drops_low_probability_tail() {
    const float logits[] = {5.0F, 5.0F, 0.0F};
    trtmc::SamplingParams params;
    params.temperature = 1.0F;
    params.top_k = 3;
    params.top_p = 1.0F;
    params.min_p = 0.75F;
    params.seed = 9;
    auto sampler = create_host_sampler(params);
    for (int i = 0; i < 32; ++i) {
        auto result = sampler->sample(logits, 3, params);
        check(result.token_id != 2, "min_p excludes low-probability token");
    }
}

#if TRTMC_HAS_LIBTORCH_MULTINOMIAL && TRTMC_HAS_CUDA_KERNELS
static bool cuda_device_available() {
    int count = 0;
    const cudaError_t status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return count > 0;
}

static void test_torch_multinomial_matches_known_hf_sequence() {
    trtmc::SamplingParams params;
    params.temperature = 1.0F;
    params.top_k = 20;
    params.top_p = 0.95F;
    params.min_p = 0.0F;
    params.seed = 1235;

    auto sampler = trtmc::create_sampler(params);
    check(std::string(sampler->sampler_type()) == "torch_multinomial", "torch sampler enabled");

    const float step0[] = {46.041664F, 43.75F, 43.541664F};
    const float step1[] = {46.875F, 45.416664F};
    const float step2[] = {51.666664F, 51.458332F};

    auto result0 = sampler->sample(step0, 3, params);
    auto result1 = sampler->sample(step1, 2, params);
    auto result2 = sampler->sample(step2, 2, params);

    check(result0.token_id == 0, "torch sampler step0 matches HF");
    check(result1.token_id == 0, "torch sampler step1 matches HF");
    check(result2.token_id == 0, "torch sampler step2 matches HF");
}

static void test_torch_multinomial_uses_full_vocab_semantics() {
    trtmc::SamplingParams params;
    params.temperature = 1.0F;
    params.top_k = 2;
    params.top_p = 1.0F;
    params.min_p = 0.0F;
    params.seed = 1235;

    auto sampler = trtmc::create_sampler(params);
    check(std::string(sampler->sampler_type()) == "torch_multinomial",
          "torch sampler enabled for sparse full-vocab test");

    std::vector<float> logits(100000, -1000.0F);
    logits[279] = 0.0F;
    logits[419] = std::log(0.45458386F / 0.54541614F);

    auto result = sampler->sample(logits.data(), static_cast<int32_t>(logits.size()), params);
    check(result.token_id == 419, "torch sampler matches full-vocab CUDA multinomial");
}

static void test_torch_multinomial_advances_offset_like_full_vocab_cuda() {
    trtmc::SamplingParams params;
    params.temperature = 1.0F;
    params.top_k = 2;
    params.top_p = 1.0F;
    params.min_p = 0.0F;
    params.seed = 1235;

    auto sampler = trtmc::create_sampler(params);
    check(std::string(sampler->sampler_type()) == "torch_multinomial",
          "torch sampler enabled for offset test");

    std::vector<float> logits(100000, -1000.0F);
    logits[279] = 0.0F;
    logits[419] = std::log(0.45458386F / 0.54541614F);

    const int expected[] = {419, 279, 279, 419, 279, 279, 419, 419};
    for (int token : expected) {
        auto result = sampler->sample(logits.data(), static_cast<int32_t>(logits.size()), params);
        check(result.token_id == token, "torch sampler preserves full-vocab offset progression");
    }
}

static void test_torch_multinomial_matches_live_step_three_way_case() {
    trtmc::SamplingParams params;
    params.temperature = 0.6F;
    params.top_k = 20;
    params.top_p = 0.95F;
    params.min_p = 0.0F;
    params.seed = 1235;

    auto sampler = trtmc::create_sampler(params);
    check(std::string(sampler->sampler_type()) == "torch_multinomial",
          "torch sampler enabled for live three-way test");

    std::vector<float> logits(151936, -1000.0F);
    logits[2014] = 27.5312F;
    logits[576] = 26.2188F;
    logits[6771] = 26.0938F;

    auto result = sampler->sample(logits.data(), static_cast<int32_t>(logits.size()), params);
    check(result.token_id == 2014, "torch sampler matches three-way live-step synthetic case");
}
#endif

int main() {
    test_sampling_params_from_config();
    test_create_sampler_greedy_only_when_sampling_disabled();
    test_top_p_alone_uses_full_vocab();
    test_top_k_zero_means_no_topk_limit();
    test_min_p_without_top_p_keeps_default_top_k();
    test_top_p_zero_is_greedy();
    test_invalid_sampling_values_are_sanitized();
    test_sampler_reset_replays_seeded_sequence();
    test_create_sampler_can_force_host_topk();
    test_top_p_truncates_tail_tokens();
    test_min_p_drops_low_probability_tail();
#if TRTMC_HAS_LIBTORCH_MULTINOMIAL && TRTMC_HAS_CUDA_KERNELS
    if (cuda_device_available()) {
        test_torch_multinomial_matches_known_hf_sequence();
        test_torch_multinomial_uses_full_vocab_semantics();
        test_torch_multinomial_advances_offset_like_full_vocab_cuda();
        test_torch_multinomial_matches_live_step_three_way_case();
    } else {
        std::cerr << "Skipping CUDA multinomial sampler tests: no CUDA device available.\n";
    }
#endif

    if (failures > 0) {
        std::cerr << failures << " sampler test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All sampler tests passed.\n";
    return 0;
}
