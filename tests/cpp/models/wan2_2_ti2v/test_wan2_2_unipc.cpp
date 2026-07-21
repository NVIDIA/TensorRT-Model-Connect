/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/wan2_2_ti2v/wan2_2_unipc.h"
#include "runtime/models/wan2_2_ti2v/wan2_2_unipc_coefficients.h"
#include "runtime/models/wan2_2_ti2v/wan2_2_unipc_coefficients_15.h"
#include "utils/sha256.h"
#include "wan2_2_unipc_full_golden.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

constexpr std::size_t kOfficialLatentCount = 48ULL * 31ULL * 44ULL * 80ULL;

struct TensorStats {
    double min;
    double max;
    double mean;
    double rms;
};

TensorStats tensor_stats(const std::vector<float>& values) {
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    double sum = 0.0;
    double square_sum = 0.0;
    for (const float value : values) {
        const double widened = value;
        minimum = std::min(minimum, widened);
        maximum = std::max(maximum, widened);
        sum += widened;
        square_sum += widened * widened;
    }
    const double count = static_cast<double>(values.size());
    return {minimum, maximum, sum / count, std::sqrt(square_sum / count)};
}

float fixture_value(std::size_t index, int32_t step, uint32_t multiplier, uint32_t step_multiplier,
                    uint32_t salt) {
    const uint32_t mixed = static_cast<uint32_t>(index) * multiplier +
                           static_cast<uint32_t>(step + 1) * step_multiplier + salt;
    const uint32_t sign = (mixed >> 31U) << 31U;
    const uint32_t exponent = (126U + ((mixed >> 30U) & 1U)) << 23U;
    const uint32_t bits = sign | exponent | (mixed & 0x007FFFFFU);
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void fill_initial_sample(std::vector<float>& sample) {
    for (std::size_t index = 0; index < sample.size(); ++index)
        sample[index] = fixture_value(index, -1, 747796405U, 0U, 2891336453U);
}

void fill_model_outputs(std::vector<float>& conditional, std::vector<float>& unconditional,
                        int32_t step) {
    for (std::size_t index = 0; index < conditional.size(); ++index) {
        conditional[index] = fixture_value(index, step, 277803737U, 1013904223U, 0x12345678U);
        unconditional[index] = fixture_value(index, step, 1664525U, 22695477U, 0x9E3779B9U);
    }
}

int stream_full_shape_replay() {
    trtmc::wan2_2_ti2v::FlowUniPC scheduler(50, 5.0F, 1000);
    std::vector<float> sample(kOfficialLatentCount);
    std::vector<float> conditional(kOfficialLatentCount);
    std::vector<float> unconditional(kOfficialLatentCount);
    std::vector<float> guided(kOfficialLatentCount);
    std::vector<float> next(kOfficialLatentCount);
    fill_initial_sample(sample);

    for (int32_t step = 0; step < 50; ++step) {
        fill_model_outputs(conditional, unconditional, step);
        for (std::size_t index = 0; index < kOfficialLatentCount; ++index)
            guided[index] =
                unconditional[index] + 5.0F * (conditional[index] - unconditional[index]);
        std::cout.write(reinterpret_cast<const char*>(guided.data()),
                        static_cast<std::streamsize>(guided.size() * sizeof(float)));
        scheduler.step(guided.data(), sample.data(), next.data(), sample.size());
        sample.swap(next);
        std::cout.write(reinterpret_cast<const char*>(sample.data()),
                        static_cast<std::streamsize>(sample.size() * sizeof(float)));
        if (!std::cout)
            return 2;
    }
    return 0;
}

void check(bool condition, const char* label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << '\n';
        ++failures;
    }
}

void check_close(float actual, float expected, float tolerance, const char* label) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL: " << label << " actual=" << actual << " expected=" << expected << '\n';
        ++failures;
    }
}

void append_update_words(std::vector<std::uint32_t>& words,
                         const trtmc::wan2_2_ti2v::unipc_coefficients::UpdateCoefficients& update) {
    words.insert(words.end(), {update.order, update.sigma_t_index, update.sigma_s0_index,
                               update.rk_count, update.rho_count, update.ratio_bits,
                               update.model_coefficient_bits, update.residual_coefficient_bits});
    words.insert(words.end(), update.rk_bits.begin(), update.rk_bits.end());
    words.insert(words.end(), update.rho_bits.begin(), update.rho_bits.end());
}

std::vector<std::uint32_t> packed_coefficient_words() {
    namespace coefficients = trtmc::wan2_2_ti2v::unipc_coefficients;
    std::vector<std::uint32_t> words = {
        static_cast<std::uint32_t>(coefficients::kStepCount),
        static_cast<std::uint32_t>(coefficients::kSigmaCount),
        coefficients::kNumTrainTimesteps,
        coefficients::kSolverOrder,
        coefficients::kFlowShiftBits,
        coefficients::kNoSigmaIndex,
    };
    words.reserve(1357U);
    words.insert(words.end(), coefficients::kTimesteps.begin(), coefficients::kTimesteps.end());
    words.insert(words.end(), coefficients::kSigmaBits.begin(), coefficients::kSigmaBits.end());
    words.insert(words.end(), coefficients::kConversionSigmaBits.begin(),
                 coefficients::kConversionSigmaBits.end());
    for (const auto& corrector : coefficients::kCorrector)
        append_update_words(words, corrector);
    for (const auto& predictor : coefficients::kPredictor)
        append_update_words(words, predictor);
    return words;
}

std::vector<std::uint32_t> packed_l0_coefficient_words() {
    namespace coefficients = trtmc::wan2_2_ti2v::unipc_coefficients_15;
    std::vector<std::uint32_t> words = {
        static_cast<std::uint32_t>(coefficients::kStepCount),
        static_cast<std::uint32_t>(coefficients::kSigmaCount),
        coefficients::kNumTrainTimesteps,
        coefficients::kSolverOrder,
        coefficients::kFlowShiftBits,
        coefficients::kNoSigmaIndex,
    };
    words.reserve(412U);
    words.insert(words.end(), coefficients::kTimesteps.begin(), coefficients::kTimesteps.end());
    words.insert(words.end(), coefficients::kSigmaBits.begin(), coefficients::kSigmaBits.end());
    words.insert(words.end(), coefficients::kConversionSigmaBits.begin(),
                 coefficients::kConversionSigmaBits.end());
    for (const auto& corrector : coefficients::kCorrector)
        append_update_words(words, corrector);
    for (const auto& predictor : coefficients::kPredictor)
        append_update_words(words, predictor);
    return words;
}

std::string packed_words_sha256(const std::vector<std::uint32_t>& words) {
    trtmc::internal::Sha256 digest;
    for (const auto word : words) {
        const std::array<std::uint8_t, 4> big_endian = {
            static_cast<std::uint8_t>(word >> 24U),
            static_cast<std::uint8_t>(word >> 16U),
            static_cast<std::uint8_t>(word >> 8U),
            static_cast<std::uint8_t>(word),
        };
        digest.update(big_endian.data(), big_endian.size());
    }
    return digest.hex_digest();
}

void test_official_coefficient_payload_is_bit_exact() {
    namespace coefficients = trtmc::wan2_2_ti2v::unipc_coefficients;
    constexpr auto expected = "a86473aed0e63c6e3f2334dafbf7c02de09dfd43f075543cb94478b6c9f19635";
    auto words = packed_coefficient_words();
    check(words.size() == 1357U, "Wan2.2 packed UniPC coefficient word count is exact");
    check(packed_words_sha256(words) == expected,
          "Wan2.2 packed UniPC coefficient payload SHA-256 is exact");

    words.front() ^= 1U;
    check(packed_words_sha256(words) != expected,
          "Wan2.2 coefficient digest detects a first-word bit change");
    words.front() ^= 1U;
    words.back() ^= 1U;
    check(packed_words_sha256(words) != expected,
          "Wan2.2 coefficient digest detects a last-word bit change");
    check(!coefficients::kOfficialSourceTrackedDirty,
          "Wan2.2 official UniPC source provenance is clean");
    check(std::string_view(coefficients::kArtifactGeneratorSha256) ==
              "adb1e0a3839924ed4982c872909ab044335fa22f8319a5048b2e896e61e053bb",
          "Wan2.2 official UniPC generator digest is current");
    check(std::string_view(coefficients::kArtifactSha256) ==
              "650f7e64cddb551bd81ee4386857967dcf2a916ea3a04cb97423b74a522cf782",
          "Wan2.2 official UniPC artifact digest is current");
}

void test_l0_coefficient_payload_and_provenance_are_exact() {
    namespace coefficients = trtmc::wan2_2_ti2v::unipc_coefficients_15;
    constexpr auto expected_packed =
        "c022a6f1a1061a043ad1951aec928248bae6c5612635427457aadfabc9541239";
    const auto words = packed_l0_coefficient_words();
    check(words.size() == 412U, "Wan2.2 L0 packed UniPC coefficient word count is exact");
    check(packed_words_sha256(words) == expected_packed,
          "Wan2.2 L0 packed UniPC coefficient payload SHA-256 is exact");
    check(coefficients::canonical_numerical_payload_fnv1a64() == 0x8d4512f130f834d2ULL,
          "Wan2.2 L0 UniPC numerical FNV is exact");
    check(std::string_view(coefficients::kOfficialSourceRevision) ==
              "42bf4cfaa384bc21833865abc2f9e6c0e67233dc",
          "Wan2.2 L0 UniPC official source revision is pinned");
    check(std::string_view(coefficients::kOfficialSchedulerSha256) ==
              "0dec8c7ed17f6f2049275c6848113314da6ccec1c8db5bdc89df43c05c6038d9",
          "Wan2.2 L0 UniPC official scheduler digest is pinned");
    check(std::string_view(coefficients::kGeneratorSha256) ==
              "adb1e0a3839924ed4982c872909ab044335fa22f8319a5048b2e896e61e053bb",
          "Wan2.2 L0 UniPC generator digest is pinned");
    check(std::string_view(coefficients::kArtifactSha256) ==
              "bcaae1cf25b1b11d35dcb75ab87b8659a4cafa33c63e0a9e68494f1d73060dd4",
          "Wan2.2 L0 UniPC artifact digest is pinned");
    check(std::string_view(coefficients::kNormalizedQualificationPayloadSha256) ==
              "fcc95b5571f76c0f7363624f1124b1bac9174c23c16724d35545b50586bf0e90",
          "Wan2.2 L0 normalized qualification payload digest is pinned");
}

void check_full_shape_close(double actual, double expected, double tolerance, int32_t step,
                            const char* field) {
    if (std::fabs(actual - expected) > tolerance) {
        std::cerr << "FAIL: full-shape step " << (step + 1) << ' ' << field << " actual=" << actual
                  << " expected=" << expected << " tolerance=" << tolerance << '\n';
        ++failures;
    }
}

void test_schedule_matches_upstream_wan22() {
    trtmc::wan2_2_ti2v::FlowUniPC scheduler(3, 5.0F, 1000);
    check(scheduler.timesteps() == std::vector<int64_t>({999, 908, 713}),
          "Wan2.2 three-step timesteps match upstream");
    const std::vector<float> expected_sigmas = {
        0.9997998476028442F,
        0.9088428020477295F,
        0.7139794230461121F,
        0.0F,
    };
    for (std::size_t i = 0; i < expected_sigmas.size(); ++i)
        check_close(scheduler.sigmas()[i], expected_sigmas[i], 1.0e-7F,
                    "Wan2.2 shifted sigma matches upstream");
}

void test_l0_schedule_matches_upstream_wan22() {
    trtmc::wan2_2_ti2v::FlowUniPC scheduler(15, 5.0F, 1000);
    const std::vector<int64_t> expected = {
        999, 985, 969, 952, 931, 908, 882, 850, 813, 768, 713, 644, 555, 434, 262,
    };
    check(scheduler.timesteps() == expected, "Wan2.2 L0 timesteps match upstream");
}

void test_updates_match_upstream_cpu_and_cuda() {
    trtmc::wan2_2_ti2v::FlowUniPC scheduler(3, 5.0F, 1000);
    std::vector<float> sample = {0.25F, -0.5F, 1.0F, -1.5F, 2.0F};
    const std::vector<std::vector<float>> model_outputs = {
        {0.1F, -0.2F, 0.3F, -0.4F, 0.5F},
        {0.13F, -0.18F, 0.29F, -0.36F, 0.48F},
        {0.16F, -0.16F, 0.28F, -0.32F, 0.46F},
    };
    const std::vector<std::vector<float>> expected = {
        {0.240904301404953F, -0.481808602809906F, 0.972712934017181F, -1.463617205619812F,
         1.954521536827087F},
        {0.213946640491486F, -0.447816818952560F, 0.916744351387024F, -1.395633816719055F,
         1.862070679664612F},
        {0.099709935486317F, -0.333580106496811F, 0.716830134391785F, -1.167160391807556F,
         1.533640146255493F},
    };

    for (std::size_t step = 0; step < model_outputs.size(); ++step) {
        scheduler.step(model_outputs[step].data(), sample.data(), sample.data(), sample.size());
        for (std::size_t i = 0; i < sample.size(); ++i)
            check_close(sample[i], expected[step][i], 2.0e-6F,
                        "Wan2.2 UniPC update matches upstream");
    }
}

void test_official_fifty_step_cuda_fixture() {
    trtmc::wan2_2_ti2v::FlowUniPC scheduler(50, 5.0F, 1000);
    const std::vector<int64_t> expected_timesteps = {
        999, 995, 991, 987, 982, 978, 973, 968, 963, 957, 952, 946, 940, 934, 927, 920, 913,
        906, 898, 890, 882, 873, 863, 854, 843, 833, 821, 809, 796, 783, 768, 753, 737, 720,
        701, 681, 660, 636, 611, 584, 555, 522, 487, 448, 405, 356, 302, 241, 172, 92,
    };
    check(scheduler.timesteps() == expected_timesteps,
          "Wan2.2 fifty-step timesteps match official CUDA fixture");

    std::vector<float> sample(17);
    for (std::size_t i = 0; i < sample.size(); ++i)
        sample[i] = -2.0F + 0.25F * static_cast<float>(i);
    std::vector<float> model_output(sample.size());
    for (int32_t step = 0; step < 50; ++step) {
        for (std::size_t i = 0; i < model_output.size(); ++i)
            model_output[i] =
                static_cast<float>(i) * 0.03125F + static_cast<float>(step + 1) * 0.0078125F;
        scheduler.step(model_output.data(), sample.data(), sample.data(), sample.size());
    }
    const std::vector<float> expected = {
        -2.2996532917022705F, -2.0808982849121094F,  -1.8621407747268677F,  -1.643385648727417F,
        -1.424628496170044F,  -1.2058722972869873F,  -0.9871161580085754F,  -0.7683598399162292F,
        -0.5496037602424622F, -0.33084753155708313F, -0.11209103465080261F, 0.10666511952877045F,
        0.32542091608047485F, 0.5441775321960449F,   0.7629338502883911F,   0.9816898107528687F,
        1.2004461288452148F,
    };
    for (std::size_t i = 0; i < expected.size(); ++i)
        check_close(sample[i], expected[i], 4.0e-6F,
                    "Wan2.2 fifty-step update matches official CUDA fixture");
}

void test_official_full_shape_per_step_metrics() {
    constexpr double kCfgExtremaTolerance = 2.0e-6;
    constexpr double kLatentExtremaTolerance = 4.0e-6;
    constexpr double kAggregateTolerance = 5.0e-7;
    constexpr double kProbeTolerance = 4.0e-6;

    trtmc::wan2_2_ti2v::FlowUniPC scheduler(50, 5.0F, 1000);
    std::vector<float> sample(kOfficialLatentCount);
    std::vector<float> conditional(kOfficialLatentCount);
    std::vector<float> unconditional(kOfficialLatentCount);
    std::vector<float> guided(kOfficialLatentCount);
    std::vector<float> next(kOfficialLatentCount);
    fill_initial_sample(sample);

    for (int32_t step = 0; step < 50; ++step) {
        fill_model_outputs(conditional, unconditional, step);
        for (std::size_t index = 0; index < kOfficialLatentCount; ++index)
            guided[index] =
                unconditional[index] + 5.0F * (conditional[index] - unconditional[index]);

        const auto cfg_stats = tensor_stats(guided);
        scheduler.step(guided.data(), sample.data(), next.data(), sample.size());
        sample.swap(next);
        const auto latent_stats = tensor_stats(sample);
        const auto& expected =
            wan22_unipc_test_fixture::kOfficialSteps[static_cast<std::size_t>(step)];

        check_full_shape_close(cfg_stats.min, expected.cfg_min, kCfgExtremaTolerance, step,
                               "CFG min");
        check_full_shape_close(cfg_stats.max, expected.cfg_max, kCfgExtremaTolerance, step,
                               "CFG max");
        check_full_shape_close(cfg_stats.mean, expected.cfg_mean, kAggregateTolerance, step,
                               "CFG mean");
        check_full_shape_close(cfg_stats.rms, expected.cfg_rms, kAggregateTolerance, step,
                               "CFG RMS");
        check_full_shape_close(latent_stats.min, expected.latent_min, kLatentExtremaTolerance, step,
                               "latent min");
        check_full_shape_close(latent_stats.max, expected.latent_max, kLatentExtremaTolerance, step,
                               "latent max");
        check_full_shape_close(latent_stats.mean, expected.latent_mean, kAggregateTolerance, step,
                               "latent mean");
        check_full_shape_close(latent_stats.rms, expected.latent_rms, kAggregateTolerance, step,
                               "latent RMS");

        const auto probe_slot = wan22_unipc_test_fixture::kRotatingProbeSlots
            [static_cast<std::size_t>(step) % wan22_unipc_test_fixture::kRotatingProbeSlots.size()];
        const auto probe_index = wan22_unipc_test_fixture::kProbeIndices[probe_slot];
        check_full_shape_close(guided[probe_index], expected.cfg_probe, kProbeTolerance, step,
                               "CFG probe");
        check_full_shape_close(sample[probe_index], expected.latent_probe, kProbeTolerance, step,
                               "latent probe");
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--stream-full")
        return stream_full_shape_replay();
    test_schedule_matches_upstream_wan22();
    test_l0_schedule_matches_upstream_wan22();
    test_official_coefficient_payload_is_bit_exact();
    test_l0_coefficient_payload_and_provenance_are_exact();
    test_updates_match_upstream_cpu_and_cuda();
    test_official_fifty_step_cuda_fixture();
    test_official_full_shape_per_step_metrics();
    if (failures != 0) {
        std::cerr << failures << " Wan2.2 UniPC test(s) failed\n";
        return 1;
    }
    return 0;
}
