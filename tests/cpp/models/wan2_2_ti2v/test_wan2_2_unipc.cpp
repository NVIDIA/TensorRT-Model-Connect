/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/wan2_2_ti2v/wan2_2_unipc.h"
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
    test_updates_match_upstream_cpu_and_cuda();
    test_official_fifty_step_cuda_fixture();
    test_official_full_shape_per_step_metrics();
    if (failures != 0) {
        std::cerr << failures << " Wan2.2 UniPC test(s) failed\n";
        return 1;
    }
    return 0;
}
