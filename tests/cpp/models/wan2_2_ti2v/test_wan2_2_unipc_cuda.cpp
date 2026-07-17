/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/wan2_2_ti2v/wan2_2_unipc_cuda.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kOfficialLatentCount = 48ULL * 31ULL * 44ULL * 80ULL;

enum class ProbeStage : uint8_t { kCorrected, kLatent };

struct ExactProbe {
    int32_t step;
    ProbeStage stage;
    std::size_t index;
    uint32_t expected_bits;
};

// Generated from official Wan2.2 revision
// 42bf4cfaa384bc21833865abc2f9e6c0e67233dc with
// generate_unipc_fixture.py --stream-stages --autocast-bf16. These probes pin
// the real denoising-loop contract rather than the non-autocast scheduler in
// isolation. In particular, steps 2 and 3 cover BF16 einsum/output rounding,
// while step 21 is the first step that exposed approximate __fdividef
// reciprocal semantics in the source-qualified synthetic trajectory.
inline constexpr std::array<ExactProbe, 7> kExactProbes = {{
    {2, ProbeStage::kLatent, 3, 0x3F0F65CEU},
    {3, ProbeStage::kCorrected, 8, 0x3F5ED98DU},
    {21, ProbeStage::kLatent, 8, 0x3F08A778U},
    {21, ProbeStage::kLatent, 79213, 0x40060F56U},
    {22, ProbeStage::kCorrected, 131, 0x3F88E3E5U},
    {50, ProbeStage::kLatent, 0, 0xC0941EFAU},
    {50, ProbeStage::kLatent, kOfficialLatentCount - 1U, 0xC06F76FAU},
}};

uint32_t float_bits(float value) {
    uint32_t bits = 0U;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

void check_exact_probes(int32_t step, const std::vector<float>& corrected,
                        const std::vector<float>& latent) {
    for (const ExactProbe& probe : kExactProbes) {
        if (probe.step != step)
            continue;
        const std::vector<float>& values =
            probe.stage == ProbeStage::kCorrected ? corrected : latent;
        const uint32_t actual_bits = float_bits(values.at(probe.index));
        if (actual_bits != probe.expected_bits) {
            const std::string stage =
                probe.stage == ProbeStage::kCorrected ? "corrected" : "latent";
            throw std::runtime_error("official BF16-autocast " + stage + " probe differs at step " +
                                     std::to_string(step) + ", index " +
                                     std::to_string(probe.index) + ": expected bits " +
                                     std::to_string(probe.expected_bits) + ", got " +
                                     std::to_string(actual_bits));
        }
    }
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

float official_cfg(float conditional, float unconditional) {
    // Match the three eager CUDA expression boundaries without allowing the
    // host compiler to contract multiply + add into FMA.
    volatile float delta = conditional - unconditional;
    volatile float scaled = 5.0F * delta;
    volatile float guided = unconditional + scaled;
    return guided;
}

} // namespace

int main(int argc, char** argv) {
    const bool stream_full = argc == 2 && std::string_view(argv[1]) == "--stream-full";
    const bool stream_stages = argc == 2 && std::string_view(argv[1]) == "--stream-stages";
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::cerr << "FAIL: could not create CUDA stream\n";
        return 1;
    }

    try {
        trtmc::wan2_2_ti2v::FlowUniPCCuda scheduler(stream, 50, 5.0F, 1000);
        const std::vector<int64_t> expected_timesteps = {
            999, 995, 991, 987, 982, 978, 973, 968, 963, 957, 952, 946, 940, 934, 927, 920, 913,
            906, 898, 890, 882, 873, 863, 854, 843, 833, 821, 809, 796, 783, 768, 753, 737, 720,
            701, 681, 660, 636, 611, 584, 555, 522, 487, 448, 405, 356, 302, 241, 172, 92,
        };
        if (scheduler.timesteps() != expected_timesteps) {
            std::cerr << "FAIL: CUDA UniPC timesteps differ from the official profile\n";
            cudaStreamDestroy(stream);
            return 1;
        }

        std::vector<float> sample(kOfficialLatentCount);
        std::vector<float> conditional(kOfficialLatentCount);
        std::vector<float> unconditional(kOfficialLatentCount);
        std::vector<float> guided(kOfficialLatentCount);
        std::vector<float> corrected(kOfficialLatentCount);
        std::vector<float> next(kOfficialLatentCount);
        fill_initial_sample(sample);
        for (int32_t step = 0; step < 50; ++step) {
            fill_model_outputs(conditional, unconditional, step);
            for (std::size_t index = 0; index < guided.size(); ++index)
                guided[index] = official_cfg(conditional[index], unconditional[index]);
            if (stream_full || stream_stages) {
                std::cout.write(reinterpret_cast<const char*>(guided.data()),
                                static_cast<std::streamsize>(guided.size() * sizeof(float)));
            }
            scheduler.step(guided.data(), sample.data(), next.data(), sample.size(),
                           corrected.data());
            sample.swap(next);
            check_exact_probes(step + 1, corrected, sample);
            if (stream_stages) {
                std::cout.write(reinterpret_cast<const char*>(corrected.data()),
                                static_cast<std::streamsize>(corrected.size() * sizeof(float)));
            }
            if (stream_full || stream_stages) {
                std::cout.write(reinterpret_cast<const char*>(sample.data()),
                                static_cast<std::streamsize>(sample.size() * sizeof(float)));
                if (!std::cout)
                    throw std::runtime_error("could not stream CUDA UniPC fixture");
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        cudaStreamDestroy(stream);
        return 1;
    }

    if (cudaStreamDestroy(stream) != cudaSuccess) {
        std::cerr << "FAIL: could not destroy CUDA stream\n";
        return 1;
    }
    return 0;
}
