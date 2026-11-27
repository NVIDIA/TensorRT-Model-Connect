/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/esmfold2/runtime/random.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <thread>

namespace trtmc::esmfold2 {
namespace {

constexpr std::uint64_t kGamma = 0x9E3779B97F4A7C15ULL;
constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr double kTwoPow53Inverse = 1.0 / 9007199254740992.0;
constexpr int kTruncationCandidates = 8;
constexpr double kTruncation = 3.0;

std::uint64_t streamKey(std::uint64_t seed, Stream stream) {
    return splitmix64_mix(splitmix64_mix(seed) + static_cast<std::uint64_t>(stream));
}

} // namespace

std::uint64_t splitmix64_mix(std::uint64_t value) {
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}

std::uint64_t RandomStreams::bits(Stream stream, std::uint64_t index) const {
    return splitmix64_mix(streamKey(seed_, stream) + (index + 1U) * kGamma);
}

float RandomStreams::normal(Stream stream, std::uint64_t index) const {
    const std::uint64_t first = bits(stream, 2U * index);
    const std::uint64_t second = bits(stream, 2U * index + 1U);
    const double u1 = static_cast<double>((first >> 11U) + 1U) * kTwoPow53Inverse; // (0, 1]
    const double u2 = static_cast<double>(second >> 11U) * kTwoPow53Inverse;       // [0, 1)
    return static_cast<float>(std::sqrt(-2.0 * std::log(u1)) * std::cos(kTwoPi * u2));
}

float RandomStreams::truncated_normal(Stream stream, std::uint64_t index, float std) const {
    float value = 0.0F;
    for (int candidate = 0; candidate < kTruncationCandidates; ++candidate) {
        value =
            normal(stream, index * kTruncationCandidates + static_cast<std::uint64_t>(candidate));
        if (std::fabs(static_cast<double>(value)) <= kTruncation)
            return value * std;
    }
    // Eight candidates beyond three standard deviations have probability below 1e-20.
    return static_cast<float>(std::clamp(static_cast<double>(value), -kTruncation, kTruncation)) *
           std;
}

bool RandomStreams::keep(Stream stream, std::uint64_t index, std::uint32_t keep_threshold) const {
    return static_cast<std::uint32_t>(bits(stream, index) >> 40U) < keep_threshold;
}

void RandomStreams::fill_normal(Stream stream, std::uint64_t first, std::vector<float>& out) const {
    run_in_chunks(out.size(), chunk_workers(out.size()), [&](std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i)
            out[i] = normal(stream, first + i);
    });
}

std::size_t chunk_workers(std::size_t count) {
    const unsigned hardware = std::max(1U, std::thread::hardware_concurrency());
    return count < (std::size_t{1} << 16U) ? 1U : std::min<std::size_t>(hardware, 32U);
}

void run_in_chunks(std::size_t count, std::size_t workers, const ChunkWork& work,
                   const ChunkLauncher& launch) {
    if (workers <= 1) {
        work(0, count);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(workers);
    auto join = [&threads] {
        for (auto& thread : threads)
            if (thread.joinable())
                thread.join();
    };
    const std::size_t chunk = (count + workers - 1) / workers;
    try {
        for (std::size_t worker = 0; worker < workers; ++worker) {
            const std::size_t begin = worker * chunk;
            const std::size_t end = std::min(count, begin + chunk);
            if (begin < end)
                threads.push_back(launch ? launch(work, begin, end)
                                         : std::thread(work, begin, end));
        }
    } catch (...) {
        join();
        throw;
    }
    join();
}

std::uint32_t keep_threshold(double dropout) {
    if (!(dropout >= 0.0 && dropout < 1.0))
        throw std::invalid_argument("ESMFold2 LM dropout must lie in [0, 1)");
    return static_cast<std::uint32_t>(std::llround((1.0 - dropout) * 16777216.0));
}

} // namespace trtmc::esmfold2
