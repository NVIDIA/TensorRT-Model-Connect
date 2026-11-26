/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace trtmc::esmfold2 {

// The random draws of one fold. Every draw is a pure function of (seed, stream, index): value i of
// a stream is SplitMix64 output i of a generator keyed by the seed and the stream, so any element
// can be produced on its own and the family's tests reproduce the same values for the reference
// model.
enum class Stream : std::uint64_t {
    kTrunkInit = 1,   // truncated-normal initial pair state, element index
    kLmDropout = 2,   // per-loop LM dropout keep mask, loop * elements + element
    kCoordinates = 3, // initial diffusion coordinates, (sample * atoms + atom) * 3 + axis
    kRotation = 4,    // per-step rotation quaternions, (step * samples + sample) * 4 + component
    kTranslation = 5, // per-step translations, (step * samples + sample) * 3 + axis
    kChurn =
        6, // per-step churn noise, step * samples * atoms * 3 + (sample * atoms + atom) * 3 + axis
};

class RandomStreams {
  public:
    explicit RandomStreams(std::uint64_t seed) : seed_(seed) {}

    std::uint64_t seed() const { return seed_; }
    // Raw 64-bit output ``index`` of ``stream``.
    std::uint64_t bits(Stream stream, std::uint64_t index) const;
    // Standard normal from outputs 2 * index and 2 * index + 1 (Box-Muller, cosine branch, double
    // arithmetic).
    float normal(Stream stream, std::uint64_t index) const;
    // Normal truncated to [-3, 3] times ``std``: the first of up to 8 candidates (index * 8 + k)
    // within the bounds.
    float truncated_normal(Stream stream, std::uint64_t index, float std) const;
    // ``true`` with probability keep_threshold / 2^24, from the top 24 bits of output ``index``.
    bool keep(Stream stream, std::uint64_t index, std::uint32_t keep_threshold) const;

    // Fill ``out`` with ``normal(stream, first + i)``, using worker threads for large ranges.
    void fill_normal(Stream stream, std::uint64_t first, std::vector<float>& out) const;

  private:
    std::uint64_t seed_;
};

// Keep threshold of a dropout probability, out of 2^24.
std::uint32_t keep_threshold(double dropout);

std::uint64_t splitmix64_mix(std::uint64_t value);

} // namespace trtmc::esmfold2
