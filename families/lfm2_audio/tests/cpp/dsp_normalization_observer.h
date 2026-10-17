/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

// Included only in a temporary, insertion-only copy compiled by the owning test.
namespace lfm2_audio_dsp_observation {
inline uint32_t word(float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}
struct ChannelValues {
    std::vector<uint32_t> first;
    bool identical{true};
    void observe(const std::vector<float>& values, int channel) {
        if (channel == 0) {
            for (float value : values)
                first.push_back(word(value));
        } else {
            if (values.size() != first.size()) {
                identical = false;
                return;
            }
            for (size_t i = 0; i < values.size(); ++i)
                identical = identical && first[i] == word(values[i]);
        }
    }
    void write(std::ostream& out, int source_stride = 1) const {
        size_t runs = 0;
        for (size_t i = 0; i < first.size(); ++i)
            runs += i == 0 || first[i] != first[i - 1];
        out << "{\"per_channel_shape\":[" << first.size() << "],\"per_channel_source_stride\":["
            << source_stride
            << "],\"all_channels_bit_identical\":" << (identical ? "true" : "false");
        if (!identical || runs > 32) {
            out << ",\"payload\":\"bounded_unavailable\"}";
            return;
        }
        out << ",\"payload\":\"lossless_first_channel_rle_repeated\",\"rle\":[";
        bool comma = false;
        for (size_t start = 0; start < first.size();) {
            size_t end = start + 1;
            while (end < first.size() && first[end] == first[start])
                ++end;
            if (comma)
                out << ',';
            out << "{\"start\":" << start << ",\"length\":" << end - start
                << ",\"bits\":" << first[start] << '}';
            comma = true;
            start = end;
        }
        out << "]}";
    }
};
struct Capture {
    int frames{0}, valid{0}, channel{-1}, phase{0}, sums{0}, means{0}, sigmas{0};
    bool available{true};
    ChannelValues features, mean_input, variance_input;
    std::vector<uint32_t> mean_sums, variance_sums, mean_values, sigma_values;
};
inline Capture capture;
inline void start(int channel, const std::vector<float>& features, int frames, int valid) noexcept {
    try {
        auto& c = capture;
        if (!c.available)
            return;
        if (channel != c.channel + 1 || channel >= 128 || frames < 2 || frames > 8193 ||
            valid != frames - 1 || features.size() != static_cast<size_t>(frames) * 128 ||
            (channel && (c.phase != 2 || c.frames != frames || c.valid != valid))) {
            c.available = false;
            return;
        }
        c.channel = channel;
        c.frames = frames;
        c.valid = valid;
        c.phase = 0;
        std::vector<float> values(frames);
        for (int time = 0; time < frames; ++time)
            values[time] = features[time * 128 + channel];
        c.features.observe(values, channel);
    } catch (...) {
        capture.available = false;
    }
}
inline void sum(const std::vector<float>& values, float result) noexcept {
    try {
        auto& c = capture;
        if (!c.available)
            return;
        if (c.channel < 0 || c.phase > 1 || values.size() != static_cast<size_t>(c.frames)) {
            c.available = false;
            return;
        }
        (c.phase == 0 ? c.mean_input : c.variance_input).observe(values, c.channel);
        (c.phase == 0 ? c.mean_sums : c.variance_sums).push_back(word(result));
        ++c.phase;
        ++c.sums;
    } catch (...) {
        capture.available = false;
    }
}
inline void mean(int channel, float value) noexcept {
    try {
        if (capture.channel != channel || capture.phase != 1)
            capture.available = false;
        capture.mean_values.push_back(word(value));
        ++capture.means;
    } catch (...) {
        capture.available = false;
    }
}
inline void sigma(int channel, float value) noexcept {
    try {
        if (capture.channel != channel || capture.phase != 2)
            capture.available = false;
        capture.sigma_values.push_back(word(value));
        ++capture.sigmas;
    } catch (...) {
        capture.available = false;
    }
}
inline void words(std::ostream& out, const std::vector<uint32_t>& values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i)
            out << ',';
        out << values[i];
    }
    out << ']';
}
inline void finish() noexcept {
    try {
        const char* path = std::getenv("TRTMC_LFM2_AUDIO_DSP_OBSERVATION");
        if (!path)
            return;
        std::ofstream out(path);
        auto& c = capture;
        const bool complete =
            c.available && c.channel == 127 && c.sums == 256 && c.means == 128 && c.sigmas == 128;
        out << "{\"schema_version\":1,\"complete\":" << (complete ? "true" : "false")
            << ",\"frames\":" << c.frames << ",\"valid_frames\":" << c.valid
            << ",\"sum_calls\":" << c.sums << ",\"mean_locals\":" << c.means
            << ",\"sigma_locals\":" << c.sigmas << ",\"features_before_normalization\":";
        c.features.write(out, 128);
        out << ",\"mean_reduction_input\":";
        c.mean_input.write(out);
        out << ",\"variance_reduction_input\":";
        c.variance_input.write(out);
        out << ",\"mean_sum_words\":";
        words(out, c.mean_sums);
        out << ",\"variance_sum_words\":";
        words(out, c.variance_sums);
        out << ",\"mean_words\":";
        words(out, c.mean_values);
        out << ",\"sigma_after_epsilon_words\":";
        words(out, c.sigma_values);
        out << "}\n";
    } catch (...) {
        // Diagnostics must not change the source function's return or failure.
    }
}
} // namespace lfm2_audio_dsp_observation
