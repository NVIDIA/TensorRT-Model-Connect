/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "visual_frontend.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace trtmc::nemotron_omni {
namespace {
size_t product(int first, int second, int third = 1) {
    if (first <= 0 || second <= 0 || third <= 0 ||
        static_cast<size_t>(first) > std::numeric_limits<size_t>::max() /
                                         static_cast<size_t>(second) / static_cast<size_t>(third))
        throw std::invalid_argument("Omni visual dimensions must be positive and representable");
    return static_cast<size_t>(first) * second * third;
}
void finite(const std::vector<float>& values) {
    if (std::any_of(values.begin(), values.end(), [](float x) { return !std::isfinite(x); }))
        throw std::invalid_argument("Omni visual values must be finite");
}
float cubic(float value) {
    value = std::abs(value);
    if (value < 1.f)
        return ((1.5f * value - 2.5f) * value) * value + 1.f;
    if (value < 2.f)
        return ((-.5f * value + 2.5f) * value - 4.f) * value + 2.f;
    return 0.f;
}
struct Filter {
    int start;
    std::vector<float> weights;
};
std::vector<Filter> filters(int source, int destination) {
    const float scale = static_cast<float>(source) / destination;
    const float support = 2.f * std::max(scale, 1.f);
    const float inverse = 1.f / std::max(scale, 1.f);
    std::vector<Filter> result;
    result.reserve(destination);
    for (int index = 0; index < destination; ++index) {
        const float center = scale * (index + .5f);
        const int start = std::max(static_cast<int>(center - support + .5f), 0);
        const int end = std::min(static_cast<int>(center + support + .5f), source);
        Filter row{start, {}};
        float total = 0;
        for (int sample = start; sample < end; ++sample) {
            const float weight = cubic((sample - center + .5f) * inverse);
            row.weights.push_back(weight);
            total += weight;
        }
        if (total == 0)
            throw std::runtime_error("Omni resize has an empty interpolation filter");
        for (auto& weight : row.weights)
            weight /= total;
        result.push_back(std::move(row));
    }
    return result;
}
} // namespace

std::vector<float> resize_rgb(const std::vector<float>& pixels, int height, int width,
                              int target_height, int target_width) {
    if (pixels.size() != product(height, width, 3))
        throw std::invalid_argument("Omni visual RGB size does not match its dimensions");
    finite(pixels);
    if (std::any_of(pixels.begin(), pixels.end(), [](float x) { return x < 0.f || x > 1.f; }))
        throw std::invalid_argument("Omni visual RGB input must lie within [0, 1]");
    std::vector<float> temporary(product(height, target_width, 3));
    const auto horizontal = filters(width, target_width);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < target_width; ++x)
            for (int channel = 0; channel < 3; ++channel) {
                float value = 0;
                const auto& filter = horizontal[x];
                for (size_t index = 0; index < filter.weights.size(); ++index)
                    value += filter.weights[index] *
                             pixels[(size_t(y) * width + filter.start + index) * 3 + channel];
                temporary[(size_t(y) * target_width + x) * 3 + channel] = value;
            }
    std::vector<float> result(product(target_height, target_width, 3));
    const auto vertical = filters(height, target_height);
    for (int y = 0; y < target_height; ++y)
        for (int x = 0; x < target_width; ++x)
            for (int channel = 0; channel < 3; ++channel) {
                float value = 0;
                const auto& filter = vertical[y];
                for (size_t index = 0; index < filter.weights.size(); ++index)
                    value += filter.weights[index] *
                             temporary[((filter.start + index) * target_width + x) * 3 + channel];
                result[(size_t(y) * target_width + x) * 3 + channel] = value;
            }
    return result;
}

std::vector<float> patch_pixels(const std::vector<std::vector<float>>& frames, int height,
                                int width, int patch) {
    const auto frame_size = product(height, width, 3);
    if (frames.empty() || frames.size() > 2 || patch <= 0 || height % patch || width % patch)
        throw std::invalid_argument("Omni patch input requires one image or one two-frame tubelet");
    for (const auto& frame : frames) {
        if (frame.size() != frame_size)
            throw std::invalid_argument("Omni tubelet frames must have equal sizes");
        finite(frame);
    }
    constexpr float mean[] = {.48145466f, .4578275f, .40821073f};
    constexpr float deviation[] = {.26862954f, .26130258f, .27577711f};
    std::vector<float> result;
    result.reserve(frame_size * frames.size());
    for (int row = 0; row < height / patch; ++row)
        for (int column = 0; column < width / patch; ++column)
            for (const auto& frame : frames)
                for (int channel = 0; channel < 3; ++channel)
                    for (int y = 0; y < patch; ++y)
                        for (int x = 0; x < patch; ++x) {
                            const auto index =
                                (size_t(row * patch + y) * width + column * patch + x) * 3 +
                                channel;
                            result.push_back((frame[index] - mean[channel]) / deviation[channel]);
                        }
    return result;
}

std::vector<int32_t> shuffle_indices(int rows, int columns) {
    const auto count = product(rows, columns);
    if (rows % 2 || columns % 2 || count > size_t(std::numeric_limits<int32_t>::max()))
        throw std::invalid_argument("Omni visual grid requires complete representable 2x2 groups");
    std::vector<int32_t> result;
    result.reserve(count);
    for (int row = 0; row < rows; row += 2)
        for (int column = 0; column < columns; column += 2) {
            result.push_back(row * columns + column);
            result.push_back(row * columns + column + 1);
            result.push_back((row + 1) * columns + column);
            result.push_back((row + 1) * columns + column + 1);
        }
    return result;
}

std::vector<float> visual_positions(const std::vector<float>& table, int side, int hidden, int rows,
                                    int columns) {
    if (table.size() != product(side, side, hidden))
        throw std::invalid_argument("Omni learned position table has invalid dimensions");
    finite(table);
    std::vector<float> result(product(rows, columns, hidden));
    const int maximum = std::max(rows, columns);
    // Reference RADIO: square bilinear align_corners=True, then top-left crop.
    const float scale = maximum == 1 ? 0.f : float(side - 1) / (maximum - 1);
    for (int row = 0; row < rows; ++row)
        for (int column = 0; column < columns; ++column) {
            const float y = scale * row, x = scale * column;
            const int y0 = int(y), x0 = int(x);
            const int y1 = std::min(y0 + 1, side - 1), x1 = std::min(x0 + 1, side - 1);
            const float dy = y - y0, dx = x - x0;
            for (int channel = 0; channel < hidden; ++channel) {
                const float top = (1 - dx) * table[(size_t(y0) * side + x0) * hidden + channel] +
                                  dx * table[(size_t(y0) * side + x1) * hidden + channel];
                const float bottom = (1 - dx) * table[(size_t(y1) * side + x0) * hidden + channel] +
                                     dx * table[(size_t(y1) * side + x1) * hidden + channel];
                result[(size_t(row) * columns + column) * hidden + channel] =
                    (1 - dy) * top + dy * bottom;
            }
        }
    return result;
}

std::vector<int32_t> video_retained_indices(const std::vector<float>& features,
                                            int tokens_per_frame, int hidden, double pruning) {
    const size_t stride = product(tokens_per_frame, hidden);
    if (features.empty() || features.size() % stride || !std::isfinite(pruning) || pruning < 0 ||
        pruning > 1 || features.size() / hidden > size_t(std::numeric_limits<int32_t>::max()))
        throw std::invalid_argument("Omni EVS requires complete frames and pruning within [0, 1]");
    finite(features);
    const int count = static_cast<int>(features.size() / hidden);
    std::vector<float> scores(count, 255.f);
    for (int token = tokens_per_frame; token < count; ++token) {
        const float* current = features.data() + size_t(token) * hidden;
        const float* previous = current - stride;
        float current_norm = 0, previous_norm = 0;
        for (int channel = 0; channel < hidden; ++channel) {
            current_norm += current[channel] * current[channel];
            previous_norm += previous[channel] * previous[channel];
        }
        current_norm = std::max(std::sqrt(current_norm), 1e-8f);
        previous_norm = std::max(std::sqrt(previous_norm), 1e-8f);
        float similarity = 0;
        for (int channel = 0; channel < hidden; ++channel)
            similarity += (current[channel] / current_norm) * (previous[channel] / previous_norm);
        scores[token] = 1.f - similarity;
    }
    std::vector<int32_t> indices(count);
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(),
                     [&](int a, int b) { return scores[a] > scores[b]; });
    indices.resize(std::max(tokens_per_frame, static_cast<int>(count * (1. - pruning))));
    std::sort(indices.begin(), indices.end());
    return indices;
}
} // namespace trtmc::nemotron_omni
