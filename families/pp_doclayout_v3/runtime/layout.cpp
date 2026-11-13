/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/pp_doclayout_v3/runtime/layout.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace trtmc::pp_doclayout_v3 {
namespace {

double cubic(double x) {
    x = std::abs(x);
    if (x < 1)
        return ((1.25 * x - 2.25) * x) * x + 1;
    if (x < 2)
        return ((-0.75 * x + 3.75) * x - 6) * x + 3;
    return 0;
}

struct Filter {
    int start;
    std::vector<int> coefficients;
    int precision;
};

// Match the official uint8 resize: no antialiasing, a=-0.75, signed-16-bit
// coefficients, and uint8 rounding between the horizontal and vertical passes.
std::vector<Filter> filters(int input, int output) {
    const double scale = static_cast<double>(input) / output;
    std::vector<Filter> result;
    std::vector<std::vector<double>> floating;
    double maximum = 0;
    for (int i = 0; i < output; ++i) {
        const double centre = (i + 0.5) * scale - 0.5;
        const int unbounded = static_cast<int>(std::floor(centre)) - 1;
        const int start = std::max(0, unbounded);
        const int end = std::min(input, unbounded + 4);
        std::vector<double> weights(end - start, 0);
        for (int j = unbounded; j < unbounded + 4; ++j) {
            auto& weight = weights[std::clamp(j, 0, input - 1) - start];
            weight += cubic(j - centre);
            maximum = std::max(maximum, weight);
        }
        result.push_back({start, {}, 0});
        floating.push_back(std::move(weights));
    }
    int precision = 0;
    while (precision < 22 && static_cast<int>(0.5 + maximum * (1 << (precision + 1))) < (1 << 15))
        ++precision;
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i].precision = precision;
        for (double weight : floating[i])
            result[i].coefficients.push_back(
                static_cast<int>(std::round(weight * (1 << precision))));
    }
    return result;
}

float sigmoid(float x) {
    return x >= 0 ? 1.0F / (1.0F + std::exp(-x)) : std::exp(x) / (1.0F + std::exp(x));
}

std::vector<std::array<float, 2>> polygon(const DetectionBox& box, const float* mask, int mask_size,
                                          int input_size, int height, int width, float threshold) {
    const int x0 = static_cast<int>(box.x_min), y0 = static_cast<int>(box.y_min);
    const int x1 = static_cast<int>(box.x_max), y1 = static_cast<int>(box.y_max);
    const std::vector<std::array<float, 2>> rectangle{
        {static_cast<float>(x0), static_cast<float>(y0)},
        {static_cast<float>(x1), static_cast<float>(y0)},
        {static_cast<float>(x1), static_cast<float>(y1)},
        {static_cast<float>(x0), static_cast<float>(y1)}};
    if (x1 <= x0 || y1 <= y0)
        return rectangle;
    // Python round uses ties-to-even. Crop in mask coordinates before resizing.
    auto coordinate = [&](int x, int extent) {
        return std::clamp(
            static_cast<int>(std::nearbyint(x * (static_cast<double>(input_size) / extent / 4))), 0,
            mask_size);
    };
    const int left = coordinate(x0, width), right = coordinate(x1, width);
    const int top = coordinate(y0, height), bottom = coordinate(y1, height);
    if (right <= left || bottom <= top)
        return rectangle;
    cv::Mat crop(bottom - top, right - left, CV_8U);
    for (int row = top; row < bottom; ++row)
        for (int column = left; column < right; ++column)
            crop.at<unsigned char>(row - top, column - left) =
                sigmoid(mask[row * mask_size + column]) > threshold ? 1 : 0;
    if (cv::countNonZero(crop) == 0)
        return rectangle;
    cv::Mat resized;
    cv::resize(crop, resized, cv::Size(x1 - x0, y1 - y0), 0, 0, cv::INTER_NEAREST);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(resized, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty())
        return rectangle;
    const auto& contour =
        *std::max_element(contours.begin(), contours.end(), [](const auto& a, const auto& b) {
            return cv::contourArea(a) < cv::contourArea(b);
        });
    std::vector<cv::Point> vertices;
    cv::approxPolyDP(contour, vertices, 0.004 * cv::arcLength(contour, true), true);
    std::vector<std::array<float, 2>> result;
    for (std::size_t i = 0; i < vertices.size(); ++i) {
        const cv::Point2d current = vertices[i];
        const cv::Point2d a =
            cv::Point2d(vertices[(i + vertices.size() - 1) % vertices.size()]) - current;
        const cv::Point2d b = cv::Point2d(vertices[(i + 1) % vertices.size()]) - current;
        if (a.y * b.x - a.x * b.y >= 0)
            continue;
        const double an = cv::norm(a), bn = cv::norm(b);
        const double angle =
            std::acos(std::clamp(a.dot(b) / (an * bn), -1.0, 1.0)) * 180 / std::acos(-1.0);
        cv::Point2d vertex = current;
        if (std::abs(angle - 45) < 1) {
            const cv::Point2d direction = a / an + b / bn;
            vertex += direction / cv::norm(direction) * ((an + bn) / 2);
        }
        result.push_back({static_cast<float>(vertex.x + x0), static_cast<float>(vertex.y + y0)});
    }
    return result.size() >= 4 ? result : rectangle;
}

} // namespace

std::vector<float> preprocess(const float* pixels, int height, int width, int size) {
    if (!pixels || height <= 0 || width <= 0 || size <= 0)
        throw std::invalid_argument("pp_doclayout_v3 requires a nonempty RGB image");
    const auto fx = filters(width, size), fy = filters(height, size);
    std::vector<unsigned char> horizontal(static_cast<std::size_t>(height) * size * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < size; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                const auto& filter = fx[x];
                std::int64_t sum = 1 << (filter.precision - 1);
                for (std::size_t k = 0; k < filter.coefficients.size(); ++k) {
                    const float value =
                        pixels[(static_cast<std::size_t>(y) * width +
                                std::clamp(filter.start + static_cast<int>(k), 0, width - 1)) *
                                   3 +
                               channel];
                    if (!std::isfinite(value) || value < 0 || value > 1)
                        throw std::invalid_argument(
                            "pp_doclayout_v3 RGB pixels must be finite and in [0, 1]");
                    sum += static_cast<int>(std::round(value * 255)) *
                           static_cast<std::int64_t>(filter.coefficients[k]);
                }
                horizontal[(static_cast<std::size_t>(y) * size + x) * 3 + channel] =
                    static_cast<unsigned char>(
                        std::clamp<std::int64_t>(sum >> filter.precision, 0, 255));
            }
        }
    }
    std::vector<float> output(static_cast<std::size_t>(size) * size * 3);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                const auto& filter = fy[y];
                std::int64_t sum = 1 << (filter.precision - 1);
                for (std::size_t k = 0; k < filter.coefficients.size(); ++k)
                    sum += horizontal[((static_cast<std::size_t>(std::clamp(
                                           filter.start + static_cast<int>(k), 0, height - 1))) *
                                           size +
                                       x) *
                                          3 +
                                      channel] *
                           static_cast<std::int64_t>(filter.coefficients[k]);
                output[(static_cast<std::size_t>(channel) * size + y) * size + x] =
                    static_cast<float>(std::clamp<std::int64_t>(sum >> filter.precision, 0, 255)) /
                    255.0F;
            }
        }
    }
    return output;
}

DocumentLayoutResult decode(const float* logits, const float* boxes, const float* order,
                            const float* masks, int queries, int labels, int mask_size,
                            int input_size, int height, int width, float threshold) {
    if (!logits || !boxes || !order || !masks || queries <= 0 || labels <= 0 || mask_size <= 0 ||
        height <= 0 || width <= 0 || input_size != mask_size * 4 || !std::isfinite(threshold) ||
        threshold <= 0 || threshold >= 1)
        throw std::invalid_argument("pp_doclayout_v3 invalid layout geometry or threshold");
    const auto require_finite = [](const float* data, std::size_t count) {
        if (!std::all_of(data, data + count, [](float v) { return std::isfinite(v); }))
            throw std::runtime_error("pp_doclayout_v3 engine returned non-finite outputs");
    };
    require_finite(logits, static_cast<std::size_t>(queries) * labels);
    require_finite(boxes, static_cast<std::size_t>(queries) * 4);
    require_finite(order, static_cast<std::size_t>(queries) * queries);
    require_finite(masks, static_cast<std::size_t>(queries) * mask_size * mask_size);
    std::vector<float> votes(queries, 0);
    for (int i = 0; i < queries; ++i)
        for (int j = i + 1; j < queries; ++j) {
            const float score = sigmoid(order[i * queries + j]);
            votes[j] += score;
            votes[i] += 1 - score;
        }
    std::vector<int> rank(queries), indices(static_cast<std::size_t>(queries) * labels);
    std::iota(rank.begin(), rank.end(), 0);
    std::stable_sort(rank.begin(), rank.end(), [&](int a, int b) { return votes[a] < votes[b]; });
    std::vector<int> sequence(queries);
    for (int i = 0; i < queries; ++i)
        sequence[rank[i]] = i;
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + queries, indices.end(), [&](int a, int b) {
        return logits[a] != logits[b] ? logits[a] > logits[b] : a < b;
    });
    indices.resize(queries);
    indices.erase(std::remove_if(indices.begin(), indices.end(),
                                 [&](int index) { return sigmoid(logits[index]) < threshold; }),
                  indices.end());
    std::stable_sort(indices.begin(), indices.end(),
                     [&](int a, int b) { return sequence[a / labels] < sequence[b / labels]; });
    DocumentLayoutResult result;
    result.image_height = height;
    result.image_width = width;
    for (int index : indices) {
        const int query = index / labels;
        const float* box = boxes + query * 4;
        DetectionBox detection{(box[0] - box[2] * 0.5F) * width, (box[1] - box[3] * 0.5F) * height,
                               (box[0] + box[2] * 0.5F) * width, (box[1] + box[3] * 0.5F) * height,
                               sigmoid(logits[index]),           index % labels};
        // A normalized network box must not produce an unbounded contour allocation.
        if (box[0] < 0 || box[0] > 1 || box[1] < 0 || box[1] > 1 || box[2] < 0 || box[2] > 1 ||
            box[3] < 0 || box[3] > 1)
            throw std::runtime_error("pp_doclayout_v3 engine returned an invalid normalized box");
        result.regions.push_back(
            {detection,
             polygon(detection, masks + static_cast<std::size_t>(query) * mask_size * mask_size,
                     mask_size, input_size, height, width, threshold)});
    }
    return result;
}

} // namespace trtmc::pp_doclayout_v3
