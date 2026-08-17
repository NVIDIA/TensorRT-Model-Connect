/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sam2_golden_fixture.h"

#include "runtime/models/sam2/sam2_engine_contract.h"
#include "tools/sam2_native_benchmark/sam2_benchmark_protocol.h"
#include "utils/sha256.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>

namespace trtmc::sam2::test {
namespace {

constexpr std::size_t kFramePixels =
    static_cast<std::size_t>(kOriginalImageHeight) * kOriginalImageWidth;
constexpr std::size_t kMaskElements = static_cast<std::size_t>(kFrameCount) * kFramePixels;
constexpr std::size_t kPackedBytes = kMaskElements / 8;
constexpr std::string_view kPackedSha256 =
    "1c7830b37739e409fbb8dab2b81c31c63b3379e6c10ae9e6b4ca2cc48a656094";
constexpr std::string_view kLogicalSha256 =
    "cf2a8ead5c2526a20d7b075ccfbc45043c9699569add1790e266d07c392faa4a";
constexpr std::array<std::string_view, 5> kFrameSha256 = {
    "03b13d0841f527e1de8445828babf3c818f15db266209359f2b3f1f764471aa2",
    "91ad1876ab6d1ce579b63fd5d02b2e165c71423d501e26bf226addfa4f7191e3",
    "f9c4e4e69c1c77b2f10ce9983fea957e7d5724221b070725a7feec4699c6ad2c",
    "116f1186181d5828bc1847ac51a1c14fd277c11f1ed5065583fdda567a1d5adc",
    "447c5352c62b310dd5547226869971c3f92f10a17a7e176ce8246c22b1a2c213",
};
constexpr std::array<std::uint64_t, 5> kForegroundPixels = {3600, 3420, 4071, 3770, 3603};

std::vector<std::uint8_t> readBytes(const std::filesystem::path& path) {
    if (std::filesystem::is_symlink(path) || !std::filesystem::is_regular_file(path))
        throw std::runtime_error("SAM2 golden file must be regular: " + path.string());
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
        throw std::runtime_error("unable to open SAM2 golden file: " + path.string());
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::string sha256(const void* data, std::size_t size) {
    internal::Sha256 hash;
    hash.update(data, size);
    return hash.hex_digest();
}

std::string sha256(const std::vector<std::uint8_t>& bytes) {
    return sha256(bytes.data(), bytes.size());
}

template <std::size_t Size>
std::array<float, Size> floatArray(const nlohmann::json& value, std::string_view label) {
    if (!value.is_array() || value.size() != Size)
        throw std::runtime_error(std::string(label) + " has the wrong shape");
    std::array<float, Size> result{};
    for (std::size_t index = 0; index < Size; ++index) {
        if (!value[index].is_number())
            throw std::runtime_error(std::string(label) + " contains a non-number");
        result[index] = value[index].get<float>();
        if (!std::isfinite(result[index]))
            throw std::runtime_error(std::string(label) + " contains a non-finite value");
    }
    return result;
}

double boxIou(const std::array<float, 4>& left, const std::array<float, 4>& right) {
    const double width = std::max(
        0.0, static_cast<double>(std::min(left[2], right[2]) - std::max(left[0], right[0])));
    const double height = std::max(
        0.0, static_cast<double>(std::min(left[3], right[3]) - std::max(left[1], right[1])));
    const double intersection = width * height;
    const double left_area = static_cast<double>(left[2] - left[0]) * (left[3] - left[1]);
    const double right_area = static_cast<double>(right[2] - right[0]) * (right[3] - right[1]);
    const double union_area = left_area + right_area - intersection;
    return union_area > 0.0 ? intersection / union_area : 0.0;
}

} // namespace

bool MaskAccuracy::passes() const {
    return std::all_of(frame_iou.begin(), frame_iou.end(),
                       [](double value) { return value >= benchmark::kMinimumFrameMaskIou; }) &&
           macro_iou >= benchmark::kMinimumMacroMaskIou &&
           global_iou >= benchmark::kMinimumGlobalMaskIou;
}

bool BboxAccuracy::passes() const {
    return label_exact && iou >= benchmark::kMinimumBboxIou &&
           max_coordinate_error <= benchmark::kMaximumBboxCoordinateError &&
           score_error <= benchmark::kMaximumBboxScoreError;
}

GoldenFixture loadGoldenFixture(const std::filesystem::path& directory) {
    if (std::filesystem::is_symlink(directory) || !std::filesystem::is_directory(directory))
        throw std::runtime_error("SAM2 golden root must be a regular directory");
    const auto manifest_bytes = readBytes(directory / "manifest.json");
    if (sha256(manifest_bytes) != kGoldenManifestSha256)
        throw std::runtime_error("SAM2 golden manifest hash mismatch");
    const auto manifest = nlohmann::json::parse(manifest_bytes.begin(), manifest_bytes.end());
    if (manifest.value("artifact_type", "") != "sam2_exact_workload_golden_evidence" ||
        manifest.value("schema_version", 0) != 1 ||
        manifest.value("producer", "") != "compatible_source_pytorch_bf16")
        throw std::runtime_error("SAM2 golden manifest identity mismatch");
    if (manifest.at("qualification").value("status", "") != "authoritative_reference_candidate" ||
        manifest.at("determinism").value("run_count", 0) != 3 ||
        !manifest.at("determinism").value("all_exact", false))
        throw std::runtime_error("SAM2 golden determinism receipt mismatch");

    const auto& bbox_json = manifest.at("frame_zero_bbox");
    GoldenFixture fixture;
    fixture.bbox.model_xyxy = floatArray<4>(bbox_json.at("model_image_xyxy_1024"), "model box");
    fixture.bbox.original_xyxy = floatArray<4>(bbox_json.at("original_image_xyxy"), "original box");
    fixture.bbox.score = bbox_json.at("score").get<float>();
    fixture.bbox.label = bbox_json.at("label").get<std::int32_t>();
    if (!std::isfinite(fixture.bbox.score) || fixture.bbox.score < 0.0F ||
        fixture.bbox.score > 1.0F || fixture.bbox.label != 1)
        throw std::runtime_error("SAM2 golden bbox metadata mismatch");

    const auto packed = readBytes(directory / "masks.bitpack");
    if (packed.size() != kPackedBytes || sha256(packed) != kPackedSha256)
        throw std::runtime_error("SAM2 golden packed mask mismatch");
    fixture.masks.resize(kMaskElements);
    for (std::size_t index = 0; index < fixture.masks.size(); ++index)
        fixture.masks[index] = static_cast<std::uint8_t>((packed[index / 8] >> (index % 8)) & 1U);
    if (sha256(fixture.masks) != kLogicalSha256)
        throw std::runtime_error("SAM2 golden logical mask mismatch");

    fixture.foreground_pixels = kForegroundPixels;
    for (std::size_t frame = 0; frame < kFrameCount; ++frame) {
        const auto begin = fixture.masks.data() + frame * kFramePixels;
        const auto foreground = static_cast<std::uint64_t>(
            std::count(begin, begin + kFramePixels, static_cast<std::uint8_t>(1)));
        if (foreground != kForegroundPixels[frame] ||
            sha256(begin, kFramePixels) != kFrameSha256[frame])
            throw std::runtime_error("SAM2 golden per-frame mask mismatch");
    }
    return fixture;
}

MaskAccuracy compareMasks(const std::vector<std::uint8_t>& candidate,
                          const GoldenFixture& reference) {
    if (candidate.size() != reference.masks.size())
        throw std::invalid_argument("SAM2 candidate mask has the wrong size");
    MaskAccuracy result;
    std::uint64_t global_intersection = 0;
    std::uint64_t global_union = 0;
    for (std::size_t frame = 0; frame < kFrameCount; ++frame) {
        std::uint64_t intersection = 0;
        std::uint64_t union_pixels = 0;
        for (std::size_t index = 0; index < kFramePixels; ++index) {
            const auto candidate_value = candidate[frame * kFramePixels + index];
            if (candidate_value > 1)
                throw std::invalid_argument("SAM2 candidate mask is not binary");
            const auto reference_value = reference.masks[frame * kFramePixels + index];
            intersection += static_cast<std::uint64_t>(candidate_value & reference_value);
            union_pixels += static_cast<std::uint64_t>(candidate_value | reference_value);
        }
        result.frame_iou[frame] =
            union_pixels == 0 ? 1.0 : static_cast<double>(intersection) / union_pixels;
        result.macro_iou += result.frame_iou[frame];
        global_intersection += intersection;
        global_union += union_pixels;
    }
    result.macro_iou /= kFrameCount;
    result.global_iou = global_union == 0 ? 1.0
                                          : static_cast<double>(global_intersection) /
                                                static_cast<double>(global_union);
    return result;
}

BboxAccuracy compareBbox(const GoldenBbox& candidate, const GoldenFixture& reference) {
    BboxAccuracy result;
    result.iou = boxIou(candidate.original_xyxy, reference.bbox.original_xyxy);
    for (std::size_t index = 0; index < candidate.original_xyxy.size(); ++index) {
        result.max_coordinate_error =
            std::max(result.max_coordinate_error,
                     std::abs(static_cast<double>(candidate.original_xyxy[index]) -
                              reference.bbox.original_xyxy[index]));
    }
    result.score_error = std::abs(static_cast<double>(candidate.score) - reference.bbox.score);
    result.label_exact = candidate.label == reference.bbox.label;
    return result;
}

} // namespace trtmc::sam2::test
