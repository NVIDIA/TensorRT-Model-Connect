/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sam2_benchmark_accuracy.h"

#include "runtime/models/sam2/sam2_engine_contract.h"
#include "utils/sha256.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace trtmc::sam2::benchmark {
namespace {

constexpr std::size_t kFramePixels =
    static_cast<std::size_t>(kOriginalImageHeight) * kOriginalImageWidth;
constexpr std::size_t kAllMaskPixels = static_cast<std::size_t>(kFrameCount) * kFramePixels;
constexpr std::size_t kPackedBytes = kAllMaskPixels / 8U;
constexpr std::string_view kPackedMaskSha256 =
    "1c7830b37739e409fbb8dab2b81c31c63b3379e6c10ae9e6b4ca2cc48a656094";
constexpr std::string_view kLogicalMaskSha256 =
    "cf2a8ead5c2526a20d7b075ccfbc45043c9699569add1790e266d07c392faa4a";
constexpr std::array<std::uint64_t, 5> kForegroundPixels = {3600, 3420, 4071, 3770, 3603};

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error("SAM2 benchmark accuracy: " + message);
}

bool sameIdentity(const struct stat& left, const struct stat& right) noexcept {
    return left.st_dev == right.st_dev && left.st_ino == right.st_ino &&
           left.st_size == right.st_size && left.st_mtim.tv_sec == right.st_mtim.tv_sec &&
           left.st_mtim.tv_nsec == right.st_mtim.tv_nsec &&
           left.st_ctim.tv_sec == right.st_ctim.tv_sec &&
           left.st_ctim.tv_nsec == right.st_ctim.tv_nsec;
}

std::vector<std::uint8_t> readRegularFile(const std::filesystem::path& path,
                                          std::size_t maximum_bytes) {
    int flags = O_RDONLY | O_CLOEXEC;
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#endif
    const int descriptor = ::open(path.c_str(), flags);
    if (descriptor < 0)
        fail("unable to open input: " + path.string() + ": " + std::strerror(errno));
    struct Closer {
        int descriptor;
        ~Closer() { (void)::close(descriptor); }
    } closer{descriptor};
    struct stat before{};
    if (::fstat(descriptor, &before) != 0 || !S_ISREG(before.st_mode) || before.st_size <= 0 ||
        static_cast<std::uint64_t>(before.st_size) > maximum_bytes) {
        fail("input must be a bounded, nonempty, non-symlink regular file: " + path.string());
    }
    std::vector<std::uint8_t> result(static_cast<std::size_t>(before.st_size));
    std::size_t offset = 0U;
    while (offset != result.size()) {
        ssize_t count = -1;
        do {
            count = ::pread(descriptor, result.data() + offset, result.size() - offset,
                            static_cast<off_t>(offset));
        } while (count < 0 && errno == EINTR);
        if (count <= 0)
            fail("short read while snapshotting input: " + path.string());
        offset += static_cast<std::size_t>(count);
    }
    struct stat after{};
    if (::fstat(descriptor, &after) != 0 || !sameIdentity(before, after))
        fail("input changed while it was snapshotted: " + path.string());
    return result;
}

std::string sha256(const void* data, std::size_t size) {
    internal::Sha256 hash;
    hash.update(data, size);
    return hash.hex_digest();
}

std::string sha256(const std::vector<std::uint8_t>& bytes) {
    return sha256(bytes.data(), bytes.size());
}

std::array<float, 4> floatBox(const nlohmann::json& value) {
    if (!value.is_array() || value.size() != 4U)
        fail("golden bbox has the wrong shape");
    std::array<float, 4> result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        if (!value[index].is_number())
            fail("golden bbox contains a non-number");
        result[index] = value[index].get<float>();
        if (!std::isfinite(result[index]))
            fail("golden bbox contains a non-finite value");
    }
    return result;
}

double bboxIou(const std::array<float, 4>& left, const std::array<float, 4>& right) {
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

void appendU32(std::vector<std::uint8_t>& bytes, std::uint32_t value) {
    for (unsigned int shift = 0U; shift < 32U; shift += 8U)
        bytes.push_back(static_cast<std::uint8_t>(value >> shift));
}

std::string bboxHash(const Sam2VideoTrack& track) {
    std::vector<std::uint8_t> canonical;
    canonical.reserve(6U * sizeof(std::uint32_t));
    appendU32(canonical, static_cast<std::uint32_t>(track.label));
    std::uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(track.detector_score));
    std::memcpy(&bits, &track.detector_score, sizeof(bits));
    appendU32(canonical, bits);
    for (const auto coordinate : track.prompt_box_xyxy) {
        std::memcpy(&bits, &coordinate, sizeof(bits));
        appendU32(canonical, bits);
    }
    return sha256(canonical);
}

} // namespace

GoldenEvidence loadGoldenEvidence(const std::filesystem::path& directory) {
    if (std::filesystem::is_symlink(directory) || !std::filesystem::is_directory(directory))
        fail("golden root must be a non-symlink directory");
    const auto manifest_bytes = readRegularFile(directory / "manifest.json", 1024U * 1024U);
    if (sha256(manifest_bytes) != kGoldenManifestSha256)
        fail("golden manifest hash mismatch");
    const auto manifest = nlohmann::json::parse(manifest_bytes.begin(), manifest_bytes.end());
    if (manifest.value("artifact_type", "") != "sam2_exact_workload_golden_evidence" ||
        manifest.value("schema_version", 0) != 1 ||
        manifest.value("producer", "") != "compatible_source_pytorch_bf16" ||
        manifest.at("qualification").value("status", "") != "authoritative_reference_candidate" ||
        manifest.at("determinism").value("run_count", 0) != 3 ||
        !manifest.at("determinism").value("all_exact", false)) {
        fail("golden manifest identity or determinism receipt mismatch");
    }

    const auto& bbox = manifest.at("frame_zero_bbox");
    GoldenEvidence result;
    result.bbox_original_xyxy = floatBox(bbox.at("original_image_xyxy"));
    result.bbox_score = bbox.at("score").get<float>();
    result.bbox_label = bbox.at("label").get<std::int32_t>();
    if (!std::isfinite(result.bbox_score) || result.bbox_score < 0.0F || result.bbox_score > 1.0F ||
        result.bbox_label != 1) {
        fail("golden bbox metadata mismatch");
    }

    const auto packed = readRegularFile(directory / "masks.bitpack", kPackedBytes);
    if (packed.size() != kPackedBytes || sha256(packed) != kPackedMaskSha256)
        fail("golden packed mask mismatch");
    result.masks.resize(kAllMaskPixels);
    for (std::size_t index = 0; index < result.masks.size(); ++index)
        result.masks[index] = static_cast<std::uint8_t>((packed[index / 8U] >> (index % 8U)) & 1U);
    if (sha256(result.masks) != kLogicalMaskSha256)
        fail("golden logical mask mismatch");
    result.foreground_pixels = kForegroundPixels;
    for (std::size_t frame = 0; frame < kForegroundPixels.size(); ++frame) {
        const auto begin = result.masks.begin() + static_cast<std::ptrdiff_t>(frame * kFramePixels);
        if (static_cast<std::uint64_t>(
                std::count(begin, begin + static_cast<std::ptrdiff_t>(kFramePixels),
                           static_cast<std::uint8_t>(1))) != kForegroundPixels[frame]) {
            fail("golden foreground count mismatch");
        }
    }
    return result;
}

AccuracyReplay evaluateAccuracy(std::size_t replay_index, const Sam2VideoPromptResult& prompt,
                                Sam2VideoFrameResults& results, const GoldenEvidence& golden) {
    if (results.size() != static_cast<std::size_t>(kFrameCount))
        fail("candidate did not return exactly five frame results");
    AccuracyReplay result;
    result.index = replay_index;
    std::vector<std::uint8_t> candidate;
    candidate.reserve(kAllMaskPixels);
    std::uint64_t global_intersection = 0U;
    std::uint64_t global_union = 0U;

    for (std::size_t frame = 0; frame < results.size(); ++frame) {
        auto& frame_result = results[frame];
        if (frame_result.frame_index != static_cast<std::int32_t>(frame) ||
            frame_result.height != kOriginalImageHeight ||
            frame_result.width != kOriginalImageWidth)
            fail("candidate frame geometry drifted");
        const auto& mask = frame_result.mask.materialize_host(kFramePixels);
        std::uint64_t intersection = 0U;
        std::uint64_t union_pixels = 0U;
        std::uint64_t foreground = 0U;
        for (std::size_t index = 0; index < mask.size(); ++index) {
            const auto value = mask[index];
            if (value > 1U)
                fail("candidate mask is not binary");
            const auto reference = golden.masks[frame * kFramePixels + index];
            intersection += static_cast<std::uint64_t>(value & reference);
            union_pixels += static_cast<std::uint64_t>(value | reference);
            foreground += value;
        }
        result.foreground_pixels[frame] = foreground;
        result.frame_iou[frame] =
            union_pixels == 0U ? 1.0 : static_cast<double>(intersection) / union_pixels;
        result.macro_iou += result.frame_iou[frame];
        global_intersection += intersection;
        global_union += union_pixels;
        candidate.insert(candidate.end(), mask.begin(), mask.end());
    }
    result.macro_iou /= results.size();
    result.global_iou = global_union == 0U ? 1.0
                                           : static_cast<double>(global_intersection) /
                                                 static_cast<double>(global_union);
    result.mask_sha256 = sha256(candidate);
    result.bbox_sha256 = bboxHash(prompt.track);
    result.candidate_bbox_label = prompt.track.label;
    result.candidate_bbox_score = prompt.track.detector_score;
    result.reference_bbox_label = golden.bbox_label;
    result.reference_bbox_score = golden.bbox_score;
    for (std::size_t index = 0; index < prompt.track.prompt_box_xyxy.size(); ++index) {
        result.candidate_bbox_xyxy[index] = prompt.track.prompt_box_xyxy[index];
        result.reference_bbox_xyxy[index] = golden.bbox_original_xyxy[index];
    }
    result.bbox_iou = bboxIou(prompt.track.prompt_box_xyxy, golden.bbox_original_xyxy);
    for (std::size_t index = 0; index < prompt.track.prompt_box_xyxy.size(); ++index) {
        result.bbox_max_coordinate_error =
            std::max(result.bbox_max_coordinate_error,
                     std::abs(static_cast<double>(prompt.track.prompt_box_xyxy[index]) -
                              golden.bbox_original_xyxy[index]));
    }
    result.bbox_score_error =
        std::abs(static_cast<double>(prompt.track.detector_score) - golden.bbox_score);
    result.bbox_label_exact = prompt.track.label == golden.bbox_label;
    result.passes = std::all_of(result.frame_iou.begin(), result.frame_iou.end(),
                                [](double value) { return value >= kMinimumFrameMaskIou; }) &&
                    result.macro_iou >= kMinimumMacroMaskIou &&
                    result.global_iou >= kMinimumGlobalMaskIou &&
                    result.bbox_iou >= kMinimumBboxIou &&
                    result.bbox_max_coordinate_error <= kMaximumBboxCoordinateError &&
                    result.bbox_score_error <= kMaximumBboxScoreError && result.bbox_label_exact;
    if (!result.passes) {
        const nlohmann::ordered_json evidence = {
            {"replay_index", result.index},
            {"thresholds",
             {{"every_frame_iou_min", kMinimumFrameMaskIou},
              {"macro_iou_min", kMinimumMacroMaskIou},
              {"global_iou_min", kMinimumGlobalMaskIou},
              {"bbox_iou_min", kMinimumBboxIou},
              {"bbox_max_coordinate_error_max", kMaximumBboxCoordinateError},
              {"bbox_score_error_max", kMaximumBboxScoreError},
              {"bbox_label_exact", true}}},
            {"candidate",
             {{"mask_sha256", result.mask_sha256},
              {"bbox_sha256", result.bbox_sha256},
              {"foreground_pixels", result.foreground_pixels},
              {"frame_iou", result.frame_iou},
              {"macro_iou", result.macro_iou},
              {"global_iou", result.global_iou},
              {"bbox_iou", result.bbox_iou},
              {"bbox_max_coordinate_error", result.bbox_max_coordinate_error},
              {"bbox_score_error", result.bbox_score_error},
              {"bbox_label_exact", result.bbox_label_exact},
              {"bbox_label", result.candidate_bbox_label},
              {"bbox_score", result.candidate_bbox_score},
              {"bbox_xyxy", result.candidate_bbox_xyxy}}},
            {"reference",
             {{"bbox_label", result.reference_bbox_label},
              {"bbox_score", result.reference_bbox_score},
              {"bbox_xyxy", result.reference_bbox_xyxy}}},
        };
        fail("candidate failed a semantic accuracy gate: " + evidence.dump());
    }
    return result;
}

} // namespace trtmc::sam2::benchmark
