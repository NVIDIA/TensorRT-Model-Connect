/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> identity_poses(int32_t count) {
    std::vector<float> poses(static_cast<std::size_t>(count) * 16, 0.0F);
    for (int32_t index = 0; index < count; ++index) {
        auto* pose = poses.data() + static_cast<std::size_t>(index) * 16;
        pose[0] = pose[5] = pose[10] = pose[15] = 1.0F;
    }
    return poses;
}

double percentile(std::vector<double> values, double fraction) {
    std::sort(values.begin(), values.end());
    const double position = fraction * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const double weight = position - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

void write_floats(const std::filesystem::path& path, const std::vector<float>& values) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!output)
        throw std::runtime_error("failed to write " + path.string());
}

trtmc::PoseCropBatch synthetic_crops(int32_t count, const std::vector<float>* poses = nullptr) {
    constexpr int32_t height = 160;
    constexpr int32_t width = 160;
    constexpr int32_t channels = 6;
    trtmc::PoseCropBatch result;
    result.num_hypotheses = count;
    result.height = height;
    result.width = width;
    result.channels = channels;
    const auto values = static_cast<std::size_t>(count) * height * width * channels;
    result.rendered_features.assign(values, 0.0F);
    result.observed_features.assign(values, 0.0F);
    for (int32_t hypothesis = 0; hypothesis < count; ++hypothesis) {
        const float pose_x =
            poses == nullptr ? 0.0F : poses->at(static_cast<std::size_t>(hypothesis) * 16 + 3);
        for (int32_t y = 0; y < height; ++y) {
            for (int32_t x = 0; x < width; ++x) {
                const float xn = (static_cast<float>(x) - 79.5F) / 80.0F;
                const float yn = (static_cast<float>(y) - 79.5F) / 80.0F;
                if (xn * xn + yn * yn > 0.72F)
                    continue;
                const auto base =
                    ((static_cast<std::size_t>(hypothesis) * height + y) * width + x) * channels;
                result.rendered_features[base] = 0.08F * (hypothesis + 1) + 0.4F * (x / 159.0F);
                result.rendered_features[base + 1] = 0.5F * (y / 159.0F);
                result.rendered_features[base + 2] = 0.35F;
                result.rendered_features[base + 3] = 0.2F * xn + pose_x;
                result.rendered_features[base + 4] = 0.2F * yn;
                result.rendered_features[base + 5] = 0.10F + 0.01F * hypothesis;
                result.observed_features[base] = 0.25F + 0.4F * (x / 159.0F);
                result.observed_features[base + 1] = 0.10F + 0.5F * (y / 159.0F);
                result.observed_features[base + 2] = 0.40F;
                result.observed_features[base + 3] = 0.2F * xn + 0.02F * hypothesis;
                result.observed_features[base + 4] = 0.2F * yn;
                result.observed_features[base + 5] = 0.11F;
            }
        }
    }
    return result;
}

int run_qualification(int argc, char** argv) {
    std::string bundle;
    std::string output_dir;
    std::string runtime_root;
    int benchmark_runs = 20;
    int warmup_runs = 3;
    int32_t num_hypotheses = 3;
    int32_t refinement_iterations = 2;
    float mesh_diameter = 0.18F;
    for (int index = 2; index < argc; ++index) {
        const std::string option = argv[index];
        if (++index >= argc)
            throw std::invalid_argument(option + " requires a value");
        const std::string value = argv[index];
        if (option == "--bundle")
            bundle = value;
        else if (option == "--output-dir")
            output_dir = value;
        else if (option == "--runtime-root")
            runtime_root = value;
        else if (option == "--benchmark")
            benchmark_runs = std::stoi(value);
        else if (option == "--warmup")
            warmup_runs = std::stoi(value);
        else if (option == "--num-hypotheses")
            num_hypotheses = std::stoi(value);
        else if (option == "--refinement-iterations")
            refinement_iterations = std::stoi(value);
        else if (option == "--mesh-diameter")
            mesh_diameter = std::stof(value);
        else
            throw std::invalid_argument("unknown qualification option: " + option);
    }
    if (bundle.empty() || output_dir.empty() || runtime_root.empty() || benchmark_runs < 1 ||
        warmup_runs < 0 || num_hypotheses < 1 || num_hypotheses > 252 ||
        refinement_iterations < 1 || refinement_iterations > 10 || !std::isfinite(mesh_diameter) ||
        mesh_diameter <= 0.0F)
        throw std::invalid_argument(
            "qualification requires bundle, output-dir, runtime-root, and valid parameters");
    std::filesystem::create_directories(output_dir);

    std::size_t free_before = 0;
    std::size_t total_memory = 0;
    if (cudaMemGetInfo(&free_before, &total_memory) != cudaSuccess)
        throw std::runtime_error("cudaMemGetInfo failed before pipeline load");
    const auto load_begin = std::chrono::steady_clock::now();
    auto task = trtmc::load_task(bundle, runtime_root);
    auto* pipeline = dynamic_cast<trtmc::IPoseHypothesisRefinement*>(task.get());
    if (pipeline == nullptr)
        throw std::runtime_error("bundle does not implement pose_hypothesis_refinement");
    const double startup_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_begin)
            .count();
    std::size_t free_after = 0;
    if (cudaMemGetInfo(&free_after, &total_memory) != cudaSuccess)
        throw std::runtime_error("cudaMemGetInfo failed after pipeline load");

    trtmc::PoseEstimationRequest registration;
    registration.candidate_poses = identity_poses(num_hypotheses);
    for (int32_t hypothesis = 0; hypothesis < num_hypotheses; ++hypothesis) {
        registration.candidate_poses[static_cast<std::size_t>(hypothesis) * 16 + 3] =
            (static_cast<float>(hypothesis) - 0.5F * (num_hypotheses - 1)) * 0.02F;
    }
    registration.num_hypotheses = num_hypotheses;
    registration.mesh_diameter = mesh_diameter;
    registration.refinement_iterations = refinement_iterations;
    registration.crop_provider = [&](const std::vector<float>& poses, trtmc::PoseCropStage stage,
                                     int32_t iteration) {
        auto batch = synthetic_crops(num_hypotheses, &poses);
        const std::string suffix =
            (stage == trtmc::PoseCropStage::kRefinement ? "refinement-" : "scoring-") +
            std::to_string(iteration) + ".f32";
        write_floats(std::filesystem::path(output_dir) / ("rendered_features." + suffix),
                     batch.rendered_features);
        write_floats(std::filesystem::path(output_dir) / ("observed_features." + suffix),
                     batch.observed_features);
        return batch;
    };
    const auto result = pipeline->estimate_pose_hypotheses(registration);
    if (!result.all_poses_rigid || result.best_index < 0)
        throw std::runtime_error("FoundationPose qualification produced invalid poses");

    write_floats(std::filesystem::path(output_dir) / "candidate_poses.f32",
                 registration.candidate_poses);
    write_floats(std::filesystem::path(output_dir) / "trt_refined_poses.f32", result.refined_poses);
    write_floats(std::filesystem::path(output_dir) / "trt_scores.f32", result.scores);

    trtmc::PoseCropBatch tracking_crops = synthetic_crops(1);
    trtmc::PoseEstimationRequest tracking;
    tracking.candidate_poses = identity_poses(1);
    tracking.num_hypotheses = 1;
    tracking.mesh_diameter = registration.mesh_diameter;
    tracking.refinement_iterations = 1;
    tracking.score_hypotheses = false;
    tracking.update_tracking_state = false;
    tracking.crop_provider = [&](const std::vector<float>&, trtmc::PoseCropStage, int32_t) {
        return tracking_crops;
    };
    for (int index = 0; index < warmup_runs; ++index)
        (void)pipeline->estimate_pose_hypotheses(tracking);
    std::vector<double> timings;
    for (int index = 0; index < benchmark_runs; ++index) {
        const auto begin = std::chrono::steady_clock::now();
        (void)pipeline->estimate_pose_hypotheses(tracking);
        timings.push_back(
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin)
                .count());
    }
    const double p50 = percentile(timings, 0.50);
    const double p95 = percentile(timings, 0.95);
    const double minimum = *std::min_element(timings.begin(), timings.end());
    const double maximum = *std::max_element(timings.begin(), timings.end());
    const double jitter = maximum - minimum;
    const double gpu_delta_mib =
        free_before > free_after ? static_cast<double>(free_before - free_after) / (1 << 20) : 0.0;
    std::cout << std::setprecision(9) << "{\"num_hypotheses\":" << result.num_hypotheses
              << ",\"best_index\":" << result.best_index << ",\"all_poses_rigid\":true"
              << ",\"tracking_latency_p50_ms\":" << p50 << ",\"tracking_latency_p95_ms\":" << p95
              << ",\"tracking_jitter_ms\":" << jitter
              << ",\"tracking_throughput_hz\":" << (p50 > 0.0 ? 1000.0 / p50 : 0.0)
              << ",\"startup_ms\":" << startup_ms << ",\"gpu_memory_delta_mib\":" << gpu_delta_mib
              << ",\"gpu_memory_total_mib\":" << static_cast<double>(total_memory) / (1 << 20)
              << ",\"refinement_ms\":" << result.refinement_ms
              << ",\"scoring_ms\":" << result.scoring_ms << "}\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) != "--qualify") {
        std::cerr << "FoundationPose qualification requires --qualify and its options\n";
        return 2;
    }
    try {
        return run_qualification(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "FoundationPose qualification failed: " << error.what() << '\n';
        return 2;
    }
}
