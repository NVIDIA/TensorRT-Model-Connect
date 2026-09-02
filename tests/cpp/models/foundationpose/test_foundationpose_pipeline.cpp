/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/foundationpose/pipeline.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

class FakeModule final : public trtmc::ITrtModule {
  public:
    explicit FakeModule(bool scorer) : scorer_(scorer) {}

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++forward_calls;
        const auto batch = static_cast<int32_t>(inputs.at("input1").shape.at(0));
        batches.push_back(batch);
        if (scorer_) {
            values_.resize(batch);
            for (int32_t index = 0; index < batch; ++index)
                values_[index] = static_cast<float>(index) - 1.0F;
            return {{"output1", {values_.data(), {1, batch}, trtmc::DType::kFloat32}}};
        }
        translation_.assign(static_cast<std::size_t>(batch) * 3, 0.0F);
        rotation_.assign(static_cast<std::size_t>(batch) * 3, 0.0F);
        for (int32_t index = 0; index < batch; ++index) {
            translation_[static_cast<std::size_t>(index) * 3] = 0.1F;
            rotation_[static_cast<std::size_t>(index) * 3 + 2] = 0.2F;
        }
        return {
            {"output1", {translation_.data(), {batch, 3}, trtmc::DType::kFloat32}},
            {"output2", {rotation_.data(), {batch, 3}, trtmc::DType::kFloat32}},
        };
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override {
        return name == "input1" || name == "input2";
    }
    bool has_output(const std::string& name) const override {
        return name == "output1" || (!scorer_ && name == "output2");
    }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void reset_execution_context() override { ++reset_calls; }
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

    bool scorer_{false};
    int forward_calls{0};
    int reset_calls{0};
    std::vector<int32_t> batches;
    std::vector<float> translation_;
    std::vector<float> rotation_;
    std::vector<float> values_;
};

std::vector<float> identity_poses(int32_t count) {
    std::vector<float> poses(static_cast<std::size_t>(count) * 16, 0.0F);
    for (int32_t index = 0; index < count; ++index) {
        auto* pose = poses.data() + static_cast<std::size_t>(index) * 16;
        pose[0] = pose[5] = pose[10] = pose[15] = 1.0F;
    }
    return poses;
}

trtmc::PoseCropProvider crops(int32_t expected_count, int* calls = nullptr) {
    return [=](const std::vector<float>& poses, trtmc::PoseCropStage, int32_t) mutable {
        if (calls != nullptr)
            ++*calls;
        const auto values = static_cast<std::size_t>(expected_count) * 2 * 2 * 6;
        trtmc::PoseCropBatch result;
        result.rendered_features.assign(values, poses.at(3));
        result.observed_features.assign(values, 0.25F);
        result.num_hypotheses = expected_count;
        result.height = 2;
        result.width = 2;
        result.channels = 6;
        return result;
    };
}

void test_refinement_scoring_and_chunking() {
    auto refiner = std::make_unique<FakeModule>(false);
    auto scorer = std::make_unique<FakeModule>(true);
    auto* refiner_ptr = refiner.get();
    auto* scorer_ptr = scorer.get();
    trtmc::FoundationPosePipeline pipeline(std::move(refiner), std::move(scorer), 2, 2, 6, 2, 5,
                                           "foundationpose-test");
    int crop_calls = 0;
    trtmc::PoseEstimationRequest request;
    request.candidate_poses = identity_poses(5);
    request.num_hypotheses = 5;
    request.mesh_diameter = 2.0F;
    request.refinement_iterations = 2;
    request.crop_provider = crops(5, &crop_calls);
    const auto result = pipeline.estimate_pose_hypotheses(request);

    check(result.num_hypotheses == 5 && result.refined_poses.size() == 80,
          "FoundationPose result shape");
    check(result.scores == std::vector<float>({-1.0F, 0.0F, 1.0F, 2.0F, 3.0F}),
          "FoundationPose score logits");
    check(result.best_index == 4, "FoundationPose best hypothesis selection");
    check(result.all_poses_rigid, "FoundationPose decoded transforms remain rigid");
    check(std::abs(result.refined_poses[3] - 0.2F) < 1.0e-6F,
          "FoundationPose iterative metric translation decoding");
    check(refiner_ptr->batches == std::vector<int32_t>({2, 2, 1, 2, 2, 1}),
          "FoundationPose chunks refiner batches at the profile limit");
    check(scorer_ptr->batches == std::vector<int32_t>({5}),
          "FoundationPose scores all hypotheses jointly");
    check(crop_calls == 3, "FoundationPose asks for rerendered crops each iteration and scoring");
}

void test_tracking_and_reset() {
    auto refiner = std::make_unique<FakeModule>(false);
    auto scorer = std::make_unique<FakeModule>(true);
    auto* refiner_ptr = refiner.get();
    auto* scorer_ptr = scorer.get();
    trtmc::FoundationPosePipeline pipeline(std::move(refiner), std::move(scorer), 2, 2, 6, 2, 5);
    trtmc::PoseEstimationRequest request;
    request.candidate_poses = identity_poses(1);
    request.num_hypotheses = 1;
    request.mesh_diameter = 1.0F;
    request.score_hypotheses = false;
    request.crop_provider = crops(1);
    const auto first = pipeline.estimate_pose_hypotheses(request);
    request.candidate_poses.clear();
    request.use_tracked_pose = true;
    const auto tracked = pipeline.estimate_pose_hypotheses(request);
    check(tracked.refined_poses[3] > first.refined_poses[3],
          "FoundationPose tracking refines the selected prior pose");
    pipeline.reset();
    bool rejected = false;
    try {
        (void)pipeline.estimate_pose_hypotheses(request);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "FoundationPose reset clears tracking state");
    check(refiner_ptr->reset_calls == 1 && scorer_ptr->reset_calls == 1,
          "FoundationPose reset reuses both engine contexts");
}

void test_invalid_contracts() {
    trtmc::FoundationPosePipeline pipeline(std::make_unique<FakeModule>(false),
                                           std::make_unique<FakeModule>(true), 2, 2, 6, 2, 5);
    auto valid = trtmc::PoseEstimationRequest{};
    valid.candidate_poses = identity_poses(1);
    valid.num_hypotheses = 1;
    valid.mesh_diameter = 1.0F;
    valid.score_hypotheses = false;
    valid.crop_provider = crops(1);
    auto rejects = [&](const trtmc::PoseEstimationRequest& request) {
        try {
            (void)pipeline.estimate_pose_hypotheses(request);
            return false;
        } catch (const std::invalid_argument&) {
            return true;
        }
    };
    auto value = valid;
    value.candidate_poses[15] = 0.0F;
    check(rejects(value), "FoundationPose rejects non-rigid candidates");
    value = valid;
    value.mesh_diameter = std::numeric_limits<float>::quiet_NaN();
    check(rejects(value), "FoundationPose rejects invalid mesh scale");
    value = valid;
    value.num_hypotheses = 6;
    value.candidate_poses = identity_poses(6);
    value.crop_provider = crops(6);
    check(rejects(value), "FoundationPose rejects batches above the bundle limit");
    value = valid;
    value.crop_provider = [](const std::vector<float>&, trtmc::PoseCropStage, int32_t) {
        return trtmc::PoseCropBatch{{0.0F}, {0.0F}, 1, 2, 2, 6};
    };
    check(rejects(value), "FoundationPose rejects malformed crop tensors");
    value = valid;
    value.crop_provider = [](const std::vector<float>&, trtmc::PoseCropStage, int32_t) {
        trtmc::PoseCropBatch result;
        result.rendered_features.assign(24, 0.0F);
        result.observed_features.assign(24, 0.0F);
        result.observed_features[3] = std::numeric_limits<float>::infinity();
        result.num_hypotheses = 1;
        result.height = result.width = 2;
        result.channels = 6;
        return result;
    };
    check(rejects(value), "FoundationPose rejects non-finite crop features");
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
    std::string backend_dir;
    std::string plugin_dir;
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
        else if (option == "--backend-dir")
            backend_dir = value;
        else if (option == "--model-plugin-dir")
            plugin_dir = value;
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
    if (bundle.empty() || output_dir.empty() || benchmark_runs < 1 || warmup_runs < 0 ||
        num_hypotheses < 1 || num_hypotheses > 252 || refinement_iterations < 1 ||
        refinement_iterations > 10 || !std::isfinite(mesh_diameter) || mesh_diameter <= 0.0F)
        throw std::invalid_argument(
            "qualification requires bundle, output-dir, and valid inference parameters");
    std::filesystem::create_directories(output_dir);

    std::size_t free_before = 0;
    std::size_t total_memory = 0;
    if (cudaMemGetInfo(&free_before, &total_memory) != cudaSuccess)
        throw std::runtime_error("cudaMemGetInfo failed before pipeline load");
    trtmc::LoadOptions options;
    if (!backend_dir.empty())
        options.backend_search_paths.push_back(backend_dir);
    if (!plugin_dir.empty())
        options.model_plugin_search_paths.push_back(plugin_dir);
    const auto load_begin = std::chrono::steady_clock::now();
    auto pipeline = trtmc::load(bundle, options);
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
    if (argc > 1 && std::string(argv[1]) == "--qualify") {
        try {
            return run_qualification(argc, argv);
        } catch (const std::exception& error) {
            std::cerr << "FoundationPose qualification failed: " << error.what() << '\n';
            return 2;
        }
    }
    test_refinement_scoring_and_chunking();
    test_tracking_and_reset();
    test_invalid_contracts();
    if (failures != 0) {
        std::cerr << failures << " FoundationPose pipeline test(s) failed\n";
        return 1;
    }
    return 0;
}
