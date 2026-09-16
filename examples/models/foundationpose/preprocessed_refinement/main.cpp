/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "task_runtime.h"
#include "trtmc/control.hpp"
#include "trtmc/perception.hpp"
#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<float> read_floats(const std::filesystem::path& path, std::size_t expected) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() != static_cast<std::streamoff>(expected * sizeof(float)))
        throw std::runtime_error(path.string() + " has an unexpected size");
    std::vector<float> values(expected);
    input.seekg(0);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(expected * sizeof(float)));
    if (!input)
        throw std::runtime_error("failed to read " + path.string());
    return values;
}

void write_result(const std::string& path, trtmc::Span<const float> poses,
                  trtmc::Span<const float> scores, int64_t best_index, bool rigid) {
    if (poses.size() != 48 || scores.size() != 3 || best_index < 0 || best_index >= 3)
        throw std::runtime_error("three refined poses and scored hypotheses are required");
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(poses.data()),
                 static_cast<std::streamsize>(poses.size() * sizeof(float)));
    output.flush();
    if (!output)
        throw std::runtime_error("failed to write refined poses");
    std::cout << "best_index=" << best_index << " score=" << scores[static_cast<size_t>(best_index)]
              << " rigid=" << std::boolalpha << rigid << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4 && argc != 5) {
            std::cerr << "Usage: " << argv[0]
                      << " MODEL.bundle [RUNTIME_ROOT] INPUT_DIR OUTPUT_POSES.f32\n";
            return 2;
        }
        constexpr int32_t hypotheses = 3;
        constexpr std::size_t crop_values = hypotheses * 160U * 160U * 6U;
        const std::string runtime_root = argc == 5 ? argv[2] : "";
        const std::filesystem::path input_dir = argv[argc - 2];
        auto poses = read_floats(input_dir / "candidate_poses.f32", hypotheses * 16U);
        auto rendered = read_floats(input_dir / "rendered_features.f32", crop_values);
        auto observed = read_floats(input_dir / "observed_features.f32", crop_values);

        const auto primary = trtmc::Bundle::open(argv[1]).info().task;
        if (trtmc::app::uses_existing_task_runtime(primary)) {
            if (runtime_root.empty())
                throw std::invalid_argument("runtime root is required for an existing bundle mode");
            auto task = trtmc::load_task(argv[1], runtime_root);
            auto* refinement = dynamic_cast<trtmc::IPoseHypothesisRefinement*>(task.get());
            if (refinement == nullptr)
                throw std::runtime_error("bundle does not implement pose_hypothesis_refinement");
            trtmc::PoseEstimationRequest request;
            request.candidate_poses = poses;
            request.num_hypotheses = hypotheses;
            request.mesh_diameter = 0.18F;
            request.refinement_iterations = 2;
            request.crop_provider = [&](const std::vector<float>&, trtmc::PoseCropStage, int32_t) {
                return trtmc::PoseCropBatch{rendered, observed, hypotheses, 160, 160, 6};
            };
            const auto result = refinement->estimate_pose_hypotheses(request);
            if (result.num_hypotheses != hypotheses)
                throw std::runtime_error("refiner changed the hypothesis count");
            write_result(argv[argc - 1], {result.refined_poses.data(), result.refined_poses.size()},
                         {result.scores.data(), result.scores.size()}, result.best_index,
                         result.all_poses_rigid);
        } else {
            trtmc::LoadOptions load;
            load.runtime_root = runtime_root;
            auto model = trtmc::Model::load(argv[1], load);
            const trtmc::PoseHypothesesCropsToRefinedPosesRequest request{
                {{poses.data(), poses.size()}, hypotheses},
                0.18F,
                [&](const trtmc::PoseCropQuery&) {
                    return trtmc::PoseCrops{rendered, observed, hypotheses, 160, 160, 6};
                }};
            const auto result = model.task<trtmc::PoseHypothesesCropsToRefinedPoses>().run(
                request, {{"refinement_iterations", int64_t{2}}, {"score_hypotheses", true}});
            const auto view = result.view();
            if (view.refined_poses.count != hypotheses)
                throw std::runtime_error("refiner changed the hypothesis count");
            write_result(
                argv[argc - 1],
                {view.refined_poses.values, static_cast<size_t>(view.refined_poses.value_count)},
                {view.scores, static_cast<size_t>(view.score_count)}, view.best_index,
                view.all_poses_rigid != 0);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
