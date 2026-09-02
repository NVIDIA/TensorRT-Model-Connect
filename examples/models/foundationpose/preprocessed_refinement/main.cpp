/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/pipeline.h"

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

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4) {
            std::cerr << "Usage: " << argv[0] << " MODEL.bundle INPUT_DIR OUTPUT_POSES.f32\n";
            return 2;
        }
        constexpr int32_t hypotheses = 3;
        constexpr std::size_t crop_values = hypotheses * 160U * 160U * 6U;
        const std::filesystem::path input_dir = argv[2];
        auto poses = read_floats(input_dir / "candidate_poses.f32", hypotheses * 16U);
        auto rendered = read_floats(input_dir / "rendered_features.f32", crop_values);
        auto observed = read_floats(input_dir / "observed_features.f32", crop_values);

        auto pipeline = trtmc::load(argv[1]);
        trtmc::PoseEstimationRequest request;
        request.candidate_poses = poses;
        request.num_hypotheses = hypotheses;
        request.mesh_diameter = 0.18F;
        request.refinement_iterations = 2;
        request.crop_provider = [&](const std::vector<float>&, trtmc::PoseCropStage, int32_t) {
            return trtmc::PoseCropBatch{rendered, observed, hypotheses, 160, 160, 6};
        };
        const auto result = pipeline->estimate_pose_hypotheses(request);
        std::ofstream output(argv[3], std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(result.refined_poses.data()),
                     static_cast<std::streamsize>(result.refined_poses.size() * sizeof(float)));
        if (!output)
            throw std::runtime_error("failed to write refined poses");
        std::cout << "best_index=" << result.best_index
                  << " score=" << result.scores.at(result.best_index) << " rigid=" << std::boolalpha
                  << result.all_poses_rigid << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
