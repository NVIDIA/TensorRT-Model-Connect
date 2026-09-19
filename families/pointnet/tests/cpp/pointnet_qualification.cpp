/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <trtmc/point_cloud.h>
#include <trtmc/trtmc.h>
#include <vector>

namespace {

trtmc_string_view str(const char* value) {
    return {value, value ? std::strlen(value) : 0};
}

void write_floats(const std::filesystem::path& path, const std::vector<float>& values) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!output)
        throw std::runtime_error("failed to write " + path.string());
}

void write_ints(const std::filesystem::path& path, const std::int32_t* values,
                std::uint64_t count) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(values),
                 static_cast<std::streamsize>(count * sizeof(std::int32_t)));
    if (!output)
        throw std::runtime_error("failed to write " + path.string());
}

int run_qualification(int argc, char** argv) {
    std::string bundle;
    std::string output_dir;
    std::string runtime_root;
    std::string points_path;
    std::uint32_t input_dim = 9;
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
        else if (option == "--points")
            points_path = value;
        else if (option == "--input-dim")
            input_dim = static_cast<std::uint32_t>(std::stoul(value));
        else
            throw std::invalid_argument("unknown qualification option: " + option);
    }
    if (bundle.empty() || output_dir.empty() || runtime_root.empty() || points_path.empty() ||
        input_dim == 0)
        throw std::invalid_argument(
            "qualification requires bundle, output-dir, runtime-root, points, input-dim");
    std::filesystem::create_directories(output_dir);

    const std::uintmax_t file_size = std::filesystem::file_size(points_path);
    if (file_size % (input_dim * sizeof(float)) != 0)
        throw std::runtime_error("point fixture size is not a multiple of the feature size");
    const std::uint64_t num_points = file_size / (input_dim * sizeof(float));
    std::vector<float> points(num_points * input_dim);
    {
        std::ifstream input(points_path, std::ios::binary);
        input.read(reinterpret_cast<char*>(points.data()),
                   static_cast<std::streamsize>(points.size() * sizeof(float)));
        if (!input)
            throw std::runtime_error("failed to read point fixture");
    }

    const trtmc_core_api_v1* core = nullptr;
    if (trtmc_get_api(1, 0, &core) != TRTMC_OK || core == nullptr)
        throw std::runtime_error("trtmc_get_api failed");

    trtmc_load_options_v1 options{};
    options.struct_size = sizeof(trtmc_load_options_v1);
    options.runtime_root = str(runtime_root.c_str());
    trtmc_model* model = nullptr;
    trtmc_error* error = nullptr;
    if (core->model_load(str(bundle.c_str()), &options, &model, &error) != TRTMC_OK)
        throw std::runtime_error("model_load failed: " +
                                 std::string(error && core->error_message(error).data
                                                 ? core->error_message(error).data
                                                 : "unknown"));

    const trtmc_api_header* header = nullptr;
    if (core->model_get_task_api(model, str(TRTMC_TASK_POINTS_TO_SEMANTIC_SEGMENTATION), 1, 0,
                                 &header, &error) != TRTMC_OK)
        throw std::runtime_error("model_get_task_api failed");
    const auto* api = reinterpret_cast<const trtmc_points_to_semantic_segmentation_api_v1*>(header);

    trtmc_points_to_semantic_segmentation_request_v1 request{points.data(), num_points, input_dim};
    trtmc_result* result = nullptr;
    if (api->run(model, &request, nullptr, &result, &error) != TRTMC_OK)
        throw std::runtime_error("pointnet run failed");

    trtmc_points_to_semantic_segmentation_view_v1 view{};
    if (api->result_view(result, &view, &error) != TRTMC_OK)
        throw std::runtime_error("pointnet result_view failed");

    write_ints(std::filesystem::path(output_dir) / "trt_labels.i32", view.labels, view.num_points);
    std::vector<float> logits(view.class_scores, view.class_scores + view.class_score_count);
    write_floats(std::filesystem::path(output_dir) / "trt_logits.f32", logits);

    std::cout << "{\"num_points\":" << view.num_points << ",\"num_classes\":" << view.num_classes
              << ",\"score_kind\":" << view.score_kind << "}\n";
    core->result_release(result);
    core->model_release(model);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) != "--qualify") {
        std::cerr << "PointNet qualification requires --qualify and its options\n";
        return 2;
    }
    try {
        return run_qualification(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "PointNet qualification failed: " << error.what() << '\n';
        return 2;
    }
}
