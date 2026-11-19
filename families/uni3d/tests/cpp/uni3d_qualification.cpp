/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Runs one point cloud through the public points_to_embedding C table and
// writes the embedding, so the E2E can compare it with the PyTorch reference.

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <trtmc/features.h>
#include <trtmc/trtmc.h>
#include <vector>

namespace {

constexpr std::uint32_t kInputDim = 6;

trtmc_string_view str(const char* value) {
    return {value, value ? std::strlen(value) : 0};
}

std::string text(trtmc_string_view value) {
    return value.data ? std::string(value.data, value.size) : std::string();
}

std::string error_text(const trtmc_core_api_v1* core, trtmc_error* error) {
    return error ? text(core->error_message(error)) : std::string("unknown");
}

std::vector<float> read_points(const std::string& path) {
    const std::uintmax_t size = std::filesystem::file_size(path);
    if (size == 0 || size % (kInputDim * sizeof(float)) != 0)
        throw std::runtime_error("point fixture is not a [N, 6] float32 file");
    std::vector<float> points(size / sizeof(float));
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(points.data()), static_cast<std::streamsize>(size));
    if (!input)
        throw std::runtime_error("failed to read point fixture");
    return points;
}

int run_qualification(int argc, char** argv) {
    std::string bundle;
    std::string output_dir;
    std::string runtime_root;
    std::string points_path;
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
        else
            throw std::invalid_argument("unknown qualification option: " + option);
    }
    if (bundle.empty() || output_dir.empty() || runtime_root.empty() || points_path.empty())
        throw std::invalid_argument(
            "qualification requires bundle, output-dir, runtime-root, points");
    std::filesystem::create_directories(output_dir);
    const std::vector<float> points = read_points(points_path);

    const trtmc_core_api_v1* core = nullptr;
    if (trtmc_get_api(1, 0, &core) != TRTMC_OK || core == nullptr)
        throw std::runtime_error("trtmc_get_api failed");
    trtmc_load_options_v1 options{};
    options.struct_size = sizeof(trtmc_load_options_v1);
    options.runtime_root = str(runtime_root.c_str());
    trtmc_model* model = nullptr;
    trtmc_error* error = nullptr;
    if (core->model_load(str(bundle.c_str()), &options, &model, &error) != TRTMC_OK)
        throw std::runtime_error("model_load failed: " + error_text(core, error));

    const trtmc_api_header* header = nullptr;
    if (core->model_get_task_api(model, str(TRTMC_TASK_POINTS_TO_EMBEDDING), 1, 0, &header,
                                 &error) != TRTMC_OK)
        throw std::runtime_error("model_get_task_api failed: " + error_text(core, error));
    const auto* api = reinterpret_cast<const trtmc_points_to_embedding_api_v1*>(header);

    trtmc_points_to_embedding_request_v1 request{points.data(), points.size() / kInputDim,
                                                 kInputDim};
    trtmc_result* result = nullptr;
    if (api->run(model, &request, nullptr, &result, &error) != TRTMC_OK)
        throw std::runtime_error("uni3d run failed: " + error_text(core, error));
    trtmc_semantic_embedding_view_v1 view{};
    if (api->result_view(result, &view, &error) != TRTMC_OK)
        throw std::runtime_error("uni3d result_view failed: " + error_text(core, error));

    std::ofstream output(std::filesystem::path(output_dir) / "trt_embedding.f32",
                         std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(view.values),
                 static_cast<std::streamsize>(view.count * sizeof(float)));
    if (!output)
        throw std::runtime_error("failed to write trt_embedding.f32");
    std::cout << "{\"count\":" << view.count << ",\"embedding_space\":\""
              << text(view.embedding_space) << "\",\"pooling\":\"" << text(view.pooling)
              << "\",\"normalization\":\"" << text(view.normalization) << "\"}\n";
    core->result_release(result);
    core->model_release(model);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) != "--qualify") {
        std::cerr << "Uni3D qualification requires --qualify and its options\n";
        return 2;
    }
    try {
        return run_qualification(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "Uni3D qualification failed: " << error.what() << '\n';
        return 2;
    }
}
