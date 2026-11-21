/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/diffusion_gemma/runtime/pipeline.h"
#include "trtmc/runtime/family_loader.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace {

using nlohmann::json;
using trtmc::diffusion_gemma::OwnedTensor;

void save_tensor(const std::filesystem::path& directory, json& tensors, const std::string& name,
                 const trtmc::Tensor& tensor) {
    const auto file = name + ".bin";
    std::ofstream output(directory / file, std::ios::binary);
    if (!output)
        throw std::runtime_error("cannot open diagnostic tensor " + file);
    if (tensor.dtype == trtmc::DType::kFloat32) {
        output.write(static_cast<const char*>(tensor.data),
                     static_cast<std::streamsize>(tensor.nbytes()));
    } else if (tensor.dtype == trtmc::DType::kBFloat16) {
        const auto* source = static_cast<const unsigned char*>(tensor.data);
        for (std::size_t index = 0; index < tensor.numel(); ++index) {
            std::uint16_t bits;
            std::memcpy(&bits, source + index * 2, sizeof(bits));
            const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
            float value;
            std::memcpy(&value, &wide, sizeof(value));
            output.write(reinterpret_cast<const char*>(&value), sizeof(value));
        }
    } else {
        throw std::runtime_error("unsupported diagnostic tensor storage dtype");
    }
    if (!output)
        throw std::runtime_error("failed to write diagnostic tensor " + file);
    tensors[name] = {{"file", file}, {"shape", tensor.shape}, {"dtype", "float32"}};
}

bool caches_equal(const std::vector<OwnedTensor>& first, const std::vector<OwnedTensor>& second) {
    if (first.size() != second.size())
        return false;
    for (std::size_t index = 0; index < first.size(); ++index)
        if (first[index].shape != second[index].shape ||
            first[index].dtype != second[index].dtype || first[index].bytes != second[index].bytes)
            return false;
    return true;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4 && argc != 5)
            throw std::invalid_argument(
                "expected BUNDLE REQUEST.json OUTPUT_DIRECTORY [RUNTIME_ROOT]");
        std::ifstream request_stream(argv[2]);
        if (!request_stream)
            throw std::runtime_error("cannot open diagnostic request");
        json request;
        request_stream >> request;
        const std::filesystem::path directory(argv[3]);
        std::filesystem::create_directories(directory);
        const char* root_environment = std::getenv("TRTMC_RUNTIME_ROOT");
        const std::string root = argc == 5 ? argv[4] : (root_environment ? root_environment : "");
        auto task = trtmc::load_task(argv[1], root);
        auto* pipeline = dynamic_cast<trtmc::diffusion_gemma::Pipeline*>(task.get());
        if (pipeline == nullptr)
            throw std::runtime_error("diagnostic requires a diffusion_gemma pipeline");
        const auto& config = pipeline->config();
        json metadata{{"tensors", json::object()},
                      {"cache_readonly", true},
                      {"precision", config.dtype == trtmc::DType::kFloat32 ? "fp32" : "bf16"},
                      {"max_sequence_length", config.max_sequence_length},
                      {"canvas_length", config.canvas_length},
                      {"format_version", 2},
                      {"engine_layout", "qkv_layer"}};
        const bool trace_layers = request.value("trace_layers", false);
        auto trace = [&](const std::string& prefix) -> trtmc::diffusion_gemma::TraceCallback {
            if (!trace_layers)
                return {};
            return [&, prefix](const std::string& name, const trtmc::Tensor& tensor) {
                save_tensor(directory, metadata["tensors"], prefix + name, tensor);
            };
        };
        pipeline->encode(request.at("input_ids").get<std::vector<std::int32_t>>(),
                         trace("encoder_"));
        for (std::size_t index = 0; index < pipeline->keys().size(); ++index) {
            save_tensor(directory, metadata["tensors"], "encoder_key_" + std::to_string(index),
                        pipeline->keys()[index].view());
            save_tensor(directory, metadata["tensors"], "encoder_value_" + std::to_string(index),
                        pipeline->values()[index].view());
        }
        const auto original_keys = pipeline->keys();
        const auto original_values = pipeline->values();
        auto canvas_document = request.at("decoder_input_ids");
        if (canvas_document.empty())
            throw std::invalid_argument("diagnostic needs at least one complete canvas");
        if (canvas_document.front().is_number_integer())
            canvas_document = json::array({canvas_document});
        const auto canvases = canvas_document.get<std::vector<std::vector<std::int32_t>>>();
        const auto temperatures =
            request.value("temperatures", std::vector<float>(canvases.size(), 0.8F));
        if (temperatures.size() != canvases.size())
            throw std::invalid_argument("one temperature is required for every diagnostic canvas");
        auto conditioning = pipeline->zero_conditioning(canvases.front().size());
        for (std::size_t step = 0; step < canvases.size(); ++step) {
            const auto prefix = "step_" + std::to_string(step) + "_";
            auto observation =
                pipeline->denoise(canvases[step], conditioning, temperatures[step], trace(prefix));
            save_tensor(directory, metadata["tensors"], prefix + "raw_logits",
                        observation.logits.view());
            save_tensor(directory, metadata["tensors"], prefix + "self_conditioning",
                        observation.self_conditioning.view());
            metadata["tensors"]["raw_logits"] = metadata["tensors"][prefix + "raw_logits"];
            conditioning = std::move(observation.self_conditioning);
            if (!caches_equal(original_keys, pipeline->keys()) ||
                !caches_equal(original_values, pipeline->values()))
                throw std::runtime_error("denoising mutated the encoder prefix KV cache");
        }
        metadata["decoder_steps"] = canvases.size();
        std::ofstream output(directory / "metadata.json");
        output << metadata.dump(2) << '\n';
        output.close();
        if (!output)
            throw std::runtime_error("cannot write diffusion_gemma diagnostic metadata");
        std::cout << (directory / "metadata.json").string() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
