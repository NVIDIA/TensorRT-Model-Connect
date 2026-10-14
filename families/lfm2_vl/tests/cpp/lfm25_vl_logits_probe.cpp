/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Reference validation only: replay token IDs through the production native
// decoder and state, exporting its logits without introducing another sampler.
#include "families/lfm2_vl/runtime/hybrid_state.h"
#include "families/lfm2_vl/runtime/image_processor.h"
#include "families/lfm2_vl/runtime/plugin_helpers.h"

#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc != 2)
            throw std::invalid_argument("usage: lfm2_vl5_logits_probe bundle");
        trtmc::BundleReader bundle(argv[1]);
        const auto cfg =
            nlohmann::json::parse(trtmc::lfm2_vl::require_text_section(bundle, "runtime.json"));
        std::unique_ptr<trtmc::IBackend, decltype(&trtmc_destroy_backend)> backend(
            trtmc_create_backend(), trtmc_destroy_backend);
        auto module = trtmc::lfm2_vl::load_module(
            *backend, trtmc::lfm2_vl::require_section(bundle, "engine.plan"));
        const auto stream = module->stream();
        const auto dtype = trtmc::lfm2_vl::state_dtype(cfg.at("precision"));
        auto kv = std::make_unique<trtmc::Lfm2VlKvCache>(
            cfg.at("num_attention_layers"), cfg.at("max_cache_length"),
            cfg.at("num_key_value_heads"), cfg.at("head_dim"), stream, dtype,
            trtmc::lfm2_vl::kv_names(cfg.at("num_attention_layers")));
        auto conv = std::make_unique<trtmc::Lfm2VlConvState>(cfg.at("num_conv_layers"),
                                                             cfg.at("hidden_size"),
                                                             cfg.at("conv_L_cache"), stream, dtype);
        auto* kv_state = kv.get();
        auto* conv_state = conv.get();
        trtmc::Lfm2VlHybridState state(std::move(kv), std::move(conv));
        if (!state.ok())
            throw std::runtime_error("state allocation failed");
        state.bind_to(*module);
        auto vision = trtmc::lfm2_vl::load_module(
            *backend, trtmc::lfm2_vl::require_section(bundle, "vision.plan"));
        const auto bytes = trtmc::lfm2_vl::require_section(bundle, "vision_positions.bin");
        std::vector<float> table(bytes.size() / sizeof(float));
        std::memcpy(table.data(), bytes.data(), bytes.size());
        std::vector<float> image_embeddings, zero(2048, 0.0F);
        std::size_t image_offset = 0;
        const int vocab = cfg.at("vocab_size");
        std::vector<float> logits(vocab);
        std::cout << "{\"ready\":true}" << std::endl;
        for (std::string line; std::getline(std::cin, line);) {
            const auto request = nlohmann::json::parse(line);
            if (request.value("reset", false)) {
                image_embeddings.clear();
                image_offset = 0;
                if (request.contains("embedding_path")) {
                    std::ifstream input(request.at("embedding_path").get<std::string>(),
                                        std::ios::binary | std::ios::ate);
                    const auto size = input.tellg();
                    if (size < 0 || size % (2048 * sizeof(float)))
                        throw std::runtime_error("invalid embedding file");
                    image_embeddings.resize(static_cast<std::size_t>(size) / sizeof(float));
                    input.seekg(0);
                    input.read(reinterpret_cast<char*>(image_embeddings.data()), size);
                } else
                    for (const auto& media : request.value("images", nlohmann::json::array())) {
                        trtmc::ImageResult image;
                        image.height = media.at("height");
                        image.width = media.at("width");
                        image.pixels.resize(static_cast<std::size_t>(image.height) * image.width *
                                            3);
                        std::ifstream input(media.at("path").get<std::string>(), std::ios::binary);
                        input.read(reinterpret_cast<char*>(image.pixels.data()),
                                   image.pixels.size() * sizeof(float));
                        if (!input)
                            throw std::runtime_error("invalid image file");
                        const auto processed = trtmc::lfm2_vl::process_image(image, {});
                        for (const auto& tile : processed.tiles) {
                            auto prepared = trtmc::lfm2_vl::prepare_vision_inputs(
                                tile, {}, table, 1152, cfg.at("precision") == "bf16");
                            const auto count = prepared.valid_patches;
                            trtmc::TensorMap tensors;
                            tensors["pixel_values"] = {prepared.pixels.data(),
                                                       {prepared.patch_count, 768},
                                                       trtmc::DType::kFloat32};
                            tensors["position_embeddings"] = {prepared.positions.data(),
                                                              {prepared.patch_count, 1152},
                                                              trtmc::DType::kFloat32};
                            tensors["pixel_attention_mask"] = {
                                prepared.mask.data(), {prepared.patch_count}, trtmc::DType::kInt32};
                            tensors["unshuffle_indices"] = {
                                const_cast<std::int32_t*>(tile.unshuffle.data()),
                                {count / 4, 4},
                                trtmc::DType::kInt32};
                            const auto result = vision->forward(tensors);
                            const auto* data =
                                static_cast<const float*>(result.at("image_embeddings").data);
                            image_embeddings.insert(image_embeddings.end(), data,
                                                    data + count / 4 * 2048);
                        }
                    }
                if (request.contains("dump_embeddings")) {
                    std::ofstream output(request.at("dump_embeddings").get<std::string>(),
                                         std::ios::binary);
                    output.write(reinterpret_cast<const char*>(image_embeddings.data()),
                                 image_embeddings.size() * sizeof(float));
                }
                state.reset();
                module->reset_execution_context();
                state.bind_to(*module);
                if (request.contains("inject_state")) {
                    const auto& supplied = request.at("inject_state");
                    const auto position = supplied.at("position").get<int32_t>();
                    if (position <= 0 || position >= state.max_length())
                        throw std::invalid_argument("invalid injected state position");
                    state.advance(position);
                    auto inject = [&](trtmc::DeviceTensor& destination,
                                      const nlohmann::json& path) {
                        std::ifstream input(path.get<std::string>(),
                                            std::ios::binary | std::ios::ate);
                        if (input.tellg() != static_cast<std::streamoff>(destination.nbytes()))
                            throw std::invalid_argument("injected state byte size mismatch");
                        std::vector<std::uint8_t> bytes(destination.nbytes());
                        input.seekg(0);
                        input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
                        if (!input || !destination.copy_from_host(bytes.data()) ||
                            cudaStreamSynchronize(stream) != cudaSuccess)
                            throw std::runtime_error("injected state transfer failed");
                    };
                    for (int32_t i = 0; i < cfg.at("num_attention_layers").get<int32_t>(); ++i) {
                        inject(kv_state->cache_k(i), supplied.at("keys").at(i));
                        inject(kv_state->cache_v(i), supplied.at("values").at(i));
                    }
                    for (int32_t i = 0; i < cfg.at("num_conv_layers").get<int32_t>(); ++i)
                        inject(conv_state->state(i), supplied.at("convolution").at(i));
                }
            }
            const auto tokens = request.at("tokens").get<std::vector<int32_t>>();
            if (tokens.empty() ||
                tokens.size() > static_cast<size_t>(state.max_length() - state.position()))
                throw std::invalid_argument("empty token input or context capacity exceeded");
            for (int32_t token : tokens) {
                if (token < 0 || token >= vocab)
                    throw std::invalid_argument("token ID is outside vocabulary");
                trtmc::TensorMap inputs;
                inputs["token_id"] = trtmc::Tensor{&token, {1}, trtmc::DType::kInt32};
                state.prepare_step(inputs);
                std::int32_t phase = request.value("decode", !request.value("reset", false));
                if (module->has_input("decode_phase"))
                    inputs["decode_phase"] = {&phase, {1, 1}, trtmc::DType::kInt32};
                std::int32_t use_image = !phase && token == 396 && !image_embeddings.empty();
                float* embedding = zero.data();
                if (use_image) {
                    if (image_offset + 2048 > image_embeddings.size())
                        throw std::runtime_error("image token mismatch");
                    embedding = image_embeddings.data() + image_offset;
                    image_offset += 2048;
                }
                inputs["image_embedding"] = {embedding, {1, 2048}, trtmc::DType::kFloat32};
                inputs["use_image"] = {&use_image, {1, 1}, trtmc::DType::kInt32};
                module->forward_async(inputs);
                module->sync();
                state.advance();
            }
            if (cudaMemcpy(logits.data(), module->device_ptr("logits"),
                           logits.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
                throw std::runtime_error("logits transfer failed");
            std::ofstream output(request.at("output").get<std::string>(), std::ios::binary);
            output.write(reinterpret_cast<const char*>(logits.data()),
                         logits.size() * sizeof(float));
            output.close();
            if (!output)
                throw std::runtime_error("logits write failed");
            std::cout
                << nlohmann::json({{"position", state.position()}, {"vocab_size", vocab}}).dump()
                << std::endl;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
