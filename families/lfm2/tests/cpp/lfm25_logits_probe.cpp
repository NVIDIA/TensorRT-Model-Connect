/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Reference validation only: replay token IDs through the production native
// decoder and state, exporting its logits without introducing another sampler.
#include "families/lfm2/runtime/hybrid_state.h"
#include "families/lfm2/runtime/plugin_helpers.h"

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
            throw std::invalid_argument("usage: lfm25_logits_probe bundle");
        trtmc::BundleReader bundle(argv[1]);
        const auto cfg =
            nlohmann::json::parse(trtmc::lfm2::require_text_section(bundle, "runtime.json"));
        std::unique_ptr<trtmc::IBackend, decltype(&trtmc_destroy_backend)> backend(
            trtmc_create_backend(), trtmc_destroy_backend);
        auto module =
            trtmc::lfm2::load_module(*backend, trtmc::lfm2::require_section(bundle, "engine.plan"));
        const auto stream = module->stream();
        const auto dtype = trtmc::lfm2::state_dtype(cfg.at("precision"));
        auto kv = std::make_unique<trtmc::Lfm2KvCache>(
            cfg.at("num_attention_layers"), cfg.at("max_cache_length"),
            cfg.at("num_key_value_heads"), cfg.at("head_dim"), stream, dtype,
            trtmc::lfm2::kv_names(cfg.at("num_attention_layers")));
        auto conv =
            std::make_unique<trtmc::Lfm2ConvState>(cfg.at("num_conv_layers"), cfg.at("hidden_size"),
                                                   cfg.at("conv_L_cache"), stream, dtype);
        trtmc::Lfm2HybridState state(std::move(kv), std::move(conv));
        if (!state.ok())
            throw std::runtime_error("state allocation failed");
        state.bind_to(*module);
        const int vocab = cfg.at("vocab_size");
        std::vector<float> logits(vocab);
        std::cout << "{\"ready\":true}" << std::endl;
        for (std::string line; std::getline(std::cin, line);) {
            const auto request = nlohmann::json::parse(line);
            if (request.value("reset", false)) {
                state.reset();
                module->reset_execution_context();
                state.bind_to(*module);
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
                int32_t phase = request.value("decode_phase", false) ? 1 : 0;
                if (module->has_input("decode_phase"))
                    inputs["decode_phase"] = trtmc::Tensor{&phase, {1, 1}, trtmc::DType::kInt32};
                state.prepare_step(inputs);
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
