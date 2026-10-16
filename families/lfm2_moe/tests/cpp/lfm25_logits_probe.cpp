/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Reference validation only: replay token IDs through the production native
// decoder and state, exporting its logits without introducing another sampler.
#include "families/lfm2_moe/runtime/chat_templates.h"
#include "families/lfm2_moe/runtime/hybrid_state.h"
#include "families/lfm2_moe/runtime/plugin_helpers.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        const bool encode_only = argc == 4 && std::string(argv[1]) == "--encode-prompt";
        if (argc != 2 && !encode_only)
            throw std::invalid_argument(
                "usage: lfm25_moe_logits_probe [--encode-prompt] bundle [prompt]");
        trtmc::BundleReader bundle(argv[encode_only ? 2 : 1]);
        const auto cfg =
            nlohmann::json::parse(trtmc::lfm2_moe::require_text_section(bundle, "runtime.json"));
        if (encode_only) {
            trtmc::Lfm2MoeTextGenConfig generation;
            generation.id_bos = cfg.at("bos_token_id");
            trtmc::lfm2_moe::apply_chat_template(bundle, generation);
            auto tokenizer = trtmc::lfm2_moe::create_tokenizer(bundle);
            std::string rendered = argv[3];
            if (!generation.chat_template_format.empty())
                rendered =
                    trtmc::lfm2_moe_apply_chat_template(generation.chat_template_format, rendered);
            auto ids = tokenizer->encode(rendered);
            if (ids.size() >= 2 && generation.id_bos >= 0 && ids[0] == generation.id_bos &&
                ids[1] == generation.id_bos)
                ids.erase(ids.begin());
            if (ids.empty() && generation.id_bos >= 0)
                ids.push_back(generation.id_bos);
            std::cout << nlohmann::json({{"prompt_ids", ids}}).dump() << std::endl;
            return 0;
        }
        std::unique_ptr<trtmc::IBackend, decltype(&trtmc_destroy_backend)> backend(
            trtmc_create_backend(), trtmc_destroy_backend);
        auto module = trtmc::lfm2_moe::load_module(
            *backend, trtmc::lfm2_moe::require_section(bundle, "engine.plan"));
        const auto stream = module->stream();
        const auto dtype = trtmc::lfm2_moe::state_dtype(cfg.at("precision"));
        auto kv = std::make_unique<trtmc::Lfm2MoeKvCache>(
            cfg.at("num_attention_layers"), cfg.at("max_cache_length"),
            cfg.at("num_key_value_heads"), cfg.at("head_dim"), stream, dtype,
            trtmc::lfm2_moe::kv_names(cfg.at("num_attention_layers")));
        auto conv = std::make_unique<trtmc::Lfm2MoeConvState>(
            cfg.at("num_conv_layers"), cfg.at("hidden_size"), cfg.at("conv_L_cache"), stream,
            dtype);
        trtmc::Lfm2MoeHybridState state(std::move(kv), std::move(conv));
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
