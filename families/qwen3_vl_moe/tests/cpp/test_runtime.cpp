/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "fakes.h"
#include "families/qwen3_vl_moe/runtime/lora_peft_loader.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using test::rejects;
using test::require;
using namespace trtmc;

void test_pipeline() {
    auto decoder = std::make_unique<test::Module>();
    auto vision = std::make_unique<test::Module>();
    auto state = std::make_unique<test::State>();
    auto* module = decoder.get();
    auto* cache = state.get();
    auto tokenizer = std::make_shared<test::Tokenizer>();
    Qwen3VlMoeConfig config;
    config.id_eos = 99;
    config.vision_output_dim = 2;
    config.prefill_max_length = 64;
    Qwen3VlMoePreprocessConfig image_config;
    image_config.fixed_image_size = 28;
    Qwen3VlMoePipeline pipeline(std::move(decoder), std::move(vision), std::move(state), config,
                                image_config, nullptr, tokenizer);
    TextGenerationConfig options;
    options.max_new_tokens = 1;
    auto out = pipeline.generate_ids({0, 0}, options);
    require(out.token_ids.size() == 3 && module->calls == 2, "executed discarded final logits");
    options.max_new_tokens = 2;
    module->calls = 0;
    pipeline.generate_ids({0, 0}, options);
    require(module->calls == 3, "decode did not preserve the preceding step");
    const auto calls = module->calls;
    const auto resets = cache->resets;
    rejects([&] { pipeline.generate_ids({0, 0, 0, 0}, options); });
    rejects([&] { pipeline.generate_ids({0, 0, 0, 0, 0, 0}, options); });
    require(module->calls == calls && cache->resets == resets, "over-budget request began prefill");
    options.max_new_tokens = 0;
    require(pipeline.generate_ids({0, 0}, options).token_ids.size() == 2,
            "zero budget changed output");
    options.max_new_tokens = 1;
    require(pipeline.generate_ids({}, options).token_ids.empty(), "empty prefix changed output");
    require(module->calls == calls, "empty or zero-budget request executed");
    module->calls = 0;
    pipeline.generate_ids({0, 0, 0, 0}, options);
    require(module->calls == 4, "exact capacity budget was not honored");

    ITask* identity = &pipeline;
    auto* model = dynamic_cast<internal::IModel*>(identity);
    require(model != nullptr, "Task SDK cannot identify the loaded model");
    const auto tasks = model->task_bindings();
    require(tasks.size() == 2 && model->lora_adapters() == nullptr, "wrong SDK capabilities");
    const internal::ConfigEntry entry{"max_new_tokens", std::int64_t{1}};
    const internal::ConfigView settings{&entry, 1};
    internal::TextContinuationRequest prefix{std::string_view{"hello"}};
    require(pipeline.run(prefix, settings).text == "x", "SDK continuation failed");
    const std::uint8_t pixels[]{0, 127, 255};
    const internal::ImageView image{pixels, 3, 1, 1, 3, internal::ImageFormat::UInt8};
    const internal::ImagesTextPart parts[]{image, internal::TextPartView{"color?"}};
    const internal::ImagesTextMessage turn{internal::MessageRole::User, {parts, 2}};
    const internal::ImagesTextToTextRequest request{{&turn, 1}};
    module->calls = 0;
    require(pipeline.run(request, settings).text == "x" && module->calls == 2,
            "SDK image generation or final-token stopping failed");
    tokenizer->tokens = 5;
    module->calls = 0;
    rejects([&] { pipeline.run(request, settings); });
    require(module->calls == 0, "oversized image prompt began prefill");
    const internal::ConfigEntry invalid{"top_k", std::int64_t{0}};
    rejects([&] { pipeline.run(request, {&invalid, 1}); });
    const internal::ImagesTextPart repeated[]{image, image};
    const internal::ImagesTextMessage repeated_turn{internal::MessageRole::User, {repeated, 2}};
    rejects(
        [&] { pipeline.run(internal::ImagesTextToTextRequest{{&repeated_turn, 1}}, settings); });
}

void test_unmapped_adapter() {
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory = std::filesystem::temp_directory_path() /
                           ("trtmc-qwen3-vl-moe-adapter-" + std::to_string(unique));
    std::filesystem::create_directory(directory);
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code error;
            std::filesystem::remove_all(path, error);
        }
    } cleanup{directory};
    std::ofstream(directory / "adapter_config.json") << nlohmann::json({
        {"r", 1},
        {"lora_alpha", 1},
        {"target_modules", {"q_proj", "down_proj"}},
    });
    const auto write_weights = [&](bool expert) {
        nlohmann::json header;
        const std::string base = "base_model.model.model.language_model.layers.0.";
        header[base + "self_attn.q_proj.lora_A.weight"] = {
            {"dtype", "F32"}, {"shape", {1, 2}}, {"data_offsets", {0, 8}}};
        header[base + "self_attn.q_proj.lora_B.weight"] = {
            {"dtype", "F32"}, {"shape", {2, 1}}, {"data_offsets", {8, 16}}};
        if (expert)
            header[base + "mlp.experts.0.down_proj.lora_A.weight"] = {
                {"dtype", "F32"}, {"shape", {1, 2}}, {"data_offsets", {16, 24}}};
        auto text = header.dump();
        while (text.size() % 8 != 0)
            text += ' ';
        std::ofstream output(directory / "adapter_model.safetensors", std::ios::binary);
        for (unsigned shift = 0; shift < 64; shift += 8)
            output.put(static_cast<char>(static_cast<std::uint64_t>(text.size()) >> shift));
        output.write(text.data(), static_cast<std::streamsize>(text.size()));
        const std::string zeros(expert ? 24 : 16, '\0');
        output.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
    };
    const std::vector<TensorInfo> contract{
        {"lora_a_layer_0_w_q", {2, 1}, DType::kFloat32},
        {"lora_b_layer_0_w_q", {1, 2}, DType::kFloat32},
    };
    write_weights(false);
    require(qwen3_vl_moe_load_peft_lora_adapter(directory.string(), contract).buffers.size() == 2,
            "valid attention adapter was rejected");
    write_weights(true);
    rejects([&] { qwen3_vl_moe_load_peft_lora_adapter(directory.string(), contract); });
}

void test_full_cache() {
    test::Module module;
    module.rank = 2;
    module.shape = {2, 4};
    Qwen3VlMoeKvCache cache(1, 2, 4, nullptr);
    require(cache.ok(), "tiny cache allocation failed");
    cache.bind_to(module);
    const float row[]{1, 2, 3, 4};
    for (int step = 0; step < 2; ++step) {
        require(cudaMemcpy(module.external.at("present_k_0"), row, sizeof(row),
                           cudaMemcpyHostToDevice) == cudaSuccess,
                "present K copy failed");
        require(cudaMemcpy(module.external.at("present_v_0"), row, sizeof(row),
                           cudaMemcpyHostToDevice) == cudaSuccess,
                "present V copy failed");
        cache.advance();
    }
    rejects([&] { cache.advance(); });
    require(cache.position() == 2, "full cache changed position");
    float actual[8]{};
    require(cudaMemcpy(actual, cache.cache_k(0).data(), sizeof(actual), cudaMemcpyDeviceToHost) ==
                cudaSuccess,
            "cache read failed");
    for (int index = 0; index < 8; ++index)
        require(actual[index] == row[index % 4], "full cache changed retained rows");
    cache.cache_v(0) = DeviceTensor{};
    require(!cache.ok(), "invalid V allocation was accepted");
}

int main(int argc, char**) {
    try {
        test_pipeline();
        test_unmapped_adapter();
        if (argc > 1)
            test_full_cache();
        std::cout << "Qwen3-VL MoE runtime regressions passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
