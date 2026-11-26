/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "families/hunyuan_vl/runtime/contract.h"
#include "families/hunyuan_vl/runtime/json.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using namespace trtmc::hunyuan_vl;
using trtmc::DType;
int failures = 0;
void expect(bool condition, const char* what) {
    if (!condition) {
        std::cerr << "FAILED: " << what << '\n';
        ++failures;
    }
}
// A small bundle description in the layout the builder writes.
nlohmann::json metadata() {
    return {
        {"format", 1},
        {"precision", "bf16"},
        {"max_cache_length", 64},
        {"max_input_length", 16},
        {"max_patches", 64},
        {"hidden_size", 8},
        {"num_hidden_layers", 2},
        {"num_key_value_heads", 2},
        {"head_dim", 8},
        {"vocab_size", 32},
        {"rope_inverse_frequencies", {1.0, 0.5, 0.25, 0.125}},
        {"rope_sections", {1, 1, 1, 1}},
        {"patch_size", 2},
        {"merge_size", 2},
        {"position_edge", 4},
        {"image",
         {{"min_pixels", 16},
          {"max_pixels", 256},
          {"mean", {0.5, 0.5, 0.5}},
          {"std", {0.5, 0.5, 0.5}},
          {"rescale_factor", 1.0 / 255}}},
        {"tokens",
         {{"bos", 1},
          {"system_end", 2},
          {"user", 3},
          {"assistant", 4},
          {"image", 5},
          {"image_start", 6},
          {"image_end", 7},
          {"eos", {8, 9}}}},
    };
}
constexpr size_t kEmbeddingBytes = 8 * 32 * sizeof(float);
bool rejected(const nlohmann::json& document, size_t bytes = kEmbeddingBytes) {
    try {
        parse_metadata(document, bytes);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}
// The tensors the backend reports for the engines the builder makes from `m`: dynamic inputs by
// their profile range, static inputs and outputs by shape (outputs at the largest inputs).
std::vector<EngineTensor> decoder_engine(const Metadata& m) {
    std::vector<EngineTensor> result{
        {"inputs_embeds", m.dtype, true, true, {}, {1, m.hidden}, {m.max_input, m.hidden}},
        {"rope_cos", DType::kFloat32, true, true, {}, {1, m.head_dim}, {m.max_input, m.head_dim}},
        {"rope_sin", DType::kFloat32, true, true, {}, {1, m.head_dim}, {m.max_input, m.head_dim}},
        {"attention_mask",
         DType::kFloat32,
         true,
         true,
         {},
         {1, 1, m.capacity + 1},
         {1, m.max_input, m.capacity + m.max_input}},
        {"logits", DType::kFloat32, false, false, {1, m.vocab}, {}, {}},
    };
    for (int i = 0; i < m.layers; ++i)
        for (const char* kind : {"k_", "v_"}) {
            result.push_back({std::string("cache_") + kind + std::to_string(i),
                              m.dtype,
                              true,
                              false,
                              {m.kv_heads, m.capacity, m.head_dim},
                              {},
                              {}});
            result.push_back({std::string("present_") + kind + std::to_string(i),
                              m.dtype,
                              false,
                              false,
                              {m.kv_heads, m.max_input, m.head_dim},
                              {},
                              {}});
        }
    return result;
}
std::vector<EngineTensor> visual_engine(const Metadata& m) {
    const int64_t width = 3 * int64_t(m.image.patch) * m.image.patch;
    return {
        {"pixel_values", m.dtype, true, true, {}, {4, width}, {m.max_patches, width}},
        {"position_ids", DType::kInt32, true, true, {}, {4, 4}, {m.max_patches, 4}},
        {"position_lambdas", DType::kFloat32, true, true, {}, {4, 4}, {m.max_patches, 4}},
        {"image_features", m.dtype, false, false, {m.max_patches / 4 + 3, m.hidden}, {}, {}},
    };
}
bool text_rejected(const std::string& text) {
    try {
        parse_json(text, "runtime.json");
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}
bool engine_rejected(const std::vector<EngineTensor>& tensors,
                     const std::vector<TensorContract>& contract) {
    try {
        check_engine("test.plan", tensors, contract);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}
EngineTensor& named(std::vector<EngineTensor>& tensors, const std::string& name) {
    return *std::find_if(tensors.begin(), tensors.end(),
                         [&](const auto& tensor) { return tensor.name == name; });
}
} // namespace

int main() {
    const auto m = parse_metadata(metadata(), kEmbeddingBytes);
    expect(m.sections == std::vector<int>{1, 1, 1, 1} && m.dtype == DType::kBFloat16 &&
               m.eos == std::vector<int32_t>{8, 9},
           "the built metadata is accepted");

    // Integer fields are integers in range; RoPE sections are positive and cover head_dim / 2.
    auto j = metadata();
    j["rope_sections"] = {-1, 3, 1, 1};
    expect(rejected(j), "a negative RoPE section whose sum still covers head_dim / 2");
    j = metadata();
    j["hidden_size"] = 8.0;
    expect(rejected(j), "a fractional integer field");
    j = metadata();
    j["num_hidden_layers"] = true;
    expect(rejected(j), "a boolean integer field");
    j = metadata();
    j["max_input_length"] = 65;
    expect(rejected(j), "an input chunk longer than the cache");
    j = metadata();
    j["tokens"]["eos"] = {32};
    expect(rejected(j), "an end-of-sequence id outside the vocabulary");
    j = metadata();
    j["tokens"]["eos"] = nlohmann::json::array();
    expect(rejected(j), "no end-of-sequence id");
    j = metadata();
    j["image"]["std"] = {1e-37, 0.5, 0.5};
    j["image"]["rescale_factor"] = 1.0;
    expect(rejected(j), "a normalization that overflows float32");
    j = metadata();
    j["image"]["std"] = {1e38, 0.5, 0.5};
    expect(rejected(j), "a standard deviation whose folded value overflows float32");
    // White pixels at 3.4e38 are finite floats but infinite bfloat16 values.
    j = metadata();
    j["image"]["mean"] = {0.0, 0.0, 0.0};
    j["image"]["std"] = {7.5e-37, 0.5, 0.5};
    j["image"]["rescale_factor"] = 1.0;
    expect(rejected(j), "a normalization that overflows bfloat16");
    j["precision"] = "fp32";
    expect(!rejected(j), "the same normalization in a float32 bundle");
    expect(contract::bfloat16_round(1.00390625F) == 1.0F &&
               contract::bfloat16_round(1.01171875F) == 1.015625F &&
               contract::bfloat16_round(-3.4e38F) == -std::numeric_limits<float>::infinity() &&
               contract::bfloat16_round(3.38e38F) < std::numeric_limits<float>::infinity(),
           "bfloat16 rounding: ties to even, overflow to infinity");
    j = metadata();
    j["image"]["mean"] = {"0.5", 0.5, 0.5};
    expect(rejected(j), "a string image mean");
    expect(rejected(metadata(), kEmbeddingBytes - 4), "an embedding table of another size");
    j = metadata();
    j["rope_inverse_frequencies"] = {1.0, 0.5, 0.25, 1e-40};
    expect(rejected(j), "a subnormal RoPE frequency");
    j["rope_inverse_frequencies"] = {1.0, 0.5, 0.25, 0.0};
    expect(rejected(j), "a zero RoPE frequency");
    j["rope_inverse_frequencies"] = {1.0, 0.5, 0.25, 1e37};
    expect(rejected(j), "RoPE angles that overflow float within the cache");
    j["rope_inverse_frequencies"] = {1.0, 0.5, 0.25, 1e36};
    expect(!rejected(j), "large RoPE angles that stay finite within the cache");

    // runtime.json text: a repeated key is an error, not a silent override.
    expect(parse_json(metadata().dump(), "runtime.json") == metadata(), "the metadata round-trips");
    expect(text_rejected(R"({"format": 1, "format": 2})"), "a repeated top-level key");
    expect(text_rejected(R"({"image": {"std": [1, 1, 1], "std": [0, 0, 0]}})"),
           "a repeated nested key");
    expect(!text_rejected(R"([{"eos": 1}, {"eos": 2}])"), "one key in two objects");

    // Engines: exactly the bound tensors, with their dtypes, fixed dimensions and profiles.
    const auto decoder = decoder_contract(m);
    expect(!engine_rejected(decoder_engine(m), decoder), "the built decoder is accepted");
    auto engine = decoder_engine(m);
    named(engine, "cache_k_0").shape = {m.kv_heads, m.capacity - 1, m.head_dim};
    expect(engine_rejected(engine, decoder), "a cache shorter than the bound buffer");
    engine = decoder_engine(m);
    named(engine, "logits").dtype = DType::kBFloat16;
    expect(engine_rejected(engine, decoder), "logits in another dtype");
    engine = decoder_engine(m);
    named(engine, "present_v_1").shape = {m.kv_heads, m.max_input, m.head_dim + 2};
    expect(engine_rejected(engine, decoder), "a present cache of another head size");
    engine = decoder_engine(m);
    named(engine, "present_k_0").shape = {m.kv_heads, m.max_input - 1, m.head_dim};
    expect(engine_rejected(engine, decoder), "a present cache shorter than the fed rows");
    engine = decoder_engine(m);
    engine.erase(std::find_if(engine.begin(), engine.end(),
                              [](const auto& tensor) { return tensor.name == "present_v_1"; }));
    expect(engine_rejected(engine, decoder), "a missing output");
    engine = decoder_engine(m);
    named(engine, "inputs_embeds").profile_max = {m.max_input - 1, m.hidden};
    expect(engine_rejected(engine, decoder), "a profile below the prefill chunk");
    engine = decoder_engine(m);
    named(engine, "attention_mask").dynamic = false;
    expect(engine_rejected(engine, decoder), "a static mask the runtime resizes");
    const auto visual = visual_contract(m);
    expect(!engine_rejected(visual_engine(m), visual), "the built vision tower is accepted");
    engine = visual_engine(m);
    named(engine, "position_ids").dtype = DType::kFloat32;
    expect(engine_rejected(engine, visual), "position ids in another dtype");
    engine = visual_engine(m);
    named(engine, "pixel_values").profile_max = {m.max_patches - 4, 12};
    expect(engine_rejected(engine, visual), "a profile below the patch bound");
    for (const int64_t rows : {int64_t(42), int64_t(m.max_patches / 4 + 2)}) {
        engine = visual_engine(m);
        named(engine, "image_features").shape = {rows, m.hidden};
        expect(engine_rejected(engine, visual), "image features without one row per merged cell");
    }
    return failures == 0 ? 0 : 1;
}
