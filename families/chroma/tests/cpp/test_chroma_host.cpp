/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Host-side Chroma runtime math and tokenizer, without a GPU.
//
// test_chroma_host                          built-in cases
// test_chroma_host <tokenizer.json> <cases>  plus tokenizer parity: every line
//                                           of <cases> is {"text", "ids"} from
//                                           the reference tokenizer, optionally
//                                           with "length", "padded_ids", "mask".

#include "families/chroma/runtime/sampling.h"
#include "families/chroma/runtime/t5_tokenizer.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace trtmc::chroma;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

float bits(std::uint32_t value) {
    float result = 0.0F;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

std::uint32_t raw(float value) {
    std::uint32_t result = 0;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

// Reference values: diffusers 0.39.0 FlowMatchEulerDiscreteScheduler(shift=3)
// set as ChromaPipeline does, and the timestep the model sees in BF16.
void test_schedule() {
    const std::vector<std::uint32_t> sigmas4 = {0x3f800000, 0x3f666666, 0x3f400000, 0x3f000000, 0};
    const std::vector<std::uint32_t> bf16_4 = {0x447a0000, 0x44610000, 0x443c0000, 0x43fa0000};
    const std::vector<std::uint32_t> fp32_4 = {0x447a0000, 0x44610000, 0x443b8000, 0x43fa0000};
    const auto four = flow_match_sigmas(4, 3.0F);
    check(four.size() == 5, "4 steps give 5 sigmas");
    for (std::size_t index = 0; index < four.size(); ++index)
        check(raw(four[index]) == sigmas4[index], "sigma " + std::to_string(index) + " of 4 steps");
    for (std::size_t index = 0; index < 4; ++index) {
        const float timestep = four[index] * 1000.0F;
        check(raw(conditioned_timestep(timestep, true)) == bf16_4[index],
              "BF16 timestep of 4 steps");
        check(raw(conditioned_timestep(timestep, false)) == fp32_4[index],
              "FP32 timestep of 4 steps");
    }
    const std::vector<std::uint32_t> sigmas35 = {
        0x3f800000, 0x3f7d83b9, 0x3f7aee41, 0x3f783e11, 0x3f75717c, 0x3f7286bc,
        0x3f6f7be0, 0x3f6c4ec6, 0x3f68fd20, 0x3f658469, 0x3f61e1e1, 0x3f5e1283,
        0x3f5a12f7, 0x3f55df97, 0x3f51745d, 0x3f4cccce, 0x3f47e3f2, 0x3f42b447,
        0x3f3d37a6, 0x3f37672b, 0x3f313b13, 0x3f2aaaab, 0x3f23ac10, 0x3f1c3412,
        0x3f1435e5, 0x3f0ba2ea, 0x3f026a43, 0x3ef0f0f2, 0x3edb6db8, 0x3ec415ca,
        0x3eaaaaac, 0x3e8ee23c, 0x3e60c7cf, 0x3e1d89d9, 0x3da60dd7, 0};
    const std::vector<std::uint32_t> bf16_35 = {
        0x447a0000, 0x44780000, 0x44750000, 0x44720000, 0x44700000, 0x446d0000, 0x446a0000,
        0x44670000, 0x44640000, 0x44600000, 0x445d0000, 0x44590000, 0x44550000, 0x44510000,
        0x444d0000, 0x44480000, 0x44430000, 0x443e0000, 0x44390000, 0x44330000, 0x442d0000,
        0x44270000, 0x44200000, 0x44190000, 0x44110000, 0x44080000, 0x44000000, 0x43eb0000,
        0x43d60000, 0x43bf0000, 0x43a70000, 0x438c0000, 0x435c0000, 0x431a0000, 0x42a20000};
    const auto many = flow_match_sigmas(35, 3.0F);
    check(many.size() == 36, "35 steps give 36 sigmas");
    for (std::size_t index = 0; index < many.size(); ++index)
        check(raw(many[index]) == sigmas35[index],
              "sigma " + std::to_string(index) + " of 35 steps");
    for (std::size_t index = 0; index < 35; ++index)
        check(raw(conditioned_timestep(many[index] * 1000.0F, true)) == bf16_35[index],
              "BF16 timestep " + std::to_string(index) + " of 35 steps");
    const auto one = flow_match_sigmas(1, 3.0F);
    check(one.size() == 2 && one[0] == 1.0F && one[1] == 0.0F, "one step goes from 1 to 0");
    bool threw = false;
    try {
        (void)flow_match_sigmas(0, 3.0F);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "zero steps are rejected");
}

void test_features() {
    // get_timestep_embedding(t, 16, flip_sin_to_cos=True, downscale_freq_shift=0) in float32.
    const std::vector<std::pair<std::uint32_t, std::vector<std::uint32_t>>> cases = {
        {0x447a0000,
         {0x3f0ff813, 0xbef46d80, 0x3f5cc0ad, 0x3f7a8afc, 0xbf56cd64, 0xbf7ff1fb, 0x3f0a5141,
          0x3f734e6a, 0x3f53ae61, 0x3f60f1a5, 0xbf01a19c, 0x3e524e03, 0xbf0b44f8, 0xbca96e97,
          0x3f576aa4, 0x3e9f3940}},
        {0x43a6aaaa,
         {0x3f72a3de, 0x3e29190a, 0xbeadee14, 0xbee0d805, 0xbf7b4efe, 0x3efcefd4, 0x3f71e8b3,
          0x3f7e943f, 0x3ea33d4d, 0xbf7c7c42, 0x3f70c6a9, 0xbf65fe81, 0xbe43242b, 0x3f5e946a,
          0x3ea7860f, 0x3dd77a87}},
        {0x41480000,
         {0x3f7f6fb5, 0xbf304765, 0x3ea171f3, 0x3f6c4252, 0x3f7e00ab, 0x3f7fccce, 0x3f7ffae1,
          0x3f7fff7d, 0xbd87d3c6, 0xbf39a387, 0x3f72f0a7, 0x3ec52805, 0x3dff5577, 0x3d21ddcd,
          0x3c4ccb6e, 0x3b8186ce}},
    };
    for (const auto& [timestep, expected] : cases) {
        const auto features = timestep_features(bits(timestep));
        for (std::size_t index = 0; index < 16; ++index)
            check(std::fabs(features[index] - bits(expected[index])) <= 2e-6F,
                  "timestep feature " + std::to_string(index) +
                      " at t=" + std::to_string(bits(timestep)));
    }
}

void test_numerics() {
    check(round_to_bfloat16(1.0F) == 1.0F, "bf16(1) = 1");
    check(round_to_bfloat16(bits(0x3f808000)) == 1.0F, "bf16 tie rounds to even (down)");
    check(raw(round_to_bfloat16(bits(0x3f818000))) == 0x3f820000, "bf16 tie rounds to even (up)");
    check(round_to_bfloat16(997.123F) == 996.0F, "bf16(997.123) = 996");
    check(std::isnan(round_to_bfloat16(std::nanf(""))), "bf16 keeps NaN");

    const auto mask = denoiser_text_mask({1, 1, 1, 0, 0, 0});
    check(mask == std::vector<float>({1, 1, 1, 1, 0, 0}), "the first padding token stays visible");
    check(denoiser_text_mask({1, 1, 1}) == std::vector<float>({1, 1, 1}),
          "a full prompt keeps all");

    const LatentShape shape{2, 4, 6};
    std::vector<float> latents(shape.count());
    for (std::size_t index = 0; index < latents.size(); ++index)
        latents[index] = static_cast<float>(index);
    const auto packed = pack_latents(latents, shape);
    check(shape.tokens() == 6, "2x2 packing of 4x6 gives 6 tokens");
    // Token (row 1, column 2), channel 1, dy 1, dx 0 = latents[1][3][4].
    const std::size_t token = 1 * 3 + 2;
    check(packed[token * 8 + 1 * 4 + 1 * 2 + 0] == latents[(1 * 4 + 3) * 6 + 4], "pack layout");
    check(unpack_latents(packed, shape) == latents, "unpack inverts pack");
    check(initial_noise(5, shape) == initial_noise(5, shape), "seeded noise is deterministic");
    check(initial_noise(5, shape) != initial_noise(6, shape), "seeds differ");

    std::vector<float> positive = {1.0F, 2.0F};
    apply_guidance(positive, {0.5F, 3.0F}, 3.0F, false);
    check(positive == std::vector<float>({2.0F, 0.0F}), "guidance");
    std::vector<float> state = {1.0F, -1.0F};
    euler_step(state, {2.0F, 4.0F}, 1.0F, 0.75F, false);
    check(state == std::vector<float>({0.5F, -2.0F}), "Euler step");
    const auto rgb = to_rgb({-1.0F, 1.0F, 0.0F, 3.0F, -3.0F, 0.5F}, 1, 2, false);
    check(rgb == std::vector<float>({0.0F, 0.5F, 0.0F, 1.0F, 1.0F, 0.75F}), "RGB conversion");

    // BF16 bundles round the sampler state where the BF16 reference pipeline does:
    // 1 + bf16(-0.5 * 0.1) = 1 - 0.050048828125 -> bf16 0.94921875 (FP32 keeps 0.95).
    std::vector<float> bf16_state = {1.0F};
    euler_step(bf16_state, {0.1F}, 1.0F, 0.5F, true);
    check(bf16_state[0] == 0.94921875F, "BF16 Euler step rounds the product and the sample");
    std::vector<float> fp32_state = {1.0F};
    euler_step(fp32_state, {0.1F}, 1.0F, 0.5F, false);
    check(fp32_state[0] != 0.94921875F, "FP32 Euler step does not round to BF16");
    std::vector<float> guided = {0.3F};
    apply_guidance(guided, {0.1F}, 3.0F, true);
    check(guided[0] ==
              round_to_bfloat16(0.1F + round_to_bfloat16(3.0F * round_to_bfloat16(0.3F - 0.1F))),
          "BF16 guidance rounds each operation");
    std::vector<float> initial = {0.1F, 1.0F};
    to_model_type(initial, true);
    check(initial[0] == round_to_bfloat16(0.1F) && initial[1] == 1.0F, "BF16 initial latents");
    std::vector<float> vae = {1.0F};
    to_vae_input(vae, 0.3611F, 0.1159F, true);
    check(vae[0] == round_to_bfloat16(round_to_bfloat16(1.0F / 0.3611F) + 0.1159F),
          "BF16 VAE input");
}

std::string synthetic_tokenizer() {
    using nlohmann::json;
    const json vocab = json::array(
        {json::array({"<pad>", 0.0}), json::array({"</s>", 0.0}), json::array({"<unk>", 0.0}),
         json::array({"\xe2\x96\x81", -2.0}),
         json::array({"\xe2\x96\x81"
                      "a",
                      -3.0}),
         json::array({"b", -4.0}),
         json::array({"\xe2\x96\x81"
                      "ab",
                      -5.0}),
         json::array({"a", -4.0}), json::array({"c", -6.0}), json::array({"<x>", 0.0})});
    json added = json::array();
    for (const auto& [content, id] : std::vector<std::pair<std::string, int>>{
             {"<pad>", 0}, {"</s>", 1}, {"<unk>", 2}, {"<x>", 9}})
        added.push_back({{"id", id},
                         {"content", content},
                         {"single_word", false},
                         {"lstrip", false},
                         {"rstrip", false},
                         {"normalized", false},
                         {"special", true}});
    return json{
        {"normalizer", nullptr},
        {"pre_tokenizer",
         {{"type", "Sequence"},
          {"pretokenizers", json::array({{{"type", "WhitespaceSplit"}},
                                         {{"type", "Metaspace"},
                                          {"replacement", "\xe2\x96\x81"},
                                          {"prepend_scheme", "always"},
                                          {"split", true}}})}}},
        {"post_processor",
         {{"type", "TemplateProcessing"},
          {"single", json::array({{{"Sequence", {{"id", "A"}, {"type_id", 0}}}},
                                  {{"SpecialToken", {{"id", "</s>"}, {"type_id", 0}}}}})}}},
        {"model", {{"type", "Unigram"}, {"unk_id", 2}, {"byte_fallback", false}, {"vocab", vocab}}},
        {"added_tokens", added}}
        .dump();
}

void test_tokenizer() {
    const auto text = synthetic_tokenizer();
    const T5Tokenizer tokenizer(text.data(), text.size());
    using Ids = std::vector<std::int32_t>;
    check(tokenizer.encode("ab") == Ids({6, 1}), "best Viterbi path");
    check(tokenizer.encode("a  b") == Ids({4, 3, 5, 1}), "whitespace split and Metaspace prefix");
    check(tokenizer.encode("zz") == Ids({3, 2, 1}), "unknown run is fused into one <unk>");
    check(tokenizer.encode("a<x>b") == Ids({4, 9, 3, 5, 1}), "special tokens split the raw text");
    check(tokenizer.encode("a\xe3\x80\x80"
                           "b") == Ids({4, 3, 5, 1}),
          "U+3000 is whitespace");
    check(tokenizer.encode("a\xe2\x96\x81"
                           "b") == Ids({4, 3, 5, 1}),
          "Metaspace splits a typed U+2581");
    check(tokenizer.encode("") == Ids({1}), "empty prompt is </s>");
    const auto padded = tokenizer.encode_padded("a", 4);
    check(padded.ids == Ids({4, 1, 0, 0}) && padded.mask == Ids({1, 1, 0, 0}) && padded.valid == 2,
          "padding");
    const auto truncated = tokenizer.encode_padded("a b c", 3);
    check(truncated.ids == Ids({4, 3, 1}) && truncated.valid == 3, "truncation keeps </s>");
    bool threw = false;
    try {
        (void)tokenizer.encode("bad \xff byte");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "invalid UTF-8 is rejected");
    auto other = nlohmann::json::parse(text);
    other["normalizer"] = {{"type", "NFKC"}};
    const auto unsupported = other.dump();
    threw = false;
    try {
        T5Tokenizer rejected(unsupported.data(), unsupported.size());
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    check(threw, "a tokenizer with a normalizer is rejected");
}

void test_parity(const char* tokenizer_path, const char* cases_path) {
    std::ifstream json_file(tokenizer_path, std::ios::binary);
    std::stringstream buffer;
    buffer << json_file.rdbuf();
    const auto text = buffer.str();
    const T5Tokenizer tokenizer(text.data(), text.size());
    std::ifstream cases(cases_path);
    std::string line;
    int count = 0;
    while (std::getline(cases, line)) {
        if (line.empty())
            continue;
        const auto record = nlohmann::json::parse(line);
        const auto text = record.at("text").get<std::string>();
        const auto expected = record.at("ids").get<std::vector<std::int32_t>>();
        check(tokenizer.encode(text) == expected, "tokenizer parity: " + text);
        // The pipeline's padding="max_length", truncation=True call at the bundle length.
        if (record.contains("length")) {
            const auto padded =
                tokenizer.encode_padded(text, record.at("length").get<std::size_t>());
            check(padded.ids == record.at("padded_ids").get<std::vector<std::int32_t>>() &&
                      padded.mask == record.at("mask").get<std::vector<std::int32_t>>(),
                  "padded tokenizer parity: " + text);
        }
        ++count;
    }
    check(count > 0, "tokenizer parity cases are present");
    std::cout << "tokenizer parity cases: " << count << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        test_schedule();
        test_features();
        test_numerics();
        test_tokenizer();
        if (argc == 3)
            test_parity(argv[1], argv[2]);
        else if (argc != 1)
            throw std::invalid_argument("usage: test_chroma_host [<tokenizer.json> <cases.jsonl>]");
    } catch (const std::exception& error) {
        std::cerr << "Unexpected: " << error.what() << '\n';
        ++failures;
    }
    std::cerr << (failures ? "SOME FAILED\n" : "ALL PASSED\n");
    return failures ? 1 : 0;
}
