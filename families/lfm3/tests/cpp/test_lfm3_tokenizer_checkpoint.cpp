/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/lfm3/runtime/plugin_helpers.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <unistd.h>
#include <utility>
#include <vector>

int main() {
    const char* path = std::getenv("TRTMC_LFM3_TOKENIZER_JSON");
    if (path == nullptr || *path == '\0') {
        std::cout << "SKIP: supply pinned LFM3 tokenizer.json via TRTMC_LFM3_TOKENIZER_JSON\n";
        return 77;
    }
    std::ifstream input(path, std::ios::binary);
    if (!input)
        return 1;
    const std::string json((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
    const auto bundle_path = std::filesystem::temp_directory_path() /
                             ("trtmc-lfm3-tokenizer-" + std::to_string(getpid()) + ".bundle");
    const std::string header =
        R"({"format":1,"family":"lfm3","task":"text_generation","backend":"trt","sections":{"tokenizer.json":{"offset":0,"length":)" +
        std::to_string(json.size()) + "}}}";
    {
        std::ofstream output(bundle_path, std::ios::binary);
        output.write("BUNDLE\x01\0", 8);
        const auto size = static_cast<std::uint64_t>(header.size());
        for (int shift = 0; shift < 64; shift += 8)
            output.put(static_cast<char>((size >> shift) & 0xffU));
        output << header << json;
        if (!output)
            return 1;
    }
    const trtmc::BundleReader bundle(bundle_path.string());
    auto tokenizer = trtmc::lfm3::create_tokenizer(bundle);
    std::filesystem::remove(bundle_path);
    // Note (Jiaxin Deng): IDs are from LiquidAI/LFM3-1.7B revision
    // ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7.
    const std::vector<std::pair<std::string, std::vector<int32_t>>> fixtures{
        {"Hello, world!", {35808, 20, 1530, 9}},
        {"Café déjà vu.", {43, 1804, 359, 18412, 23665, 22}},
        {"¿Dónde está París?", {17494, 44, 58337, 6952, 35319, 39}},
        {"中文标点，测试123。", {108820, 9756, 5837, 697, 57331, 9792, 646}},
        {"日本語・テスト123", {62005, 2427, 108526, 9792}},
        {"한국어 테스트 123", {60581, 4692, 110799, 229, 9792}},
        {"Привет, мир! 123", {6744, 12827, 20, 48681, 9, 229, 9792}},
        {"مرحبا، العالم ١٢٣", {17880, 2262, 68059, 3279, 35772, 229, 71062, 71727, 101606}},
        {"नमस्ते दुनिया १२३",
         {79861, 72929, 73086, 49507, 72979, 73522, 73016, 29669, 229, 84312, 84378, 85888}},
        {"Grüße, Welt! 123", {10205, 641, 6516, 20, 9307, 9, 229, 9792}},
        {"Bonjour, ça va ? 123", {34432, 12642, 20, 18256, 3722, 4718, 229, 9792}},
        {"Việt", {74657, 73301}},
        {"Ελληνικά,", {84574, 83716, 86424, 85528, 20}},
        {"שלום,", {87294, 83860, 20}},
        {"ภาษาไทย!", {111064, 73809, 9}},
        {"ＡＢＣ,", {70368, 110353, 72199, 20}},
        {"ｶﾀｶﾅ!", {11991, 123, 56673, 231, 11991, 123, 56673, 236, 9}},
        {"²½Ⅻ", {8387, 14583, 46236, 113}},
        {"Հայերեն!", {85701, 83297, 83263, 83768, 9}},
        {"ქართული!", {84436, 85222, 84702, 9}},
        {"I'M", {49, 50750}},
        {"WE'RE", {23934, 15, 2776}},
        {"HE'LL", {5846, 15, 4591}},
        {"DON'T", {44, 1831, 32139}},
        {"SHE'S", {59, 5846, 23905}},
        {"a  b", {73, 229, 293}},
        {"a  5", {73, 229, 229, 29}},
        {"a \t b", {73, 45981, 293}},
        {"trailing  ", {18809, 7705, 266}},
        {"def f():\n    return  x", {3162, 291, 32210, 360, 1288, 229, 2552}},
        {"| a |  b |", {100, 267, 1670, 229, 293, 1670}},
        {"nclu", {64112}},
        {"Counte", {65281}},
        {"Sectio", {65387}},
        {":\nA) Paris", {833, 41, 17, 4741}},
        {"<|startoftext|><|im_start|>user\nWhat is the capital of France? Answer in one "
         "word.<|im_end|>\n<|im_start|>assistant\n<think></think>",
         {124894, 124899, 5922, 207, 2992,   355, 278,    5205,  302, 3980,   39,    41774,
          296,    734,    1858, 22,  124900, 207, 124899, 63514, 207, 124901, 124902}},
    };
    int failures = 0;
    for (const auto& [text, ids] : fixtures) {
        if (tokenizer->encode(text) != ids) {
            std::cerr << "FAIL: checkpoint tokenizer IDs: " << text << '\n';
            ++failures;
        }
    }
    if (tokenizer->decode({833, 41, 17, 4741}) != ":\nA) Paris")
        ++failures;
    return failures;
}
