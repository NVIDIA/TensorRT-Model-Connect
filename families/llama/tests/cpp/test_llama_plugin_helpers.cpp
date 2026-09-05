/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/llama/runtime/plugin_helpers.h"
#include "families/llama/runtime/tokenizer.h"
#include "trtmc/bundle.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

void check_ids(const std::vector<std::int32_t>& actual, const std::vector<std::int32_t>& expected,
               const char* name) {
    check(actual == expected, name);
}

std::string tokenizer_json() {
    return R"({
      "model": {
        "type": "BPE",
        "unk_token": "<unk>",
        "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2, "\u2581": 3, "h": 4, "e": 5},
        "merges": []
      },
      "added_tokens": [
        {"id": 1, "content": "<s>", "special": true},
        {"id": 2, "content": "</s>", "special": true}
      ],
      "pre_tokenizer": {
        "type": "Metaspace", "replacement": "\u2581", "add_prefix_space": true
      },
      "post_processor": {
        "type": "TemplateProcessing",
        "single": [
          {"SpecialToken": {"id": "<s>", "type_id": 0}},
          {"Sequence": {"id": "A", "type_id": 0}}
        ]
      }
    })";
}

std::filesystem::path write_tokenizer_bundle(std::string_view tokenizer) {
    char path[] = "/tmp/trtmc_llama_plugin_helpers_XXXXXX";
    const int descriptor = mkstemp(path);
    if (descriptor < 0)
        throw std::runtime_error("mkstemp failed");
    close(descriptor);
    const std::string header =
        R"({"format":1,"family":"llama","task":"text_generation","backend":"trt","sections":{"tokenizer.json":{"offset":0,"length":)" +
        std::to_string(tokenizer.size()) + "}}}";
    static constexpr unsigned char magic[8] = {'B', 'U', 'N', 'D', 'L', 'E', '\x01', '\0'};
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(magic), sizeof(magic));
    const auto header_size = static_cast<std::uint64_t>(header.size());
    for (int shift = 0; shift < 64; shift += 8)
        output.put(static_cast<char>((header_size >> shift) & 0xffU));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write(tokenizer.data(), static_cast<std::streamsize>(tokenizer.size()));
    output.close();
    return path;
}

void test_exact_special_frame_uses_bundle_tokenizer() {
    const std::string json = tokenizer_json();
    const auto path = write_tokenizer_bundle(json);
    try {
        const trtmc::BundleReader bundle(path.string());
        const auto tokenizer = trtmc::llama::create_tokenizer(bundle);
        check(tokenizer != nullptr, "create tokenizer from current file-backed bundle");
        check_ids(tokenizer->encode("he"), {1, 3, 4, 5},
                  "exact post-processor frame adds BOS without fallback EOS");
    } catch (...) {
        std::filesystem::remove(path);
        throw;
    }
    std::filesystem::remove(path);
}

void test_exact_special_frame_respects_add_special_false() {
    const std::string json = tokenizer_json();
    const auto tokenizer = trtmc::CreateBpeTokenizer(json.data(), json.size(), false);
    check(tokenizer != nullptr, "create tokenizer with special frame disabled");
    check_ids(tokenizer->encode("he"), {3, 4, 5},
              "exact post-processor frame respects add_special_tokens=false");
}

} // namespace

int main() {
    test_exact_special_frame_uses_bundle_tokenizer();
    test_exact_special_frame_respects_add_special_false();
    if (failures != 0)
        std::cerr << failures << " Llama plugin-helper test(s) failed\n";
    return failures;
}
