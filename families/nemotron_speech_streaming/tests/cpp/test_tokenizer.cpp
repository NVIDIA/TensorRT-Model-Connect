/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/nemotron_speech_streaming/runtime/plugin_helpers.h"

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

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

std::filesystem::path write_bundle(std::string_view tokenizer) {
    static constexpr std::string_view runtime =
        R"({"tokenizer_add_special_tokens":false,"tokenizer_prefix_ids":[],"tokenizer_suffix_ids":[]})";
    char path[] = "/tmp/trtmc_nemotron_speech_streaming_tokenizer_XXXXXX";
    const int descriptor = mkstemp(path);
    if (descriptor < 0)
        throw std::runtime_error("mkstemp failed");
    close(descriptor);

    const std::string header =
        R"({"format":1,"family":"nemotron_speech_streaming","task":"speech_to_text","backend":"trt","sections":{"runtime.json":{"offset":0,"length":)" +
        std::to_string(runtime.size()) + R"(},"tokenizer.json":{"offset":)" +
        std::to_string(runtime.size()) + R"(,"length":)" + std::to_string(tokenizer.size()) + "}}}";
    static constexpr unsigned char magic[8] = {'B', 'U', 'N', 'D', 'L', 'E', '\x01', '\0'};
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(magic), sizeof(magic));
    const auto header_size = static_cast<std::uint64_t>(header.size());
    for (int shift = 0; shift < 64; shift += 8)
        output.put(static_cast<char>((header_size >> shift) & 0xffU));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write(runtime.data(), static_cast<std::streamsize>(runtime.size()));
    output.write(tokenizer.data(), static_cast<std::streamsize>(tokenizer.size()));
    output.close();
    return path;
}

std::shared_ptr<trtmc::ITokenizer> load_tokenizer(std::string_view tokenizer_json) {
    const auto path = write_bundle(tokenizer_json);
    try {
        const trtmc::BundleReader bundle(path.string());
        auto tokenizer = trtmc::create_tokenizer_from_bundle(bundle);
        std::filesystem::remove(path);
        return tokenizer;
    } catch (...) {
        std::filesystem::remove(path);
        throw;
    }
}

bool load_throws(std::string_view tokenizer_json) {
    try {
        (void)load_tokenizer(tokenizer_json);
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

} // namespace

int main() {
    auto tokenizer = load_tokenizer(R"({
      "model": {
        "type": "BPE",
        "vocab": {"h": 0, "i": 1, "hi": 2},
        "merges": ["h i"]
      }
    })");
    const std::vector<std::int32_t> expected{2};
    check(tokenizer->encode("hi") == expected, "BPE object vocab encodes a merged token");
    check(tokenizer->decode(expected) == "hi", "BPE object vocab decodes the merged token");
    check(load_throws(R"({"model":{"type":"WordPiece","vocab":{"hi":0}}})"),
          "unsupported tokenizer model.type fails closed");
    return failures;
}
