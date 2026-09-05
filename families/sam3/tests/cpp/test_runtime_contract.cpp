/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/sam3/runtime/plugin_helpers.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
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

std::filesystem::path write_bundle(std::string_view runtime) {
    static constexpr std::string_view tokenizer = R"({
      "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}
    })";
    char path[] = "/tmp/trtmc_sam3_runtime_XXXXXX";
    const int descriptor = mkstemp(path);
    if (descriptor < 0)
        throw std::runtime_error("mkstemp failed");
    close(descriptor);

    const std::string header =
        R"({"format":1,"family":"sam3","task":"text_prompted_segmentation","backend":"trt","sections":{"runtime.json":{"offset":0,"length":)" +
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

std::shared_ptr<trtmc::ITokenizer> load_tokenizer(std::string_view runtime) {
    const auto path = write_bundle(runtime);
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

bool load_throws(std::string_view runtime) {
    try {
        (void)load_tokenizer(runtime);
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

} // namespace

int main() {
    auto tokenizer = load_tokenizer(
        R"({"tokenizer_add_special_tokens":false,"tokenizer_prefix_ids":[7],"tokenizer_suffix_ids":[8]})");
    check(tokenizer->encode("a") == std::vector<std::int32_t>({7, 0, 8}),
          "runtime applies the explicit SAM3 BOS/EOS frame exactly once");
    check(load_throws(R"({"tokenizer_prefix_ids":[7],"tokenizer_suffix_ids":[8]})"),
          "missing tokenizer_add_special_tokens fails closed");
    check(load_throws(R"({"tokenizer_add_special_tokens":false,"tokenizer_suffix_ids":[8]})"),
          "missing tokenizer_prefix_ids fails closed");
    check(
        load_throws(
            R"({"tokenizer_add_special_tokens":0,"tokenizer_prefix_ids":[7],"tokenizer_suffix_ids":[8]})"),
        "non-boolean tokenizer_add_special_tokens fails closed");
    return failures;
}
