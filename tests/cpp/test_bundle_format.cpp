/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-BDL-CPP-01
// Architecture:   ARCH-BDL-001
// Unit Design:    UD-BDL-01
// Intent:         Bundle magic validation, section parsing, read/write round-trip
// Preconditions:  Valid and invalid .trtfb test data available
// Postconditions: BundleFile correctly populated or error returned
// =============================================================================

// Test suite: .trtfb bundle format reading, magic validation, and error handling.
//
// Purpose:
//   Validates the bundle reader for .trtfb files. Tests cover magic byte
//   validation, error handling for truncated/invalid files, and the
//   IsBundle/InspectBundle utility functions. Write tests are omitted
//   because bundle writing is now handled by the Python tensorrt_model_connect package.
//
// Dependencies:
//   - bundle/bundle_format.h: BundleFile, ReadBundleFile, IsBundle,
//     InspectBundle, kBundleMagic.
//   - Filesystem access (temp directories via mkdtemp).
//   - No TRT, GPU, or CUDA required.

#include "bundle/bundle_format.h"
#include "test_helpers.h"
#include "utils/sha256.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <streambuf>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

static std::filesystem::path make_temp_dir() {
    char pattern[] = "/tmp/trtfb_test_XXXXXX";
    char* dir = mkdtemp(pattern);
    if (dir == nullptr) {
        throw std::runtime_error(std::string("mkdtemp failed: ") + std::strerror(errno));
    }
    return std::filesystem::path(dir);
}

// Helper: write a minimal valid .trtfb file manually (bypasses WriteBundleFile).
static void write_minimal_bundle(const std::string& path, const std::string& header_json) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(trtmc::kBundleMagic), 8);
    uint64_t len = header_json.size();
    unsigned char bytes[8];
    for (int i = 0; i < 8; ++i)
        bytes[i] = static_cast<unsigned char>((len >> (8 * i)) & 0xFF);
    out.write(reinterpret_cast<const char*>(bytes), 8);
    out.write(header_json.data(), static_cast<std::streamsize>(header_json.size()));
}

// Helper: write a bundle with sections
static void write_bundle_with_sections(const std::string& path, const std::string& header_json,
                                       const std::vector<std::vector<char>>& section_data) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(trtmc::kBundleMagic), 8);
    uint64_t len = header_json.size();
    unsigned char bytes[8];
    for (int i = 0; i < 8; ++i)
        bytes[i] = static_cast<unsigned char>((len >> (8 * i)) & 0xFF);
    out.write(reinterpret_cast<const char*>(bytes), 8);
    out.write(header_json.data(), static_cast<std::streamsize>(header_json.size()));
    for (const auto& data : section_data) {
        out.write(data.data(), static_cast<std::streamsize>(data.size()));
    }
}

static void test_read_valid_bundle() {
    const auto tmp = make_temp_dir();
    const auto path = (tmp / "test.trtfb").string();

    const std::string json = R"({
  "model_id": "test-model",
  "model_type": "decoder",
  "family": "generic_text",
  "trt_version": "10.15.0",
  "trt_abi": "10.15",
  "gpu_name": "GeForce RTX 4090",
  "created_at": "2026-02-14T12:00:00Z",
  "vocab_size": 151936,
  "hidden_size": 1024,
  "num_layers": 28,
  "num_attention_heads": 16,
  "num_key_value_heads": 4,
  "max_cache_length": 2048,
  "sections": {
    "engine_plan": {"offset": 0, "size": 4},
    "tokenizer_json": {"offset": 4, "size": 2}
  }
})";

    std::vector<char> plan_data = {'P', 'L', 'A', 'N'};
    std::vector<char> tok_data = {'{', '}'};
    write_bundle_with_sections(path, json, {plan_data, tok_data});

    const auto loaded = trtmc::ReadBundleFile(path);
    check(loaded.info.model_id == "test-model", "read model_id");
    check(loaded.info.model_type == "decoder", "read model_type");
    check(loaded.info.family == "generic_text", "read family");
    check(loaded.info.trt_version == "10.15.0", "read trt_version");
    check(loaded.info.trt_abi == "10.15", "read trt_abi");
    check(loaded.info.vocab_size == 151936, "read vocab_size");
    check(loaded.info.hidden_size == 1024, "read hidden_size");
    check(loaded.info.num_layers == 28, "read num_layers");
    check(loaded.info.num_attention_heads == 16, "read num_attention_heads");
    check(loaded.info.num_key_value_heads == 4, "read num_key_value_heads");
    check(loaded.info.max_cache_length == 2048, "read max_cache_length");
    check(!loaded.info.tokenizer_add_special_tokens_present,
          "read missing tokenizer_add_special_tokens present flag");
    check(loaded.sections.size() == 2, "read section count");
    check(loaded.sections[0].name == "engine_plan", "read section 0 name");
    check(loaded.sections[0].data == plan_data, "read section 0 data");
    check(loaded.sections[1].name == "tokenizer_json", "read section 1 name");
    check(loaded.sections[1].data == tok_data, "read section 1 data");

    trtmc_test::remove_all_safe(tmp);
}

static void test_magic_validation() {
    const auto tmp = make_temp_dir();
    const auto path = (tmp / "bad.trtfb").string();

    std::ofstream out(path, std::ios::binary);
    out.write("NOTMAGIC", 8);
    out.close();

    bool threw = false;
    try {
        trtmc::ReadBundleFile(path);
    } catch (const std::runtime_error& e) {
        threw = true;
        const std::string msg = e.what();
        check(msg.find("magic") != std::string::npos || msg.find("Invalid") != std::string::npos,
              "magic error message is descriptive");
    }
    check(threw, "invalid magic throws");

    trtmc_test::remove_all_safe(tmp);
}

static void test_empty_sections() {
    const auto tmp = make_temp_dir();
    const auto path = (tmp / "empty.trtfb").string();

    const std::string json = R"({"model_id": "empty", "sections": {}})";
    write_minimal_bundle(path, json);

    const auto loaded = trtmc::ReadBundleFile(path);
    check(loaded.info.model_id == "empty", "empty sections model_id");
    check(loaded.sections.empty(), "empty sections count");

    trtmc_test::remove_all_safe(tmp);
}

static void test_is_bundle_valid() {
    const auto tmp = make_temp_dir();
    const auto path = (tmp / "valid.trtfb").string();

    write_minimal_bundle(path, R"({"model_id": "valid"})");
    check(trtmc::IsBundle(path), "IsBundle true for valid file");

    trtmc_test::remove_all_safe(tmp);
}

static void test_is_bundle_invalid() {
    const auto tmp = make_temp_dir();

    const auto text_path = (tmp / "readme.txt").string();
    std::ofstream(text_path) << "Hello world";
    check(!trtmc::IsBundle(text_path), "IsBundle false for text file");
    check(!trtmc::IsBundle(tmp.string()), "IsBundle false for directory");
    check(!trtmc::IsBundle((tmp / "nonexistent").string()), "IsBundle false for nonexistent");

    trtmc_test::remove_all_safe(tmp);
}

static void test_inspect_returns_metadata() {
    const auto tmp = make_temp_dir();
    const auto path = (tmp / "inspect.trtfb").string();

    const std::string json = R"({
  "model_id": "inspectable",
  "vocab_size": 50000,
  "num_layers": 12,
  "sections": {
    "data": {"offset": 0, "size": 3}
  }
})";
    write_bundle_with_sections(path, json, {{'X', 'Y', 'Z'}});

    const auto info = trtmc::InspectBundle(path);
    check(info.model_id == "inspectable", "inspect model_id");
    check(info.vocab_size == 50000, "inspect vocab_size");
    check(info.num_layers == 12, "inspect num_layers");

    trtmc_test::remove_all_safe(tmp);
}

static void test_inspect_is_header_only() {
    const auto tmp = make_temp_dir();
    const auto path = (tmp / "header-only-inspect.trtfb").string();
    // Deliberately declare an engine payload that is not present. Header-only
    // inspection must still succeed; a full bundle read would reject it.
    write_minimal_bundle(path, R"({
  "model_id": "header-only",
  "sections": {
    "engine_plan": {"offset": 1048576, "size": 1048576}
  }
})");

    const auto info = trtmc::InspectBundle(path);
    check(info.model_id == "header-only", "inspect reads header only");
    check(info.sections.size() == 1, "inspect returns engine metadata only");

    bool full_read_rejected = false;
    try {
        (void)trtmc::ReadBundleFile(path);
    } catch (const std::runtime_error&) {
        full_read_rejected = true;
    }
    check(full_read_rejected, "full read rejects the payload that inspect never touches");

    trtmc_test::remove_all_safe(tmp);
}

static void test_runtime_memory_contract_parse_and_legacy_omission() {
    const auto tmp = make_temp_dir();
    const auto path = (tmp / "runtime-memory.trtfb").string();
    const std::string json = R"({
  "model_id": "Qwen/Qwen3-0.6B",
  "precision": "bf16",
  "max_cache_length": 40960,
  "runtime_memory": {
    "contract_version": 1,
    "qualified_model_id": "Qwen/Qwen3-0.6B",
    "qualified_model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    "qualified_config_sha256": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "qualified_target": "gb300-trt-11.2",
    "qualified_runtime_stack": {
      "sm": "sm103",
      "tensorrt": "11.2.0.113",
      "cuda_runtime": "13.3",
      "cudnn_backend": "9.20.0",
      "cudnn_frontend_revision": "7b9b711c22b6823e87150213ecd8449260db8610",
      "nvrtc": "13.3",
      "driver": "580.105.08"
    },
    "native_kv_plugin_abi": 2,
    "model_context_limit": 40960,
    "prefill_chunk_limit": 2048,
    "kv_layout": "contiguous_runtime_v1",
    "kv_dtype": "bfloat16",
    "kv_bytes_per_token": 114688,
    "active_kv_profile_limits": [128, 512, 2048, 8192, 32768, 40960],
    "runtime_owned": true
  },
  "sections": {}
})";
    write_minimal_bundle(path, json);

    const auto info = trtmc::InspectBundle(path);
    const auto& memory = info.runtime_memory;
    check(memory.present, "runtime_memory present");
    check(memory.contract_version == 1, "runtime_memory version");
    check(memory.qualified_model_id == "Qwen/Qwen3-0.6B", "runtime_memory qualified model");
    check(memory.qualified_model_revision == "c1899de289a04d12100db370d81485cdf75e47ca",
          "runtime_memory qualified revision");
    check(memory.qualified_config_sha256 ==
              "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
          "runtime_memory config fingerprint");
    check(memory.qualified_target == "gb300-trt-11.2", "runtime_memory target");
    check(memory.qualified_runtime_stack.sm == "sm103" &&
              memory.qualified_runtime_stack.tensorrt == "11.2.0.113" &&
              memory.qualified_runtime_stack.cuda_runtime == "13.3" &&
              memory.qualified_runtime_stack.cudnn_backend == "9.20.0" &&
              memory.qualified_runtime_stack.cudnn_frontend_revision ==
                  "7b9b711c22b6823e87150213ecd8449260db8610" &&
              memory.qualified_runtime_stack.nvrtc == "13.3" &&
              memory.qualified_runtime_stack.driver == "580.105.08",
          "runtime_memory qualified stack");
    check(memory.native_kv_plugin_abi == 2, "runtime_memory plugin ABI");
    check(memory.model_context_limit == 40960, "runtime_memory M");
    check(memory.prefill_chunk_limit == 2048, "runtime_memory C");
    check(memory.kv_layout == "contiguous_runtime_v1", "runtime_memory layout");
    check(memory.kv_dtype == "bfloat16", "runtime_memory dtype");
    check(memory.kv_bytes_per_token == 114688, "runtime_memory B");
    check(memory.active_kv_profile_limits ==
              std::vector<int32_t>({128, 512, 2048, 8192, 32768, 40960}),
          "runtime_memory buckets");
    check(memory.runtime_owned, "runtime_memory runtime owned");

    const auto legacy_path = (tmp / "legacy-runtime-memory.trtfb").string();
    write_minimal_bundle(legacy_path,
                         R"({"model_id":"legacy","max_cache_length":256,"sections":{}})");
    const auto legacy = trtmc::InspectBundle(legacy_path);
    check(!legacy.runtime_memory.present, "legacy runtime_memory omission remains static");
    check(legacy.max_cache_length == 256, "legacy max_cache_length remains available");

    trtmc_test::remove_all_safe(tmp);
}

static void test_invalid_runtime_memory_contract_is_rejected() {
    const auto tmp = make_temp_dir();
    const auto path = (tmp / "invalid-runtime-memory.trtfb").string();
    write_minimal_bundle(path, R"({
  "model_id": "Qwen/Qwen3-0.6B",
  "precision": "bf16",
  "max_cache_length": 40960,
  "runtime_memory": {
    "contract_version": 1,
    "qualified_model_id": "Qwen/Qwen3-0.6B",
    "qualified_model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
    "qualified_config_sha256": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "qualified_target": "gb300-trt-11.2",
    "qualified_runtime_stack": {
      "sm": "sm103",
      "tensorrt": "11.2.0.113",
      "cuda_runtime": "13.3",
      "cudnn_backend": "9.20.0",
      "cudnn_frontend_revision": "7b9b711c22b6823e87150213ecd8449260db8610",
      "nvrtc": "13.3",
      "driver": "580.105.08"
    },
    "native_kv_plugin_abi": 2,
    "model_context_limit": 40960,
    "prefill_chunk_limit": 2048,
    "kv_layout": "contiguous_runtime_v1",
    "kv_dtype": "bfloat16",
    "kv_bytes_per_token": 114688,
    "active_kv_profile_limits": [128, 512, 2048, 8192, 32768],
    "runtime_owned": true
  },
  "sections": {}
})");

    bool rejected = false;
    try {
        (void)trtmc::InspectBundle(path);
    } catch (const std::runtime_error& error) {
        rejected = std::string(error.what()).find("runtime_memory") != std::string::npos;
    }
    check(rejected, "runtime_memory whose final bucket does not cover M is rejected");

    trtmc_test::remove_all_safe(tmp);
}

static void test_tokenizer_add_special_tokens_header() {
    const auto tmp = make_temp_dir();
    const auto path = (tmp / "tokenizer_flag.trtfb").string();

    const std::string json = R"({
  "model_id": "tokenizer-flag",
  "tokenizer_add_special_tokens": 0,
  "sections": {}
})";
    write_minimal_bundle(path, json);

    const auto loaded = trtmc::ReadBundleFile(path);
    check(loaded.info.tokenizer_add_special_tokens_present, "tokenizer_add_special_tokens present");
    check(!loaded.info.tokenizer_add_special_tokens, "tokenizer_add_special_tokens false");

    trtmc_test::remove_all_safe(tmp);
}

static void test_truncated_bundle_throws() {
    const auto tmp = make_temp_dir();
    const auto path = (tmp / "truncated.trtfb").string();

    {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(trtmc::kBundleMagic), 8);
        uint64_t len = 1000;
        unsigned char bytes[8];
        for (int i = 0; i < 8; ++i)
            bytes[i] = static_cast<unsigned char>((len >> (8 * i)) & 0xFF);
        out.write(reinterpret_cast<const char*>(bytes), 8);
        out.write("short", 5);
    }

    bool threw = false;
    try {
        trtmc::ReadBundleFile(path);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    check(threw, "truncated bundle throws");

    trtmc_test::remove_all_safe(tmp);
}

static void test_max_batch_size_parse_and_back_compat() {
    // New bundles carry the per-component cap; legacy bundles omit the
    // block and the parser defaults all three components to 1.
    const auto tmp = make_temp_dir();
    std::vector<char> plan_data = {'P', 'L', 'A', 'N'};

    const auto new_path = (tmp / "new.trtfb").string();
    write_bundle_with_sections(new_path, R"({
  "model_id": "diffusion-bs4", "family": "diffusion",
  "max_batch_size": {"dit": 4, "text_encoder": 8, "vae": 1},
  "sections": {"engine_plan": {"offset": 0, "size": 4}}
})",
                               {plan_data});
    const auto loaded_new = trtmc::ReadBundleFile(new_path);
    check(loaded_new.info.max_batch_size.dit == 4 &&
              loaded_new.info.max_batch_size.text_encoder == 8 &&
              loaded_new.info.max_batch_size.vae == 1,
          "max_batch_size parsed from header");

    const auto legacy_path = (tmp / "legacy.trtfb").string();
    write_bundle_with_sections(legacy_path, R"({
  "model_id": "legacy", "family": "generic",
  "sections": {"engine_plan": {"offset": 0, "size": 4}}
})",
                               {plan_data});
    const auto loaded_legacy = trtmc::ReadBundleFile(legacy_path);
    check(loaded_legacy.info.max_batch_size.dit == 1 &&
              loaded_legacy.info.max_batch_size.text_encoder == 1 &&
              loaded_legacy.info.max_batch_size.vae == 1,
          "absent max_batch_size defaults to {1,1,1}");

    trtmc_test::remove_all_safe(tmp);
}

class TrackingStreamBuffer final : public std::streambuf {
  public:
    std::vector<char> data;
    std::streamsize largest_write{0};

  protected:
    std::streamsize xsputn(const char* source, std::streamsize count) override {
        largest_write = std::max(largest_write, count);
        data.insert(data.end(), source, source + count);
        return count;
    }
};

static void test_copy_section_streams_in_bounded_chunks() {
    const auto tmp = make_temp_dir();
    const auto path = (tmp / "streamed.trtfb").string();
    std::vector<char> payload(3 * 1024 * 1024 + 17, 'E');
    const std::string header =
        "{\"model_id\":\"streamed\",\"sections\":{\"edge/llm.engine\":{\"offset\":0,\"size\":" +
        std::to_string(payload.size()) + "}}}";
    write_bundle_with_sections(path, header, {payload});

    const auto info = trtmc::ReadBundleHeader(path);
    check(info.sections.size() == 1, "streamed section metadata parsed");
    TrackingStreamBuffer buffer;
    std::ostream output(&buffer);
    trtmc::CopyBundleSection(path, info.sections.front(), output);
    check(buffer.data == payload, "streamed section preserves all bytes");
    check(buffer.largest_write <= 1024 * 1024, "streamed section uses bounded chunks");

    trtmc_test::remove_all_safe(tmp);
}

static void test_sha256_known_vectors() {
    trtmc::internal::Sha256 empty;
    check(empty.hex_digest() == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
          "SHA-256 empty vector");
    trtmc::internal::Sha256 abc;
    abc.update("abc");
    check(abc.hex_digest() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "SHA-256 abc vector");
    trtmc::internal::Sha256 multi_block_padding;
    multi_block_padding.update("abcdbcdecdefdefgefghfghighijhijk");
    multi_block_padding.update("ijkljklmklmnlmnomnopnopq");
    check(multi_block_padding.hex_digest() ==
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
          "SHA-256 incremental multi-block padding vector");
}

int main() {
    test_read_valid_bundle();
    test_magic_validation();
    test_empty_sections();
    test_is_bundle_valid();
    test_is_bundle_invalid();
    test_inspect_returns_metadata();
    test_inspect_is_header_only();
    test_runtime_memory_contract_parse_and_legacy_omission();
    test_invalid_runtime_memory_contract_is_rejected();
    test_tokenizer_add_special_tokens_header();
    test_truncated_bundle_throws();
    test_max_batch_size_parse_and_back_compat();
    test_copy_section_streams_in_bounded_chunks();
    test_sha256_known_vectors();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All bundle_format tests passed.\n";
    return 0;
}
