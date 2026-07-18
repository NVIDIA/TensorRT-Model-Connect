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
  "source_model_id": "example-org/test-model",
  "source_revision": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
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
    check(loaded.info.source_model_id == "example-org/test-model", "read source_model_id");
    check(loaded.info.source_revision == "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
          "read source_revision");
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

static void test_read_single_bundle_section() {
    const auto tmp = make_temp_dir();
    const auto path = (tmp / "single.trtfb").string();
    const std::string json = R"({
  "sections": {
    "first": {"offset": 0, "size": 4},
    "target": {"offset": 4, "size": 3},
    "last": {"offset": 7, "size": 2}
  }
})";
    write_bundle_with_sections(path, json, {{'s', 'k', 'i', 'p'}, {'T', 'R', 'T'}, {'x', 'x'}});

    check(trtmc::ReadBundleSection(path, "target") == std::vector<char>({'T', 'R', 'T'}),
          "single-section read seeks to the requested payload");

    bool missing_threw = false;
    try {
        (void)trtmc::ReadBundleSection(path, "missing");
    } catch (const std::runtime_error& error) {
        missing_threw = std::string(error.what()).find("was not found") != std::string::npos;
    }
    check(missing_threw, "single-section read rejects a missing name");
    trtmc_test::remove_all_safe(tmp);
}

static void test_read_single_bundle_section_validates_bounds() {
    const auto tmp = make_temp_dir();
    const auto path = (tmp / "out_of_bounds.trtfb").string();
    const std::string json = R"({
  "sections": {
    "target": {"offset": 2, "size": 8}
  }
})";
    write_bundle_with_sections(path, json, {{'x', 'y', 'z'}});

    bool bounds_threw = false;
    try {
        (void)trtmc::ReadBundleSection(path, "target");
    } catch (const std::runtime_error& error) {
        bounds_threw = std::string(error.what()).find("outside file bounds") != std::string::npos;
    }
    check(bounds_threw, "single-section read rejects a truncated payload");
    trtmc_test::remove_all_safe(tmp);
}

static bool section_metadata_is_rejected(const std::filesystem::path& path,
                                         const std::string& fields) {
    const std::string json = "{\"sections\":{\"target\":{" + fields + "}}}";
    write_bundle_with_sections(path.string(), json, {{'x', 'y', 'z'}});
    try {
        (void)trtmc::ReadBundleSection(path.string(), "target");
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

static void test_read_single_bundle_section_rejects_invalid_numbers() {
    const auto tmp = make_temp_dir();
    const auto path = tmp / "invalid_numbers.trtfb";

    check(section_metadata_is_rejected(path, "\"size\":1"),
          "single-section read rejects missing offset");
    check(section_metadata_is_rejected(path, "\"offset\":0"),
          "single-section read rejects missing size");
    check(section_metadata_is_rejected(path, "\"offset\":-1,\"size\":1"),
          "single-section read rejects negative offset");
    check(section_metadata_is_rejected(path, "\"offset\":0,\"size\":-1"),
          "single-section read rejects negative size");
    check(section_metadata_is_rejected(path, "\"offset\":\"bad\",\"size\":1"),
          "single-section read rejects malformed offset");
    check(section_metadata_is_rejected(path, "\"offset\":0,\"size\":1.5"),
          "single-section read rejects malformed size");
    check(section_metadata_is_rejected(path, "\"offset\":18446744073709551616,\"size\":1"),
          "single-section read rejects overflowing offset");
    check(section_metadata_is_rejected(path, "\"offset\":0,\"size\":18446744073709551616"),
          "single-section read rejects overflowing size");

    trtmc_test::remove_all_safe(tmp);
}

static void test_pinned_section_reader_survives_path_replacement() {
    const auto tmp = make_temp_dir();
    const auto path = tmp / "staged.trtfb";
    const auto pinned_path = tmp / "staged.original.trtfb";
    const std::string json = R"({
  "sections": {
    "eager": {"offset": 0, "size": 3},
    "lazy": {"offset": 3, "size": 4}
  }
})";
    write_bundle_with_sections(path.string(), json, {{'O', 'L', 'D'}, {'L', 'A', 'Z', 'Y'}});
    trtmc::BundleSectionReader reader(path.string());
    const auto materialized = trtmc::ReadBundleFile(reader);
    check(materialized.sections.size() == 2, "shared reader materializes every section");
    if (materialized.sections.size() == 2) {
        check(materialized.sections[0].data == std::vector<char>({'O', 'L', 'D'}),
              "shared reader materializes eager bytes from original bundle");
        check(materialized.sections[1].data == std::vector<char>({'L', 'A', 'Z', 'Y'}),
              "shared reader materializes lazy bytes from original bundle");
    }

    std::filesystem::rename(path, pinned_path);
    write_bundle_with_sections(path.string(), json, {{'N', 'E', 'W'}, {'F', 'I', 'L', 'E'}});
    check(reader.read("lazy") == std::vector<char>({'L', 'A', 'Z', 'Y'}),
          "pinned reader is not redirected by path replacement");
    check(trtmc::ReadBundleSection(path.string(), "lazy") ==
              std::vector<char>({'F', 'I', 'L', 'E'}),
          "new readers observe the replacement bundle");

    std::filesystem::remove(pinned_path);
    check(reader.read("lazy") == std::vector<char>({'L', 'A', 'Z', 'Y'}),
          "pinned reader survives unlink of its original file");
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
  "source_model_id": "example-org/inspectable",
  "source_revision": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "vocab_size": 50000,
  "num_layers": 12,
  "sections": {
    "data": {"offset": 0, "size": 3}
  }
})";
    write_bundle_with_sections(path, json, {{'X', 'Y', 'Z'}});

    const auto info = trtmc::InspectBundle(path);
    check(info.model_id == "inspectable", "inspect model_id");
    check(info.source_model_id == "example-org/inspectable", "inspect source_model_id");
    check(info.source_revision == "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
          "inspect source_revision");
    check(info.vocab_size == 50000, "inspect vocab_size");
    check(info.num_layers == 12, "inspect num_layers");

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
    check(loaded_legacy.info.source_model_id.empty() && loaded_legacy.info.source_revision.empty(),
          "legacy source provenance defaults empty");
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
    test_read_single_bundle_section();
    test_read_single_bundle_section_validates_bounds();
    test_read_single_bundle_section_rejects_invalid_numbers();
    test_pinned_section_reader_survives_path_replacement();
    test_magic_validation();
    test_empty_sections();
    test_is_bundle_valid();
    test_is_bundle_invalid();
    test_inspect_returns_metadata();
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
