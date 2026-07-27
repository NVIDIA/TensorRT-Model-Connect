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
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <streambuf>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

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

#if defined(__linux__)
static std::filesystem::path make_cache_test_temp_dir() {
    const auto pattern_path = std::filesystem::current_path() / "trtfb_cache_test_XXXXXX";
    const std::string pattern = pattern_path.string();
    std::vector<char> mutable_pattern(pattern.begin(), pattern.end());
    mutable_pattern.push_back('\0');
    char* dir = mkdtemp(mutable_pattern.data());
    if (dir == nullptr) {
        throw std::runtime_error(std::string("mkdtemp failed: ") + std::strerror(errno));
    }
    return std::filesystem::path(dir);
}

static double resident_fraction(const std::string& path, std::uint64_t offset, std::uint64_t size) {
    const int descriptor = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor < 0)
        throw std::runtime_error("open failed while checking bundle residency");

    struct stat metadata{};
    if (fstat(descriptor, &metadata) != 0 || metadata.st_size <= 0) {
        close(descriptor);
        throw std::runtime_error("fstat failed while checking bundle residency");
    }

    const auto file_size = static_cast<std::size_t>(metadata.st_size);
    void* mapping = mmap(nullptr, file_size, PROT_NONE, MAP_SHARED, descriptor, 0);
    if (mapping == MAP_FAILED) {
        close(descriptor);
        throw std::runtime_error("mmap failed while checking bundle residency");
    }

    const auto page_size = static_cast<std::uint64_t>(sysconf(_SC_PAGESIZE));
    std::vector<unsigned char> residency((file_size + page_size - 1) / page_size);
    if (mincore(mapping, file_size, residency.data()) != 0) {
        munmap(mapping, file_size);
        close(descriptor);
        throw std::runtime_error("mincore failed while checking bundle residency");
    }

    munmap(mapping, file_size);
    close(descriptor);

    const std::uint64_t first_page = (offset + page_size - 1) / page_size;
    const std::uint64_t end_page = (offset + size) / page_size;
    if (first_page >= end_page)
        throw std::runtime_error("bundle residency range has no complete pages");

    std::size_t resident_pages = 0;
    for (std::uint64_t page = first_page; page < end_page; ++page)
        resident_pages += residency[page] & 1U;
    return static_cast<double>(resident_pages) / static_cast<double>(end_page - first_page);
}

static bool cache_residency_drops_below(const std::string& path, std::uint64_t offset,
                                        std::uint64_t size, double threshold) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    do {
        if (resident_fraction(path, offset, size) < threshold)
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}

static void test_read_bundle_file_drops_payload_cache() {
    // Use the CTest build volume instead of the container overlay filesystem:
    // overlayfs may accept POSIX_FADV_DONTNEED without exposing eviction through
    // mincore, while production bundles live on a regular mounted filesystem.
    const auto tmp = make_cache_test_temp_dir();
    const auto path = (tmp / "cache-drop.trtfb").string();
    std::vector<char> payload(8 * 1024 * 1024, 'C');
    const std::string header =
        "{\"model_id\":\"cache-drop\",\"sections\":{\"engine_plan\":{\"offset\":0,\"size\":" +
        std::to_string(payload.size()) + "}}}";
    write_bundle_with_sections(path, header, {payload});

    // DONTNEED discards clean pages. Make the freshly written fixture match a
    // stable model bundle before checking the runtime's cache release.
    const int descriptor = open(path.c_str(), O_RDWR | O_CLOEXEC);
    check(descriptor >= 0 && fsync(descriptor) == 0, "cache-drop fixture synced");
    if (descriptor >= 0)
        close(descriptor);

    const std::uint64_t payload_offset = trtmc::kBundleHeaderOffset + header.size();
    check(resident_fraction(path, payload_offset, payload.size()) > 0.9,
          "cache-drop fixture starts resident");

    const auto loaded = trtmc::ReadBundleFile(path);
    check(loaded.sections.size() == 1 && loaded.sections.front().data == payload,
          "cache-drop read preserves section data");
    check(cache_residency_drops_below(path, payload_offset, payload.size(), 0.1),
          "bundle payload cache dropped after read");

    trtmc_test::remove_all_safe(tmp);
}
#endif

static void test_sha256_known_vectors() {
    trtmc::internal::Sha256 empty;
    check(empty.hex_digest() == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
          "SHA-256 empty vector");
    trtmc::internal::Sha256 abc;
    abc.update("abc");
    abc.update(nullptr, 0);
    check(abc.hex_digest() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
          "SHA-256 abc vector and zero-size null update");
    trtmc::internal::Sha256 multi_block_padding;
    multi_block_padding.update("abcdbcdecdefdefgefghfghighijhijk");
    multi_block_padding.update("ijkljklmklmnlmnomnopnopq");
    check(multi_block_padding.hex_digest() ==
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
          "SHA-256 incremental multi-block padding vector");
}

static void test_sha256_boundaries_and_chunking() {
    struct ExpectedDigest {
        std::size_t size;
        const char* hex;
    };
    static constexpr ExpectedDigest expected[] = {
        {0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {1, "0bfe935e70c321c7ca3afc75ce0d0ca2f98b5422e008bb31c00c6d7f1f1c0ad6"},
        {55, "13eb4480c03a102020e430341f50315f7ab0e2eab3a84c8c630e019c8baa68a0"},
        {56, "95879a336f9ac08995bd30220a50c60710b5e5712314c357dae0375596e7cdc7"},
        {63, "0fd92b28d7de02f75f5c140ffe00bc834a2911c50841b1d734868d30d17013be"},
        {64, "8509136a1d80b20158213ca1508c046646abdf827af2fc1f4023e557cb59ef61"},
        {65, "28ae167bf703e3582d77be270a09fd3b8c250a3125adc3b3b30ee4a883685ab1"},
        {127, "aeb49743ff155ac07a69cfad222e391d8ba711cd8336b4c7b38058cb9bcb5604"},
        {128, "2fe61ce346d4861e8f5ddfdc9da12797a78660ddadbf2ab83e630c9f956a5742"},
        {129, "f71a9a7856fa2f3b7176c0fce89e44eeed1f69076ffffbe864f51b7180458b33"},
        {1024, "e219ea8e440569c6815638fde3d38d2c9028a6de36c21bb96b313cd0dc4e9474"},
        {65537, "76bc99ec74df99033f031e2b96c6b07b9bafbb06769a670b94a6b7e5eb6d551c"},
    };

    std::vector<std::uint8_t> input(expected[std::size(expected) - 1].size);
    std::uint32_t random_state = 0x12345678U;
    for (auto& byte : input) {
        random_state = random_state * 1664525U + 1013904223U;
        byte = static_cast<std::uint8_t>(random_state >> 24U);
    }

    static constexpr std::size_t chunk_sizes[] = {1, 3, 7, 31, 64, 65, 127, 1024};
    for (const auto& vector : expected) {
        trtmc::internal::Sha256 one_shot;
        one_shot.update(input.data(), vector.size);
        check(one_shot.hex_digest() == vector.hex, "SHA-256 deterministic boundary vector");

        for (const std::size_t chunk_size : chunk_sizes) {
            trtmc::internal::Sha256 chunked;
            for (std::size_t offset = 0; offset < vector.size; offset += chunk_size) {
                const std::size_t count = std::min(chunk_size, vector.size - offset);
                chunked.update(input.data() + offset, count);
            }
            check(chunked.hex_digest() == vector.hex,
                  "SHA-256 fixed-size incremental chunk parity");
        }

        trtmc::internal::Sha256 random_chunks;
        std::size_t offset = 0;
        std::uint32_t chunk_state = 0x9e3779b9U;
        while (offset < vector.size) {
            chunk_state = chunk_state * 1103515245U + 12345U;
            const std::size_t requested = 1 + ((chunk_state >> 16U) & 511U);
            const std::size_t count = std::min(requested, vector.size - offset);
            random_chunks.update(input.data() + offset, count);
            offset += count;
        }
        check(random_chunks.hex_digest() == vector.hex,
              "SHA-256 pseudo-random incremental chunk parity");
    }
}

int main() {
    test_read_valid_bundle();
    test_magic_validation();
    test_empty_sections();
    test_is_bundle_valid();
    test_is_bundle_invalid();
    test_inspect_returns_metadata();
    test_tokenizer_add_special_tokens_header();
    test_truncated_bundle_throws();
    test_max_batch_size_parse_and_back_compat();
    test_copy_section_streams_in_bounded_chunks();
#if defined(__linux__)
    test_read_bundle_file_drops_payload_cache();
#endif
    test_sha256_known_vectors();
    test_sha256_boundaries_and_chunking();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All bundle_format tests passed.\n";
    return 0;
}
