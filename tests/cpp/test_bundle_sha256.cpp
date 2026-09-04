/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_format.h"
#include "runtime/backend/file_plan_validation.h"
#include "utils/sha256.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

std::filesystem::path make_temp_dir() {
    const auto nonce = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    const auto root = std::filesystem::temp_directory_path();
    for (int attempt = 0; attempt < 100; ++attempt) {
        const auto candidate =
            root / ("trtmc_bundle_sha256_" + std::to_string(nonce) + "_" + std::to_string(attempt));
        std::error_code error;
        if (std::filesystem::create_directory(candidate, error))
            return candidate;
    }
    throw std::runtime_error("Unable to create bundle SHA-256 test directory");
}

void write_bundle(const std::filesystem::path& path, const std::string& header,
                  const std::vector<char>& payload) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("Unable to create bundle SHA-256 fixture");
    output.write(reinterpret_cast<const char*>(trtmc::kBundleMagic), 8);
    const auto header_size = static_cast<std::uint64_t>(header.size());
    for (int index = 0; index < 8; ++index)
        output.put(static_cast<char>((header_size >> (8 * index)) & 0xffU));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    if (!output)
        throw std::runtime_error("Unable to write bundle SHA-256 fixture");
}

void test_streaming_attestation_rejects_tamper_and_truncation() {
    const auto directory = make_temp_dir();
    const auto path = directory / "attested.bundle";
    std::vector<char> payload(3 * 1024 * 1024 + 17);
    for (std::size_t index = 0; index < payload.size(); ++index)
        payload[index] = static_cast<char>('A' + (index % 23));
    trtmc::internal::Sha256 digest;
    digest.update(payload.data(), payload.size());
    const std::string expected = digest.hex_digest();
    const std::string header =
        "{\"model_id\":\"attested\",\"sections\":{\"denoiser_transition_17_plan\":"
        "{\"offset\":0,\"size\":" +
        std::to_string(payload.size()) + "}}}";
    write_bundle(path, header, payload);

    const auto info = trtmc::ReadBundleHeader(path.string());
    check(info.sections.size() == 1, "attested section metadata parsed");
    const auto section = info.sections.front();
    trtmc::ValidateBundleSectionSha256(path.string(), section, expected);

    const auto range = trtmc::ResolveBundleSectionFileRange(path.string(), section);
    {
        std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
        file.seekp(static_cast<std::streamoff>(range.offset + payload.size() / 2));
        file.put('!');
    }
    bool rejected_tamper = false;
    try {
        trtmc::ValidateBundleSectionSha256(path.string(), section, expected);
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        rejected_tamper = message.find(section.name) != std::string::npos &&
                          message.find("SHA-256") != std::string::npos;
    }
    check(rejected_tamper, "content tamper reports the failing section");

    write_bundle(path, header, payload);
    std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
    bool rejected_truncation = false;
    try {
        trtmc::ValidateBundleSectionSha256(path.string(), section, expected);
    } catch (const std::runtime_error& error) {
        rejected_truncation = std::string(error.what()).find(section.name) != std::string::npos;
    }
    check(rejected_truncation, "truncated payload reports the failing section");

    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::filesystem::remove(directory, ignored);
}

void test_bounds_only_validation_does_not_scan_payload() {
    const auto directory = make_temp_dir();
    const auto path = directory / "bounds-only.bundle";
    const std::vector<char> payload = {'P', 'L', 'A', 'N'};
    const std::string header =
        R"({"model_id":"bounds-only","sections":{"denoiser_plan":{"offset":0,"size":4}}})";
    write_bundle(path, header, payload);

    const auto info = trtmc::ReadBundleHeader(path.string());
    check(info.sections.size() == 1, "bounds-only section metadata parsed");
    trtmc::ValidateBundleSectionBounds(path.string(), info.sections);

    // Same-size payload corruption is intentionally not detected by the fast
    // bounds pass. Full validation remains available and must detect it.
    {
        std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
        file.seekp(-1, std::ios::end);
        file.put('X');
    }
    trtmc::ValidateBundleSectionBounds(path.string(), info.sections);

    bool digest_threw = false;
    trtmc::internal::Sha256 original_digest;
    original_digest.update(payload.data(), payload.size());
    try {
        trtmc::ValidateBundleSectionSha256(path.string(), info.sections.front(),
                                           original_digest.hex_digest());
    } catch (const std::runtime_error&) {
        digest_threw = true;
    }
    check(digest_threw, "full validation still rejects same-size payload corruption");

    std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
    bool bounds_threw = false;
    try {
        trtmc::ValidateBundleSectionBounds(path.string(), info.sections);
    } catch (const std::runtime_error&) {
        bounds_threw = true;
    }
    check(bounds_threw, "bounds-only validation rejects truncated payload");
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::filesystem::remove(directory, ignored);
}

void test_backend_skips_plan_preverification_when_disabled() {
    struct CountingReader {
        int calls{0};
        void verify_sha256() { ++calls; }
    };

    CountingReader fast_reader;
    trtmc::ModuleCreateOptions fast_options;
    fast_options.verify_plan_sha256 = false;
    trtmc::internal::verify_plan_sha256_if_requested(fast_reader, fast_options);
    check(fast_reader.calls == 0, "fast backend policy skips plan-content SHA-256 attestation");

    CountingReader strict_reader;
    trtmc::ModuleCreateOptions strict_options;
    strict_options.verify_plan_sha256 = true;
    trtmc::internal::verify_plan_sha256_if_requested(strict_reader, strict_options);
    check(strict_reader.calls == 1, "strict backend policy performs plan SHA-256 attestation");
}

void test_plan_cache_identity_scopes_retained_engines_to_file_and_section() {
    const std::string digest_a(64, 'a');
    const std::string digest_b(64, 'b');
    const auto identity =
        trtmc::internal::make_plan_cache_identity("bundle-file-a", 4096, 8192, digest_a);

    check(identity ==
              trtmc::internal::make_plan_cache_identity("bundle-file-a", 4096, 8192, digest_a),
          "same file section keeps a stable retained-engine cache identity");
    check(identity !=
              trtmc::internal::make_plan_cache_identity("bundle-file-b", 4096, 8192, digest_a),
          "different bundle file cannot reuse a retained engine");
    check(identity !=
              trtmc::internal::make_plan_cache_identity("bundle-file-a", 4097, 8192, digest_a),
          "different plan offset cannot reuse a retained engine");
    check(identity !=
              trtmc::internal::make_plan_cache_identity("bundle-file-a", 4096, 8193, digest_a),
          "different plan size cannot reuse a retained engine");
    check(identity !=
              trtmc::internal::make_plan_cache_identity("bundle-file-a", 4096, 8192, digest_b),
          "different declared digest cannot reuse a retained engine");
}

} // namespace

int main() {
    test_streaming_attestation_rejects_tamper_and_truncation();
    test_bounds_only_validation_does_not_scan_payload();
    test_backend_skips_plan_preverification_when_disabled();
    test_plan_cache_identity_scopes_retained_engines_to_file_and_section();
    if (failures != 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All bundle SHA-256 tests passed.\n";
    return 0;
}
