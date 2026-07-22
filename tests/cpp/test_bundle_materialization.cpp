/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "bundle/bundle_format.h"
#include "runtime/registry/bundle_materialization.h"
#include "test_helpers.h"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
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

std::filesystem::path temporary_directory() {
    char pattern[] = "/tmp/trtfb_materialization_XXXXXX";
    char* directory = mkdtemp(pattern);
    if (directory == nullptr)
        throw std::runtime_error(std::string("mkdtemp failed: ") + std::strerror(errno));
    return directory;
}

void write_bundle(const std::filesystem::path& path, const std::string& header,
                  const std::vector<char>& payload) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(trtmc::kBundleMagic), 8);
    const std::uint64_t length = header.size();
    for (int shift = 0; shift < 64; shift += 8)
        output.put(static_cast<char>((length >> shift) & 0xffU));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write(payload.data(), static_cast<std::streamsize>(payload.size()));
}

std::string two_section_header(std::size_t config_size, std::uint64_t plan_size) {
    std::ostringstream header;
    header << R"({"model_id":"test","sections":{)"
           << R"("config.json":{"offset":0,"size":)" << config_size << "},"
           << R"("denoiser_plan":{"offset":)" << config_size << R"(,"size":)" << plan_size << "}}}";
    return header.str();
}

const trtmc::BundleSection* find_section(const trtmc::BundleFile& bundle, const std::string& name) {
    for (const auto& section : bundle.sections) {
        if (section.name == name)
            return &section;
    }
    return nullptr;
}

void test_staged_policy_leaves_plan_bytes_unread() {
    const auto directory = temporary_directory();
    const auto path = directory / "staged.trtfb";
    const std::string config =
        R"({"bundle_loading":{"mode":"staged","eager_sections":["config.json"],"lazy_sections":["denoiser_plan"]}})";
    const std::uint64_t absent_plan_size = 1ULL << 40;
    write_bundle(path, two_section_header(config.size(), absent_plan_size),
                 std::vector<char>(config.begin(), config.end()));

    const auto materialized = trtmc::detail::materialize_pipeline_bundle(path.string());
    check(materialized.bundle.sections.size() == 1,
          "staged policy materializes only eager sections");
    check(find_section(materialized.bundle, "config.json") != nullptr,
          "staged policy materializes config");
    check(find_section(materialized.bundle, "denoiser_plan") == nullptr,
          "staged policy leaves plans lazy");
    check(materialized.bundle.info.sections.size() == 2,
          "staged policy retains lazy section metadata");

    trtmc_test::remove_all_safe(directory);
}

void test_invalid_partition_fails_before_plan_read() {
    const auto directory = temporary_directory();
    const auto path = directory / "invalid.trtfb";
    const std::string config =
        R"({"bundle_loading":{"mode":"staged","eager_sections":["config.json"],"lazy_sections":["config.json"]}})";
    write_bundle(path, two_section_header(config.size(), 1ULL << 40),
                 std::vector<char>(config.begin(), config.end()));

    std::string error;
    try {
        (void)trtmc::detail::materialize_pipeline_bundle(path.string());
    } catch (const std::runtime_error& exception) {
        error = exception.what();
    }
    check(error == "Staged bundle_loading must partition bundle sections exactly",
          "invalid staged partition is rejected");
    check(error.find("extends outside file") == std::string::npos,
          "invalid policy is rejected before plan bytes are read");

    trtmc_test::remove_all_safe(directory);
}

void test_bundle_without_policy_preserves_read_all_behavior() {
    const auto directory = temporary_directory();
    const auto path = directory / "legacy.trtfb";
    const std::string config = R"({"runtime_strategy":"legacy_decoder"})";
    const std::vector<char> plan = {'P', 'L', 'A', 'N'};
    std::vector<char> payload(config.begin(), config.end());
    payload.insert(payload.end(), plan.begin(), plan.end());
    write_bundle(path, two_section_header(config.size(), plan.size()), payload);

    const auto materialized = trtmc::detail::materialize_pipeline_bundle(path.string());
    const auto* loaded_plan = find_section(materialized.bundle, "denoiser_plan");
    check(materialized.bundle.sections.size() == 2,
          "bundle without staged policy retains read-all behavior");
    check(loaded_plan != nullptr && loaded_plan->data == plan,
          "bundle without staged policy reads plan bytes");

    trtmc_test::remove_all_safe(directory);
}

} // namespace

int main() {
    test_staged_policy_leaves_plan_bytes_unread();
    test_invalid_partition_fails_before_plan_read();
    test_bundle_without_policy_preserves_read_all_behavior();
    if (failures != 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All bundle materialization tests passed\n";
    return 0;
}
