/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Regression coverage for the generic pipeline bundle materialization policy.
// Staged bundles must not read large lazy sections during pipeline construction;
// bundles without the opt-in policy must retain read-all behavior.

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

std::filesystem::path make_temp_dir() {
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
    unsigned char bytes[8];
    for (int index = 0; index < 8; ++index)
        bytes[index] = static_cast<unsigned char>((length >> (8 * index)) & 0xff);
    output.write(reinterpret_cast<const char*>(bytes), 8);
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write(payload.data(), static_cast<std::streamsize>(payload.size()));
}

std::string staged_config(const std::string& mode, const std::string& eager,
                          const std::string& lazy) {
    return "{\"bundle_loading\":{\"mode\":\"" + mode + "\",\"eager_sections\":" + eager +
           ",\"lazy_sections\":" + lazy + "}}";
}

std::string staged_materialization_error(const std::filesystem::path& path,
                                         const std::string& config,
                                         const std::vector<std::string>& lazy_header_names) {
    const std::uint64_t huge_section_size = 1ULL << 40;
    std::ostringstream header;
    header << R"({"model_id":"policy-test","sections":{)"
           << R"("config.json":{"offset":0,"size":)" << config.size() << "}";
    for (const auto& name : lazy_header_names) {
        header << ",\"" << name << R"(":{"offset":)" << config.size() << R"(,"size":)"
               << huge_section_size << "}";
    }
    header << "}}";
    write_bundle(path, header.str(), std::vector<char>(config.begin(), config.end()));

    try {
        trtmc::BundleSectionReader reader(path.string());
        (void)trtmc::detail::materialize_pipeline_bundle(reader);
    } catch (const std::runtime_error& error) {
        return error.what();
    }
    return {};
}

const trtmc::BundleSection* find_materialized(const trtmc::BundleFile& bundle,
                                              const std::string& name) {
    for (const auto& section : bundle.sections) {
        if (section.name == name)
            return &section;
    }
    return nullptr;
}

void test_staged_policy_does_not_eagerly_read_large_sections() {
    const auto temporary = make_temp_dir();
    const auto path = temporary / "staged.trtfb";
    const std::string config = R"({
      "runtime_strategy": "test_staged_runtime",
      "bundle_loading": {
        "mode": "staged",
        "eager_sections": [
          "tokenizer.json",
          "config.json"
        ],
        "lazy_sections": [
          "lazy_plugin.so",
          "lazy_plan_0",
          "lazy_plan_1",
          "lazy_plan_2",
          "lazy_plan_3"
        ]
      }
    })";

    // The physical file ends after the two eager payloads. Each plan claims
    // a 1 TiB range outside the file. A read-all implementation must fail;
    // successful materialization therefore proves no plan payload was read.
    const std::uint64_t config_offset = 2;
    const std::uint64_t lazy_offset = config_offset + config.size();
    const std::uint64_t huge_plan_size = 1ULL << 40;
    std::ostringstream header;
    header << R"({"model_id":"staged-test","sections":{)"
           << R"("tokenizer.json":{"offset":0,"size":2},)"
           << R"("config.json":{"offset":)" << config_offset << R"(,"size":)" << config.size()
           << "},"
           << R"("lazy_plan_0":{"offset":)" << lazy_offset << R"(,"size":)" << huge_plan_size
           << "},"
           << R"("lazy_plan_1":{"offset":)" << lazy_offset << R"(,"size":)" << huge_plan_size
           << "},"
           << R"("lazy_plan_2":{"offset":)" << lazy_offset << R"(,"size":)" << huge_plan_size
           << "},"
           << R"("lazy_plan_3":{"offset":)" << lazy_offset << R"(,"size":)" << huge_plan_size
           << "},"
           << R"("lazy_plugin.so":{"offset":)" << lazy_offset << R"(,"size":)" << huge_plan_size
           << "}}}";
    std::vector<char> payload = {'{', '}'};
    payload.insert(payload.end(), config.begin(), config.end());
    write_bundle(path, header.str(), payload);

    trtmc::BundleSectionReader reader(path.string());
    const auto materialized = trtmc::detail::materialize_pipeline_bundle(reader);
    check(materialized.bundle.sections.size() == 2,
          "staged policy materializes exactly two eager sections");
    check(find_materialized(materialized.bundle, "config.json") != nullptr,
          "staged policy materializes config");
    check(find_materialized(materialized.bundle, "tokenizer.json") != nullptr,
          "staged policy materializes tokenizer");
    check(find_materialized(materialized.bundle, "lazy_plan_1") == nullptr,
          "staged policy leaves plan sections lazy");
    check(find_materialized(materialized.bundle, "lazy_plugin.so") == nullptr,
          "staged policy leaves plugin sections lazy");
    check(materialized.bundle.info.sections.size() == 7,
          "staged view retains metadata for all seven sections");

    bool lazy_read_failed = false;
    try {
        (void)reader.read("lazy_plan_1");
    } catch (const std::runtime_error& error) {
        lazy_read_failed =
            std::string(error.what()).find("outside file bounds") != std::string::npos;
    }
    check(lazy_read_failed, "deferred plan range is validated only when lazily requested");
    trtmc_test::remove_all_safe(temporary);
}

void test_invalid_staged_policies_fail_before_lazy_payload_reads() {
    const auto temporary = make_temp_dir();
    const auto path = temporary / "invalid-policy.trtfb";
    const std::vector<std::string> normal_header{"lazy_plan"};
    auto expect_error = [&](const std::string& config, const std::vector<std::string>& header_names,
                            const std::string& expected, const char* name) {
        const auto error = staged_materialization_error(path, config, header_names);
        check(error.find(expected) != std::string::npos, name);
        check(error.find("outside file bounds") == std::string::npos,
              "invalid staged policy is rejected without reading lazy payload bytes");
    };

    expect_error(staged_config("eager", R"(["config.json"])", R"(["lazy_plan"])"), normal_header,
                 "Unsupported bundle_loading mode: eager", "unsupported staged mode is rejected");
    expect_error(staged_config("staged", "[]", R"(["lazy_plan"])"), normal_header,
                 "requires non-empty eager_sections and lazy_sections",
                 "empty eager section list is rejected");
    expect_error(staged_config("staged", R"(["config.json"])", "[]"), normal_header,
                 "requires non-empty eager_sections and lazy_sections",
                 "empty lazy section list is rejected");
    expect_error(staged_config("staged", R"(["config.json","config.json"])", R"(["lazy_plan"])"),
                 normal_header, "Duplicate or empty staged bundle section: config.json",
                 "duplicate eager section is rejected");
    expect_error(staged_config("staged", R"(["config.json"])", R"(["config.json","lazy_plan"])"),
                 normal_header, "Duplicate or empty staged bundle section: config.json",
                 "section duplicated across eager and lazy lists is rejected");
    expect_error(staged_config("staged", R"(["other"])", R"(["config.json","lazy_plan"])"),
                 normal_header, "must eagerly materialize config.json",
                 "config must be in the eager section list");
    expect_error(staged_config("staged", R"(["config.json"])", R"(["lazy_plan"])"),
                 {"lazy_plan", "lazy_plan"}, "Duplicate section in staged bundle header",
                 "duplicate staged header section is rejected");
    expect_error(staged_config("staged", R"(["config.json"])", R"(["lazy_plan"])"),
                 {"lazy_plan", "extra"}, "must partition the bundle header exactly",
                 "staged lists must exactly partition the header");

    trtmc_test::remove_all_safe(temporary);
}

void test_policy_absence_preserves_existing_read_all_behavior() {
    const auto temporary = make_temp_dir();
    const auto path = temporary / "legacy.trtfb";
    const std::string config = R"({"runtime_strategy":"legacy_decoder"})";
    const std::vector<char> engine = {'P', 'L', 'A', 'N'};
    std::ostringstream header;
    header << R"({"model_id":"legacy","sections":{)"
           << R"("config.json":{"offset":0,"size":)" << config.size() << "},"
           << R"("engine_plan":{"offset":)" << config.size() << R"(,"size":)" << engine.size()
           << "}}}";
    std::vector<char> payload(config.begin(), config.end());
    payload.insert(payload.end(), engine.begin(), engine.end());
    write_bundle(path, header.str(), payload);

    trtmc::BundleSectionReader reader(path.string());
    const auto materialized = trtmc::detail::materialize_pipeline_bundle(reader);
    const auto* engine_section = find_materialized(materialized.bundle, "engine_plan");
    check(materialized.bundle.sections.size() == 2,
          "bundle without staged policy retains read-all section count");
    check(engine_section != nullptr && engine_section->data == engine,
          "bundle without staged policy eagerly reads existing plugin plan");
    trtmc_test::remove_all_safe(temporary);
}

} // namespace

int main() {
    test_staged_policy_does_not_eagerly_read_large_sections();
    test_invalid_staged_policies_fail_before_lazy_payload_reads();
    test_policy_absence_preserves_existing_read_all_behavior();
    if (failures != 0) {
        std::cerr << failures << " test(s) FAILED\n";
        return 1;
    }
    std::cerr << "All bundle materialization tests passed\n";
    return 0;
}
