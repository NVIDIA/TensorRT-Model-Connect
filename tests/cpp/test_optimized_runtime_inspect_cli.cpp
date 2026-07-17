/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Executable regression test for capsule-generic `trtmc inspect`. The bundle
// has no native engine section; its payload and runtime DSO are opaque
// artifacts owned by one optimized-runtime implementation.

#include "bundle/bundle_format.h"
#include "utils/sha256.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#ifndef TRTMC_TEST_OPTIMIZED_PROVIDER_DSO
#error "TRTMC_TEST_OPTIMIZED_PROVIDER_DSO must be defined"
#endif

namespace {

namespace fs = std::filesystem;

constexpr const char* kImplementationId = "example-optimized-runtime";
constexpr const char* kRuntimeLibrary = "libtrtmc_impl_example_optimized_runtime.so";
constexpr const char* kModelId = "Example/Optimized-Model";

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) {
        std::cout << "  PASS: " << message << '\n';
    } else {
        std::cerr << "  FAIL: " << message << '\n';
        ++failures;
    }
}

void write_u64_le(std::ostream& output, std::uint64_t value) {
    for (int index = 0; index < 8; ++index)
        output.put(static_cast<char>((value >> (8 * index)) & 0xffU));
}

std::string shell_quote(const fs::path& path) {
    std::string quoted = "'";
    for (char character : path.string()) {
        if (character == '\'')
            quoted += "'\\''";
        else
            quoted += character;
    }
    return quoted + "'";
}

std::string read_binary(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("failed to read optimized-runtime test DSO");
    return {(std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>()};
}

std::string read_text(const fs::path& path) {
    std::ifstream input(path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void update_tree_record(trtmc::internal::Sha256& tree, const std::string& kind,
                        const std::string& path) {
    constexpr char terminator = '\0';
    tree.update(kind);
    tree.update(&terminator, 1);
    tree.update(path);
    tree.update(&terminator, 1);
}

std::string artifact_tree_hash(const std::vector<std::pair<std::string, std::string>>& files) {
    trtmc::internal::Sha256 tree;
    update_tree_record(tree, "directory", "payload");
    for (const auto& [path, contents] : files) {
        trtmc::internal::Sha256 file;
        file.update(contents);
        const auto digest = file.digest();
        update_tree_record(tree, "file", path);
        tree.update(std::to_string(contents.size()));
        constexpr char terminator = '\0';
        tree.update(&terminator, 1);
        tree.update(digest.data(), digest.size());
    }
    return tree.hex_digest();
}

struct NamedSection {
    std::string name;
    std::string contents;
};

void write_optimized_bundle(const fs::path& path) {
    std::vector<std::pair<std::string, std::string>> artifacts = {
        {"payload/runtime.data", "synthetic-runtime-data"},
        {kRuntimeLibrary, read_binary(TRTMC_TEST_OPTIMIZED_PROVIDER_DSO)},
    };
    std::sort(artifacts.begin(), artifacts.end());
    std::uint64_t total_size = 0;
    for (const auto& artifact : artifacts)
        total_size += artifact.second.size();

    std::ostringstream descriptor;
    descriptor << R"({"schema_version":2,"implementation_id":")" << kImplementationId
               << R"(","model_id":")" << kModelId << R"(","profile_id":"generic-profile",)"
               << R"("runtime_library":")" << kRuntimeLibrary
               << R"(","factory_abi":1,"implementation_metadata_section":"implementation.json",)"
               << R"("runtime":{)"
               << R"("name":"Example Optimized Runtime","version":"1.2.3",)"
               << R"("commit":"0123456789abcdef0123456789abcdef01234567"},)"
               << R"("artifact":{"section_prefix":"optimized_runtime_artifacts",)"
               << R"("directories":["payload"],"file_count":)" << artifacts.size()
               << R"(,"total_size":)" << total_size << R"(,"tree_sha256":")"
               << artifact_tree_hash(artifacts) << R"("}})";

    std::vector<NamedSection> sections = {
        {"optimized_runtime.json", descriptor.str()},
        {"implementation.json", R"({"capsule":"example-test"})"},
    };
    for (auto& [name, contents] : artifacts)
        sections.push_back({"optimized_runtime_artifacts/" + name, std::move(contents)});

    std::ostringstream header;
    header << R"({"model_id":")" << kModelId
           << R"(","model_type":"optimized_runtime","family":"optimized_runtime",)"
           << R"("precision":"","vocab_size":0,)"
           << R"("max_cache_length":0,"sections":{)";
    std::uint64_t offset = 0;
    for (std::size_t index = 0; index < sections.size(); ++index) {
        if (index != 0)
            header << ',';
        header << '"' << sections[index].name << R"(":{"offset":)" << offset << R"(,"size":)"
               << sections[index].contents.size() << '}';
        offset += sections[index].contents.size();
    }
    header << "}}";

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("failed to create optimized-runtime inspect bundle");
    output.write(reinterpret_cast<const char*>(trtmc::kBundleMagic), 8);
    const std::string header_text = header.str();
    write_u64_le(output, header_text.size());
    output.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));
    for (const auto& section : sections) {
        output.write(section.contents.data(),
                     static_cast<std::streamsize>(section.contents.size()));
    }
}

int run_inspect(const fs::path& bundle, const fs::path& output, const fs::path& error) {
    std::ostringstream command;
    command << shell_quote(TRTMC_TEST_CLI) << " inspect " << shell_quote(bundle);
    command << " >" << shell_quote(output) << " 2>" << shell_quote(error);
    return std::system(command.str().c_str());
}

} // namespace

int main() {
    const fs::path root =
        fs::temp_directory_path() /
        ("trtmc-optimized-inspect-" + std::to_string(static_cast<long long>(getpid())));
    fs::remove_all(root);
    fs::create_directories(root);
    const fs::path bundle = root / "synthetic-optimized.trtfb";
    write_optimized_bundle(bundle);

    const fs::path inspect_out = root / "inspect.out";
    const fs::path inspect_err = root / "inspect.err";
    check(run_inspect(bundle, inspect_out, inspect_err) == 0,
          "generic optimized-runtime bundle inspection succeeds");
    const std::string inspected = read_text(inspect_out);
    check(inspected.find("Model ID:           Example/Optimized-Model") != std::string::npos,
          "bundle inspection reports the public model identity");
    check(inspected.find("optimized_runtime.json") != std::string::npos,
          "bundle inspection reports generic dispatch metadata");
    check(inspected.find("optimized_runtime_artifacts/payload/runtime.data") != std::string::npos,
          "bundle inspection reports opaque capsule artifacts");
    check(inspected.find("Runtime strategy:") == std::string::npos,
          "shared inspection does not invent a modality strategy");
    check(read_text(inspect_err).empty(), "valid bundle inspection has no error output");

    fs::remove_all(root);
    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "All optimized-runtime inspect CLI tests passed.\n";
    return 0;
}
