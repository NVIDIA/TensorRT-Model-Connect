/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/bundle/bundle_format.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <unistd.h>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

std::filesystem::path temp_dir() {
    char pattern[] = "/tmp/trtmc_bundle_v1_XXXXXX";
    char* path = mkdtemp(pattern);
    if (path == nullptr)
        throw std::runtime_error("mkdtemp failed");
    return path;
}

void write_bundle(const std::filesystem::path& path, const std::string& header,
                  const std::string& payload = {}) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(trtmc::kBundleMagic), 8);
    const std::uint64_t length = header.size();
    for (int shift = 0; shift < 64; shift += 8)
        output.put(static_cast<char>((length >> shift) & 0xffU));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write(payload.data(), static_cast<std::streamsize>(payload.size()));
}

bool read_throws(const std::filesystem::path& path) {
    try {
        (void)trtmc::BundleReader(path.string());
        return false;
    } catch (const std::runtime_error&) {
        return true;
    }
}

class BoundedSink final : public std::streambuf {
  public:
    std::size_t total{0};
    std::size_t largest_write{0};
    bool valid{true};

  protected:
    std::streamsize xsputn(const char* data, std::streamsize count) override {
        largest_write = std::max(largest_write, static_cast<std::size_t>(count));
        total += static_cast<std::size_t>(count);
        valid = valid && std::all_of(data, data + count, [](char value) { return value == 'x'; });
        return count;
    }
};

} // namespace

int main() {
    const auto directory = temp_dir();
    const auto valid = directory / "valid.bundle";
    write_bundle(
        valid,
        R"({"format":1,"family":"fake","task":"time_series_forecast","backend":"fake","sections":{"engine.plan":{"offset":0,"length":4}}})",
        "PLAN");

    const trtmc::BundleReader valid_reader(valid.string());
    check(valid_reader.info().format == 1, "format is one");
    check(valid_reader.info().family == "fake", "family parsed");
    check(valid_reader.info().task == "time_series_forecast", "task parsed");
    check(valid_reader.info().backend == "fake", "backend parsed");
    check(valid_reader.info().sections.size() == 1, "one section descriptor");
    check(valid_reader.info().sections.front().length == 4, "section length parsed");
    check(valid_reader.read_section("engine.plan").size() == 4, "section payload read");

    const auto original_directory = std::filesystem::current_path();
    std::filesystem::current_path(directory);
    std::filesystem::create_directory(directory / "nested");
    const trtmc::BundleReader reader("nested/../valid.bundle");
    std::filesystem::current_path(original_directory);
    check(std::filesystem::path(reader.path()).is_absolute(), "reader path is absolute");
    check(std::filesystem::path(reader.path()) == valid.lexically_normal(),
          "reader path is lexically normalized");
    const auto lazy_plan = reader.read_section("engine.plan");
    check(std::string(lazy_plan.begin(), lazy_plan.end()) == "PLAN",
          "file-backed reader owns an absolute path");

    const auto large = directory / "large.bundle";
    const std::string large_payload(2 * 1024 * 1024 + 17, 'x');
    write_bundle(large,
                 "{\"format\":1,\"family\":\"fake\",\"task\":\"time_series_forecast\","
                 "\"backend\":\"fake\",\"sections\":{\"engine.plan\":{\"offset\":0,\"length\":" +
                     std::to_string(large_payload.size()) + "}}}",
                 large_payload);
    const trtmc::BundleReader large_reader(large.string());
    BoundedSink buffer;
    std::ostream sink(&buffer);
    large_reader.copy_section("engine.plan", sink);
    check(buffer.total == large_payload.size(), "streaming copy writes the complete section");
    check(buffer.largest_write <= 1024 * 1024, "streaming copy uses bounded chunks");
    check(buffer.valid, "streaming copy preserves section bytes");

    const auto old_size = directory / "old-size.bundle";
    write_bundle(
        old_size,
        R"({"format":1,"family":"fake","task":"time_series_forecast","backend":"fake","sections":{"engine.plan":{"offset":0,"size":4}}})",
        "PLAN");
    check(read_throws(old_size), "old size field rejected");

    const auto old_header = directory / "old-header.bundle";
    write_bundle(
        old_header,
        R"({"format":1,"family":"fake","task":"time_series_forecast","backend":"fake","model_id":"legacy","sections":{}})");
    check(read_throws(old_header), "extra legacy header field rejected");

    const auto out_of_bounds = directory / "out-of-bounds.bundle";
    write_bundle(
        out_of_bounds,
        R"({"format":1,"family":"fake","task":"time_series_forecast","backend":"fake","sections":{"engine.plan":{"offset":0,"length":5}}})",
        "PLAN");
    check(read_throws(out_of_bounds), "out of bounds section rejected");

    std::filesystem::remove_all(directory);
    std::cerr << (failures == 0 ? "ALL PASSED\n" : "SOME FAILED\n");
    return failures;
}
