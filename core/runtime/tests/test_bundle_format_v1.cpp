/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#if defined(TRTMC_TEST_BUNDLE_STREAM_READER)

#include "runtime/bundle/bundle_format.h"
#include "trtmc/bundle.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

void check(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

template <class Function>
void rejects(Function function, const char* message) {
    bool failed = false;
    try {
        function();
    } catch (const std::exception&) {
        failed = true;
    }
    check(failed, message);
}

struct TemporaryDirectory {
    std::filesystem::path path;
    TemporaryDirectory() {
        std::string pattern =
            (std::filesystem::temp_directory_path() / "trtmc-stream-XXXXXX").string();
        auto* created = ::mkdtemp(pattern.data());
        if (!created)
            throw std::runtime_error("Cannot create owned test directory");
        path = created;
    }
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

std::uint64_t write_bundle(const std::filesystem::path& path, std::uint64_t length) {
    const auto header = std::string("{\"format\":1,\"family\":\"fake\",\"task\":\"fake\","
                                    "\"backend\":\"fake\",\"sections\":{") +
                        "\"before\":{\"offset\":0,\"length\":7}," +
                        "\"engine.plan\":{\"offset\":7,\"length\":" + std::to_string(length) +
                        "}," + "\"empty\":{\"offset\":7,\"length\":0}," +
                        "\"after\":{\"offset\":" + std::to_string(7 + length) + ",\"length\":4}}}";
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(trtmc::kBundleMagic), sizeof(trtmc::kBundleMagic));
    for (int byte = 0; byte != 8; ++byte)
        file.put(static_cast<char>(header.size() >> (byte * 8)));
    file.write(header.data(), static_cast<std::streamsize>(header.size()));
    const auto offset = static_cast<std::uint64_t>(file.tellp()) + 7;
    file.write("BEFORE!first", 12);
    file.seekp(static_cast<std::streamoff>(offset + length - 4));
    file.write("lastTAIL", 8);
    check(static_cast<bool>(file), "Cannot create sparse test bundle");
    return offset;
}

void check_large_stream(const std::filesystem::path& path, std::uint64_t length) {
    // A section-sized vector cannot fit, whereas bounded reads must succeed.
    const auto pid = ::fork();
    check(pid >= 0, "Cannot fork memory-bounded test");
    if (pid == 0) {
        const rlimit limit{128U << 20, 128U << 20};
        if (::setrlimit(RLIMIT_AS, &limit) != 0)
            ::_exit(2);
        try {
            const trtmc::BundleReader bundle(path.string());
            auto reader = bundle.open_section("engine.plan");
            check(reader->size() == length, "64-bit section extent changed");
            std::array<char, 65536> bytes{};
            for (const auto position : {1ULL << 20, 1ULL << 32, 49ULL << 30}) {
                check(reader->seek(position, trtmc::PlanSeekOrigin::begin), "Large seek failed");
                check(reader->read(bytes.data(), bytes.size()) == bytes.size(),
                      "Large read failed");
                for (char value : bytes)
                    check(value == 0, "Sparse source bytes changed");
            }
            ::_exit(0);
        } catch (...) {
            ::_exit(3);
        }
    }
    int status = 0;
    check(::waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "Stream reader allocated a section-sized buffer or failed bounded reads");
}

} // namespace

int main() {
    try {
        TemporaryDirectory directory;
        const auto path = directory.path / "source.bundle";
        const std::uint64_t length = 50ULL << 30;
        const auto offset = write_bundle(path, length);
        check_large_stream(path, length);

        trtmc::BundleReader bundle(path.string());
        auto reader = bundle.open_section("engine.plan");
        auto independent = trtmc::BundleReader(bundle).open_section("engine.plan");
        std::array<char, 16> bytes{};
        check(reader->read(bytes.data(), 5) == 5 && std::string(bytes.data(), 5) == "first",
              "Section starts at the wrong file offset");
        check(independent->read(bytes.data(), 5) == 5 && std::string(bytes.data(), 5) == "first",
              "Independent reader did not own its cursor and file");
        check(
            !reader->seek(std::numeric_limits<std::int64_t>::min(), trtmc::PlanSeekOrigin::current),
            "Signed minimum seek accepted");
        check(!reader->seek(std::numeric_limits<std::int64_t>::max(), trtmc::PlanSeekOrigin::end),
              "Overflowing end-relative seek accepted");
        check(!reader->seek(0, static_cast<trtmc::PlanSeekOrigin>(999)), "Invalid origin accepted");
        check(reader->seek(-5, trtmc::PlanSeekOrigin::current), "Invalid seek moved the cursor");
        check(reader->seek(-2, trtmc::PlanSeekOrigin::end), "End-relative seek failed");
        bytes.fill('!');
        check(reader->read(bytes.data(), bytes.size()) == 2 && bytes[0] == 's' && bytes[1] == 't',
              "End read lost its final bytes");
        for (std::size_t i = 2; i != bytes.size(); ++i)
            check(bytes[i] == '!', "Read escaped the section boundary");
        check(reader->read(bytes.data(), bytes.size()) == 0, "EOF read leaked the next section");
        auto empty = bundle.open_section("empty");
        check(empty->size() == 0 && empty->read(nullptr, 1) == 0 &&
                  empty->seek(0, trtmc::PlanSeekOrigin::end),
              "Empty section contract changed");
        rejects([&] { bundle.open_section("missing"); }, "Missing section accepted");
        check(reader->seek(0, trtmc::PlanSeekOrigin::begin), "Rewind failed");
        rejects([&] { reader->read(nullptr, 1); }, "Null nonempty destination accepted");
        check(reader->read(nullptr, 0) == 0, "Zero-byte read failed");

        // Preserve length while changing bytes: size checks alone are insufficient.
        {
            std::fstream changed(path, std::ios::in | std::ios::out | std::ios::binary);
            changed.seekp(static_cast<std::streamoff>(offset));
            changed.put('X');
        }
        rejects([&] { reader->read(bytes.data(), 1); }, "In-place bundle mutation was ignored");
        rejects([&] { independent->seek(0, trtmc::PlanSeekOrigin::begin); },
                "Mutation ignored on seek");
        std::filesystem::resize_file(path, offset + 8);
        rejects([&] { bundle.open_section("engine.plan"); }, "Changed file extent accepted");
        std::cout << "Bundle stream bounds, identity and 50-GiB sparse reads passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

#else

#include "runtime/bundle/bundle_format.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
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

/// Verify bounded-chunk section boundaries, empty sections and late I/O failures.
void test_copy_section(const std::filesystem::path& directory) {
    const auto path = directory / "stream.bundle";
    const std::string payload(2 * 64 * 1024 + 17, 'x');
    const std::string header =
        R"({"format":1,"family":"fake","task":"text","backend":"fake","sections":{"data":{"offset":3,"length":)" +
        std::to_string(payload.size()) + R"(},"empty":{"offset":0,"length":0}}})";
    write_bundle(path, header, "PRE" + payload + "POST");
    const trtmc::BundleReader reader(path.string());
    std::ostringstream output;
    reader.copy_section("data", output);
    check(output.str() == payload, "stream copy preserves boundaries across chunks");
    reader.copy_section("empty", output);
    check(output.str() == payload, "empty section appends nothing");
    auto fails = [&](const char* section, std::ostream& destination) {
        try {
            reader.copy_section(section, destination);
            return false;
        } catch (const std::runtime_error&) {
            return true;
        }
    };
    check(fails("missing", output), "stream copy rejects missing sections");
    std::ostringstream broken;
    broken.setstate(std::ios::badbit);
    check(fails("data", broken), "stream copy reports output failure");
    std::filesystem::resize_file(path, 16 + header.size() + 3 + payload.size() - 1);
    check(fails("data", output), "stream copy detects truncation after validation");
}

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

    test_copy_section(directory);
    std::filesystem::remove_all(directory);
    std::cerr << (failures == 0 ? "ALL PASSED\n" : "SOME FAILED\n");
    return failures;
}

#endif
