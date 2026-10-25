/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
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
