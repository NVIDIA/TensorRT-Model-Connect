/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2/sam2_jpeg_decoder.h"
#include "utils/sha256.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kExpectedRgbBytes = 1280U * 1088U * 3U;

template <typename Function>
void expectThrows(Function&& function, std::string_view context,
                  std::string_view expected_message = {}) {
    try {
        function();
    } catch (const std::exception& error) {
        if (!expected_message.empty() &&
            std::string_view(error.what()).find(expected_message) == std::string_view::npos) {
            throw std::runtime_error(std::string(context) +
                                     " threw the wrong error: " + error.what());
        }
        return;
    }
    throw std::runtime_error(std::string(context) + " did not throw");
}

std::vector<std::uint8_t> readBytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
        throw std::runtime_error("unable to open SAM2 JPEG test fixture: " + path.string());
    const auto end = input.tellg();
    if (end < 0)
        throw std::runtime_error("unable to size SAM2 JPEG test fixture");
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!input)
        throw std::runtime_error("unable to read SAM2 JPEG test fixture");
    return bytes;
}

std::string sha256(const std::vector<std::uint8_t>& bytes) {
    trtmc::internal::Sha256 hash;
    hash.update(bytes.data(), bytes.size());
    return hash.hex_digest();
}

class TempDirectory {
  public:
    TempDirectory() {
        std::array<char, 40> pattern{};
        constexpr std::string_view kTemplate = "/tmp/trtmc-sam2-jpeg-XXXXXX";
        std::copy(kTemplate.begin(), kTemplate.end(), pattern.begin());
        if (::mkdtemp(pattern.data()) == nullptr)
            throw std::runtime_error("unable to create SAM2 JPEG test directory");
        path_ = pattern.data();
    }
    TempDirectory(const TempDirectory&) = delete;
    TempDirectory& operator=(const TempDirectory&) = delete;
    ~TempDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

  private:
    std::filesystem::path path_;
};

void writeBytes(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    if (!output)
        throw std::runtime_error("unable to write SAM2 JPEG negative test fixture");
}

void testOwnedBytesAndErrorManager() {
    expectThrows([] { (void)trtmc::sam2::decodeSam2JpegBytes({}); }, "empty byte input",
                 "must not be empty");

    // Repeated malformed streams exercise libjpeg's error_exit/longjmp path.
    // A sanitizer run verifies that each partially created decoder is released.
    const std::vector<std::uint8_t> truncated = {0xffU, 0xd8U, 0xffU, 0xc0U};
    for (int attempt = 0; attempt < 32; ++attempt) {
        expectThrows([&] { (void)trtmc::sam2::decodeSam2JpegBytes(truncated); },
                     "truncated byte input", "decode failed");
    }

    // jpeg_read_header accepts this complete SOF/SOS envelope. The decoder
    // must reject its 1x1 geometry before asking libjpeg for pixel storage.
    const std::vector<std::uint8_t> one_by_one_header = {
        0xffU, 0xd8U, 0xffU, 0xc0U, 0x00U, 0x11U, 0x08U, 0x00U, 0x01U, 0x00U, 0x01U, 0x03U, 0x01U,
        0x11U, 0x00U, 0x02U, 0x11U, 0x00U, 0x03U, 0x11U, 0x00U, 0xffU, 0xdaU, 0x00U, 0x0cU, 0x03U,
        0x01U, 0x00U, 0x02U, 0x00U, 0x03U, 0x00U, 0x00U, 0x3fU, 0x00U, 0xffU, 0xd9U,
    };
    expectThrows([&] { (void)trtmc::sam2::decodeSam2JpegBytes(one_by_one_header); },
                 "wrong JPEG geometry", "must be 1088x1280");
}

void testRegularFileContract() {
    TempDirectory temporary;
    const auto empty = temporary.path() / "empty.jpg";
    writeBytes(empty, {});
    expectThrows([&] { (void)trtmc::sam2::decodeSam2JpegFile(empty); }, "empty regular file",
                 "must not be empty");

    const auto malformed = temporary.path() / "malformed.jpg";
    writeBytes(malformed, {0xffU, 0xd8U, 0xffU, 0xd9U});
    expectThrows([&] { (void)trtmc::sam2::decodeSam2JpegFile(malformed); },
                 "malformed regular file", "decode failed");

    const auto symlink = temporary.path() / "symlink.jpg";
    std::filesystem::create_symlink(malformed, symlink);
    expectThrows([&] { (void)trtmc::sam2::decodeSam2JpegFile(symlink); }, "symlink input");
    expectThrows([&] { (void)trtmc::sam2::decodeSam2JpegFile(temporary.path()); },
                 "directory input", "regular file");
    expectThrows([&] { (void)trtmc::sam2::decodeSam2JpegFile(temporary.path() / "missing.jpg"); },
                 "missing input");

    const auto oversized = temporary.path() / "oversized.jpg";
    writeBytes(oversized, {0U});
    std::filesystem::resize_file(oversized, 64U * 1024U * 1024U + 1U);
    expectThrows([&] { (void)trtmc::sam2::decodeSam2JpegFile(oversized); },
                 "oversized regular file", "64 MiB");
}

void testExactDeliveredFramesIfAvailable() {
    const char* directory_text = std::getenv("TRTMC_SAM2_JPEG_DIR");
    if (directory_text == nullptr || *directory_text == '\0') {
        std::cout << "SKIP: set TRTMC_SAM2_JPEG_DIR for exact five-frame JPEG parity\n";
        return;
    }

    constexpr std::array<std::string_view, 5> kEncodedSha256 = {
        "8a398f40747d5053cfc0d47d45090f2070a10afa4722e7d5b827a6ad0825a5aa",
        "2871555bca47da7473762ca87314b17bd55d100a0f982f78d6449080ff86856f",
        "5594181db7dd1c5da3ce05b945f74e66a5d8d098d71a7cb9e5e43834a393bbe2",
        "c3abc03371458939d09faf331749c2a87cc6fc91128eaab3901b179adb096a35",
        "3d8ea6042c82e7b340277c00666c4c2cefbae5de265ef06a71fe964905ed720b",
    };
    constexpr std::array<std::string_view, 5> kDecodedSha256 = {
        "0bcadde0e5a6f8ba04f79c44f064c5b00d3cd1b250e2f2f3bbf10ef0630a9ce9",
        "0abfd57f9e3886a8c3068bf6bcc353b26d1e3a8a43819a80dfeb00f309b24ec3",
        "9166cc263c3edb262065fa3b98ee062cbf6d781dd656bae13def7f4141b7d025",
        "77525faadfc8a607e4e1556135887caaddd0b64d7cd677fcf47c38ecf9e25a4f",
        "cb0801b490ba13dfb6d36aeef06b049ff67ff11864ef62ccd858a0096d97c6af",
    };

    for (std::size_t frame = 0; frame < kDecodedSha256.size(); ++frame) {
        const std::string filename = "00000" + std::to_string(frame) + ".jpg";
        const auto path = std::filesystem::path(directory_text) / filename;
        auto encoded = readBytes(path);
        if (sha256(encoded) != kEncodedSha256[frame])
            throw std::runtime_error("delivered SAM2 JPEG hash mismatch: " + filename);

        const auto from_file = trtmc::sam2::decodeSam2JpegFile(path);
        const auto from_owned_bytes = trtmc::sam2::decodeSam2JpegBytes(std::move(encoded));
        if (from_file.height != trtmc::sam2::kOriginalImageHeight ||
            from_file.width != trtmc::sam2::kOriginalImageWidth ||
            from_file.rgb_hwc.size() != kExpectedRgbBytes ||
            from_file.rgb_hwc != from_owned_bytes.rgb_hwc ||
            sha256(from_file.rgb_hwc) != kDecodedSha256[frame]) {
            throw std::runtime_error("SAM2 libjpeg RGB parity drifted: " + filename);
        }
        std::cout << "PARITY: " << filename << " decoded_sha256=" << kDecodedSha256[frame] << '\n';
    }
}

} // namespace

int main() {
    testOwnedBytesAndErrorManager();
    testRegularFileContract();
    testExactDeliveredFramesIfAvailable();
    std::cout << "PASS: strict pure C++ SAM2 libjpeg decoder\n";
    return 0;
}
