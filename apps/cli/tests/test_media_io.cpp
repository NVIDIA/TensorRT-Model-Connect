/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/io.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    int failures = 0;
    const auto check = [&](bool condition, const std::string& message) {
        if (!condition) {
            std::cerr << message << '\n';
            ++failures;
        }
    };
    const fs::path fixtures(argv[1]);
    for (const auto* name : {"rgb-444", "rgb-422", "rgb-420", "progressive-420", "grayscale"}) {
        const auto actual = trtmc::cli::io::read_image((fixtures / (std::string(name) + ".jpg")).string());
        const auto golden = trtmc::cli::io::read_image((fixtures / (std::string(name) + ".png")).string());
        check(actual.width == 37 && actual.height == 29 &&
                  golden.width == 37 && golden.height == 29,
              std::string(name) + ": original dimensions must be preserved");
        check(!actual.empty() && actual.pixels == golden.pixels,
              std::string(name) + ": decoded RGB must exactly match Pillow");
    }
    check(trtmc::cli::io::read_image((fixtures / "missing.jpg").string()).empty(),
          "missing input must return an empty image");

    // The decoder is selected by content, not by a filename extension. Use a
    // unique task directory so separate test processes cannot overwrite it.
    const auto temporary = fs::temp_directory_path() /
        ("trtmc-jpeg-" + std::to_string(
            fs::file_time_type::clock::now().time_since_epoch().count()));
    fs::create_directory(temporary);
    fs::copy_file(fixtures / "rgb-420.jpg", temporary / "input.bin");
    check(trtmc::cli::io::read_image((temporary / "input.bin").string()).pixels ==
              trtmc::cli::io::read_image((fixtures / "rgb-420.png").string()).pixels,
          "JPEG decoding must not depend on the extension");

    std::ifstream source(fixtures / "rgb-420.jpg", std::ios::binary);
    const std::vector<char> bytes{std::istreambuf_iterator<char>(source), {}};
    for (const auto size : {std::size_t{2}, bytes.size() / 2, bytes.size() - 2}) {
        const auto path = temporary / ("truncated-" + std::to_string(size) + ".jpg");
        {
            std::ofstream output(path, std::ios::binary);
            output.write(bytes.data(), static_cast<std::streamsize>(size));
        }
        check(trtmc::cli::io::read_image(path.string()).empty(),
              "truncated JPEG must not be accepted with partial pixels");
    }
    {
        std::ofstream corrupt(temporary / "corrupt.jpg", std::ios::binary);
        const unsigned char invalid[] = {0xff, 0xd8, 0xff, 0xd9};
        corrupt.write(reinterpret_cast<const char*>(invalid), sizeof(invalid));
    }
    check(trtmc::cli::io::read_image((temporary / "corrupt.jpg").string()).empty(),
          "invalid JPEG header must return an empty image");
    // A bad decode must not poison another independent decoder instance.
    check(!trtmc::cli::io::read_image((fixtures / "rgb-420.jpg").string()).empty(),
          "valid input must still decode after an invalid JPEG");
    fs::remove_all(temporary);
    return failures == 0 ? 0 : 1;
}
