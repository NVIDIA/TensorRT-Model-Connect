/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Dynamic-HD preprocessing against goldens from the checkpoint's own
// Phi4MMImageProcessor (tests/data/dynamic-hd/generate.py).

#include "families/phi4_multimodal/runtime/image_preprocessor.h"
#include "families/phi4_multimodal/runtime/phi4_dynamic_hd.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace {
int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

trtmc::runtime::adapters::io::DecodedImage load_png(const std::string& path) {
    trtmc::runtime::adapters::io::DecodedImage image;
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* pixels = stbi_load(path.c_str(), &width, &height, &channels, 3);
    if (pixels == nullptr)
        return image;
    image.pixels.assign(pixels, pixels + static_cast<std::size_t>(width) * height * 3);
    stbi_image_free(pixels);
    image.width = width;
    image.height = height;
    image.channels = 3;
    return image;
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: test_phi4_dynamic_hd DATA_DIR\n";
        return 2;
    }
    const std::string directory = argv[1];
    std::ifstream stream(directory + "/hf-dynamic-hd.json");
    const auto goldens = nlohmann::json::parse(stream);

    trtmc::Phi4MultimodalPreprocessConfig config;
    config.preprocessor_type = "phi4_hd_chw";
    config.fixed_image_size = 448;
    for (int c = 0; c < 3; ++c) {
        config.image_mean[c] = 0.5F;
        config.image_std[c] = 0.5F;
    }

    for (const auto& golden : goldens.at("cases")) {
        const std::string name = golden.at("image");
        const auto image = load_png(directory + "/" + name);
        check(!image.empty(), name + ": image loads");
        const auto layout = trtmc::phi4_multimodal::hd_layout(image.width, image.height);
        check(layout.crop_cols == golden.at("grid")[0] && layout.crop_rows == golden.at("grid")[1],
              name + ": crop grid");

        const auto out = trtmc::phi4_multimodal_preprocess_decoded_image(image, config);
        check(out.ok, name + ": preprocessing succeeds");
        if (!out.ok)
            continue;
        const auto& valid = golden.at("crop_valid");
        check(out.hd_crop_valid.size() == valid.size() * 2, name + ": vision batch entries");
        for (std::size_t entry = 0;
             entry < valid.size() && entry * 2 + 1 < out.hd_crop_valid.size(); ++entry)
            check(out.hd_crop_valid[entry * 2] == valid[entry][0] &&
                      out.hd_crop_valid[entry * 2 + 1] == valid[entry][1],
                  name + ": valid patch rows/cols of entry " + std::to_string(entry));
        check(static_cast<int>(out.hd_token_rows.size()) == golden.at("num_image_tokens"),
              name + ": image token count");
        check(out.pixel_values.size() == valid.size() * 3U * 448U * 448U, name + ": pixel count");

        double crop_error = 0.0;
        double global_error = 0.0;
        for (const auto& sample : golden.at("pixel_samples")) {
            const std::size_t entry = sample[0];
            const std::size_t index =
                ((entry * 3 + sample[1].get<std::size_t>()) * 448 + sample[2].get<std::size_t>()) *
                    448 +
                sample[3].get<std::size_t>();
            if (index >= out.pixel_values.size())
                continue;
            const double error = std::abs(out.pixel_values[index] - sample[4].get<double>());
            (entry == 0 ? global_error : crop_error) =
                std::max(entry == 0 ? global_error : crop_error, error);
        }
        // The crops are 8-bit resizes: exact. The global view is a float
        // bicubic resize, equal up to float summation order.
        check(crop_error == 0.0,
              name + ": crop pixels exact (max error " + std::to_string(crop_error) + ")");
        check(global_error <= 1e-4,
              name + ": global view pixels (max error " + std::to_string(global_error) + ")");
    }

    // Token rows follow sub_glb: every row ends with sub_GN (row 0), then glb_GN
    // (row 1) and the 16 x 17 global view whose features start at row 2.
    const auto single = trtmc::phi4_multimodal::hd_layout(448, 448);
    check(single.token_rows.size() == 16 * 17 + 1 + 16 * 17, "448x448 token count");
    check(single.token_rows[16] == 0 && single.token_rows[16 * 17] == 1,
          "448x448 separator positions");
    check(single.token_rows[0] == 2 + 256 && single.token_rows[16 * 17 + 1] == 2,
          "448x448 first crop and global feature rows");

    if (failures == 0)
        std::cout << "test_phi4_dynamic_hd: all checks passed\n";
    return failures == 0 ? 0 : 1;
}
