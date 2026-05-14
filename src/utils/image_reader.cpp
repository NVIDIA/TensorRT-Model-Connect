// image_reader.cpp — implements trtmc::io::read_image() and
// trtmc::io::save_png() using stb_image / stb_image_write.

#include "stb_image.h"
#include "stb_image_write.h"
#include "trtmc/trtmc_io.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::io {

LoadedImage read_image(const std::string& path) {
    LoadedImage result;

    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* raw = stbi_load(path.c_str(), &width, &height, &channels, 3);
    if (raw == nullptr) {
        return result; // empty = decode failure
    }

    result.width = static_cast<int32_t>(width);
    result.height = static_cast<int32_t>(height);

    // Convert uint8 [0, 255] to float [0, 1] in HWC layout
    auto npixels = static_cast<std::size_t>(width) * height * 3;
    result.pixels.resize(npixels);
    for (std::size_t i = 0; i < npixels; ++i) {
        result.pixels[i] = static_cast<float>(raw[i]) / 255.0F;
    }

    stbi_image_free(raw);
    return result;
}

void save_png(const std::string& path, const std::vector<float>& hwc_pixels, int width,
              int height) {
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("save_png: width/height must be positive (got " +
                                 std::to_string(width) + "x" + std::to_string(height) + ")");
    }
    const std::size_t expected =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3U;
    if (hwc_pixels.size() != expected) {
        throw std::runtime_error("save_png: pixel buffer size mismatch (expected " +
                                 std::to_string(expected) + ", got " +
                                 std::to_string(hwc_pixels.size()) + ")");
    }
    std::vector<std::uint8_t> out(expected);
    for (std::size_t i = 0; i < expected; ++i) {
        const float v = std::max(0.0F, std::min(1.0F, hwc_pixels[i]));
        out[i] = static_cast<std::uint8_t>(v * 255.0F + 0.5F);
    }
    const int stride = width * 3;
    if (!stbi_write_png(path.c_str(), width, height, 3, out.data(), stride)) {
        throw std::runtime_error("save_png: stbi_write_png failed for " + path);
    }
}

} // namespace trtmc::io
