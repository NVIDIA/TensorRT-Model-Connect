#pragma once

// trtmc_io.hpp — header-only file I/O utilities for trtmc pipeline results.
//
// Usage:
//   #include <trtmc/trtmc_io.hpp>
//   auto pipe = trtmc::load("model.trtfb");
//   auto img = pipe->generate_image("a cat");
//   trtmc::io::save_png(img, "output.png");

#include "trtmc/pipeline.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace trtmc::io {

// Write a WAV file from AudioResult (IEEE float32 mono).
inline void write_wav(const AudioResult& audio, const std::string& path)
{
    if (audio.samples.empty())
        throw std::runtime_error("write_wav: empty audio");

    std::ofstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("write_wav: cannot open " + path);

    const int32_t num_samples = static_cast<int32_t>(audio.samples.size());
    const int32_t sample_rate = audio.sample_rate;
    const int16_t num_channels = 1;
    const int16_t bits_per_sample = 32;
    const int32_t byte_rate = sample_rate * num_channels * (bits_per_sample / 8);
    const int16_t block_align = static_cast<int16_t>(num_channels * (bits_per_sample / 8));
    const int32_t data_size = num_samples * block_align;
    const int32_t chunk_size = 36 + data_size;

    // RIFF header
    f.write("RIFF", 4);
    f.write(reinterpret_cast<const char*>(&chunk_size), 4);
    f.write("WAVE", 4);

    // fmt chunk
    f.write("fmt ", 4);
    int32_t fmt_size = 16;
    int16_t audio_format = 3; // IEEE float
    f.write(reinterpret_cast<const char*>(&fmt_size), 4);
    f.write(reinterpret_cast<const char*>(&audio_format), 2);
    f.write(reinterpret_cast<const char*>(&num_channels), 2);
    f.write(reinterpret_cast<const char*>(&sample_rate), 4);
    f.write(reinterpret_cast<const char*>(&byte_rate), 4);
    f.write(reinterpret_cast<const char*>(&block_align), 2);
    f.write(reinterpret_cast<const char*>(&bits_per_sample), 2);

    // data chunk
    f.write("data", 4);
    f.write(reinterpret_cast<const char*>(&data_size), 4);
    f.write(reinterpret_cast<const char*>(audio.samples.data()),
            static_cast<std::streamsize>(data_size));
}

// Read a WAV file into an AudioResult (mono float32).
inline AudioResult read_wav(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("read_wav: cannot open " + path);

    // Read RIFF header
    char riff[4];
    f.read(riff, 4);
    if (std::string(riff, 4) != "RIFF")
        throw std::runtime_error("read_wav: not a RIFF file");

    int32_t chunk_size = 0;
    f.read(reinterpret_cast<char*>(&chunk_size), 4);

    char wave[4];
    f.read(wave, 4);
    if (std::string(wave, 4) != "WAVE")
        throw std::runtime_error("read_wav: not a WAVE file");

    int32_t sample_rate = 0;
    int16_t num_channels = 0;
    int16_t audio_format = 0;
    int16_t bits_per_sample = 0;

    // Find fmt and data chunks
    std::vector<char> data_bytes;
    while (f)
    {
        char id[4];
        if (!f.read(id, 4)) break;
        int32_t size = 0;
        f.read(reinterpret_cast<char*>(&size), 4);

        if (std::string(id, 4) == "fmt ")
        {
            f.read(reinterpret_cast<char*>(&audio_format), 2);
            f.read(reinterpret_cast<char*>(&num_channels), 2);
            f.read(reinterpret_cast<char*>(&sample_rate), 4);
            f.seekg(4, std::ios::cur); // byte_rate
            f.seekg(2, std::ios::cur); // block_align
            f.read(reinterpret_cast<char*>(&bits_per_sample), 2);
            if (size > 16)
                f.seekg(size - 16, std::ios::cur);
        }
        else if (std::string(id, 4) == "data")
        {
            data_bytes.resize(static_cast<std::size_t>(size));
            f.read(data_bytes.data(), size);
        }
        else
        {
            f.seekg(size, std::ios::cur);
        }
    }

    // Decode samples based on format
    AudioResult result;
    result.sample_rate = sample_rate;
    const auto nc = std::max<int16_t>(num_channels, 1);
    if (audio_format == 3 && bits_per_sample == 32)
    {
        // IEEE float32
        std::size_t n = data_bytes.size() / (sizeof(float) * nc);
        result.samples.resize(n);
        const auto* fp = reinterpret_cast<const float*>(data_bytes.data());
        if (nc <= 1) {
            for (std::size_t i = 0; i < n; ++i) result.samples[i] = fp[i];
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                float sum = 0.0f;
                for (int16_t ch = 0; ch < nc; ++ch) sum += fp[i * nc + ch];
                result.samples[i] = sum / static_cast<float>(nc);
            }
        }
    }
    else
    {
        // PCM int16
        std::size_t n = data_bytes.size() / (2 * nc);
        result.samples.resize(n);
        const auto* sp = reinterpret_cast<const int16_t*>(data_bytes.data());
        if (nc <= 1) {
            for (std::size_t i = 0; i < n; ++i)
                result.samples[i] = static_cast<float>(sp[i]) / 32768.0f;
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                float sum = 0.0f;
                for (int16_t ch = 0; ch < nc; ++ch)
                    sum += static_cast<float>(sp[i * nc + ch]);
                result.samples[i] = sum / (32768.0f * static_cast<float>(nc));
            }
        }
    }
    result.num_samples = static_cast<int32_t>(result.samples.size());
    return result;
}

// Loaded image: float RGB pixels in HWC layout [height * width * 3], values in [0, 1].
struct LoadedImage {
    std::vector<float> pixels;  // [H * W * 3] float32 in [0, 1]
    int32_t height{0};
    int32_t width{0};

    bool empty() const { return pixels.empty(); }
};

// Load an image file (JPEG, PNG, BMP, etc.) and return float RGB pixels.
// Uses stb_image internally (linked via trtmc_core).
// Throws on file-not-found; returns empty LoadedImage on decode failure.
LoadedImage read_image(const std::string& path);

// Save float RGB pixels (HWC layout, values in [0, 1]) to a PNG file.
// Out-of-range values are clamped to [0, 1] before being quantised to uint8.
// Uses stb_image_write internally (linked via trtmc_core).
// Throws std::runtime_error on size mismatch or write failure.
void save_png(const std::string& path,
              const std::vector<float>& hwc_pixels,
              int width,
              int height);

// Convenience overload: save an ImageResult (first frame only) directly.
// For single-frame results (num_frames <= 1) writes result.pixels;
// for multi-frame results (video) writes only frame 0 — use the
// generate-video CLI command instead to dump every frame.
inline void save_png(const ImageResult& image, const std::string& path)
{
    const auto frame_pixels =
        static_cast<std::size_t>(image.height) * static_cast<std::size_t>(image.width) * 3U;
    if (image.pixels.size() < frame_pixels)
        throw std::runtime_error("save_png: ImageResult pixel buffer is smaller than H*W*3");
    if (image.pixels.size() == frame_pixels) {
        save_png(path, image.pixels, image.width, image.height);
        return;
    }
    // Multi-frame: only frame 0 is written.
    std::vector<float> frame0(image.pixels.begin(),
                              image.pixels.begin() + static_cast<std::ptrdiff_t>(frame_pixels));
    save_png(path, frame0, image.width, image.height);
}

// Legacy placeholder (prefer read_image).
inline std::vector<float> decode_image(const std::string& path, int& h, int& w)
{
    auto img = read_image(path);
    h = img.height;
    w = img.width;
    return std::move(img.pixels);
}

} // namespace trtmc::io
