/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2_hoi/jpeg_decoder.h"

#include <algorithm>
#include <atomic>
#include <csetjmp>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <future>
#include <jpeglib.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::sam2_hoi {
namespace {

struct JpegErrorManager {
    jpeg_error_mgr base;
    std::jmp_buf jump;
};

void jump_on_jpeg_error(j_common_ptr common) {
    auto* error = reinterpret_cast<JpegErrorManager*>(common->err);
    std::longjmp(error->jump, 1);
}

bool valid_rgb_output(const jpeg_decompress_struct& decoder) {
    return decoder.output_width != 0 && decoder.output_height != 0 &&
           decoder.output_components == 3 &&
           decoder.output_width <= static_cast<JDIMENSION>(std::numeric_limits<int32_t>::max()) &&
           decoder.output_height <= static_cast<JDIMENSION>(std::numeric_limits<int32_t>::max());
}

bool rgb_buffer_layout(std::size_t width, std::size_t height, std::size_t& row_stride,
                       std::size_t& value_count) {
    if (width > std::numeric_limits<std::size_t>::max() / 3U)
        return false;
    row_stride = width * 3U;
    if (height > std::numeric_limits<std::size_t>::max() / row_stride)
        return false;
    value_count = height * row_stride;
    return true;
}

void read_rgb_scanlines(jpeg_decompress_struct& decoder, std::uint8_t* pixels,
                        std::size_t row_stride, jpeg_error_mgr& error) {
    while (decoder.output_scanline < decoder.output_height) {
        JSAMPROW row = pixels + static_cast<std::size_t>(decoder.output_scanline) * row_stride;
        if (jpeg_read_scanlines(&decoder, &row, 1) != 1)
            error.error_exit(reinterpret_cast<j_common_ptr>(&decoder));
    }
}

Sam2HoiVideoFrame make_video_frame(std::unique_ptr<std::uint8_t, decltype(&std::free)> pixels,
                                   std::size_t width, std::size_t height, std::size_t value_count) {
    Sam2HoiVideoFrame frame;
    frame.height = static_cast<int32_t>(height);
    frame.width = static_cast<int32_t>(width);
    frame.pixels.resize(value_count);
    for (std::size_t index = 0; index < value_count; ++index)
        frame.pixels[index] = static_cast<float>(pixels.get()[index]) / 255.0F;
    return frame;
}

void decode_worker(const std::vector<std::string>& paths,
                   const std::function<Sam2HoiVideoFrame(const std::string&)>& decoder,
                   std::vector<Sam2HoiVideoFrame>& frames,
                   std::vector<std::exception_ptr>& failures,
                   std::atomic<std::size_t>& next_index) {
    while (true) {
        const std::size_t index = next_index.fetch_add(1U, std::memory_order_relaxed);
        if (index >= paths.size())
            return;
        try {
            auto frame = decoder(paths[index]);
            if (frame.empty()) {
                throw std::runtime_error("SAM2 HOI failed to decode JPEG frame " +
                                         std::to_string(index) + ": " + paths[index]);
            }
            frames[index] = std::move(frame);
        } catch (...) {
            failures[index] = std::current_exception();
        }
    }
}

template <typename Worker>
std::vector<std::future<void>> launch_decode_workers(std::size_t worker_count,
                                                     const Worker& worker) {
    std::vector<std::future<void>> workers;
    workers.reserve(worker_count);
    for (std::size_t index = 0; index < worker_count; ++index) {
        try {
            workers.push_back(std::async(std::launch::async, worker));
        } catch (...) {
            // An already-launched worker drains the shared queue. If the first
            // launch fails, the caller runs the same queue on its thread.
            break;
        }
    }
    return workers;
}

std::exception_ptr join_decode_workers(std::vector<std::future<void>>& workers) {
    std::exception_ptr worker_failure;
    for (auto& future : workers) {
        try {
            future.get();
        } catch (...) {
            if (worker_failure == nullptr)
                worker_failure = std::current_exception();
        }
    }
    return worker_failure;
}

void rethrow_decode_failure(const std::vector<std::exception_ptr>& failures,
                            const std::exception_ptr& worker_failure) {
    for (const auto& failure : failures) {
        if (failure != nullptr)
            std::rethrow_exception(failure);
    }
    if (worker_failure != nullptr)
        std::rethrow_exception(worker_failure);
}

} // namespace

Sam2HoiVideoFrame decode_jpeg_pillow_rgb(const std::string& path) {
    std::FILE* input = std::fopen(path.c_str(), "rb");
    if (input == nullptr)
        return {};

    jpeg_decompress_struct decoder{};
    JpegErrorManager error{};
    decoder.err = jpeg_std_error(&error.base);
    error.base.error_exit = jump_on_jpeg_error;

    volatile bool decoder_created = false;
    // The pointer itself must be volatile: a non-volatile automatic changed
    // after setjmp has an indeterminate value when libjpeg longjmps here.
    std::uint8_t* volatile raw_pixels = nullptr;
    if (setjmp(error.jump) != 0) {
        std::free(const_cast<std::uint8_t*>(raw_pixels));
        if (decoder_created)
            jpeg_destroy_decompress(&decoder);
        std::fclose(input);
        return {};
    }

    jpeg_create_decompress(&decoder);
    decoder_created = true;
    jpeg_stdio_src(&decoder, input);
    jpeg_read_header(&decoder, TRUE);

    // Pillow's RGB JPEG decode uses libjpeg's accurate integer IDCT and fancy
    // chroma upsampling defaults. State them explicitly so the pixel contract
    // does not depend on another caller's libjpeg defaults.
    decoder.out_color_space = JCS_RGB;
    decoder.dct_method = JDCT_ISLOW;
    decoder.do_fancy_upsampling = TRUE;
    jpeg_start_decompress(&decoder);

    if (!valid_rgb_output(decoder)) {
        jpeg_destroy_decompress(&decoder);
        std::fclose(input);
        return {};
    }

    const std::size_t width = decoder.output_width;
    const std::size_t height = decoder.output_height;
    std::size_t row_stride = 0;
    std::size_t value_count = 0;
    if (!rgb_buffer_layout(width, height, row_stride, value_count)) {
        jpeg_destroy_decompress(&decoder);
        std::fclose(input);
        return {};
    }
    raw_pixels = static_cast<std::uint8_t*>(std::malloc(value_count));
    if (raw_pixels == nullptr) {
        jpeg_destroy_decompress(&decoder);
        std::fclose(input);
        return {};
    }

    read_rgb_scanlines(decoder, const_cast<std::uint8_t*>(raw_pixels), row_stride, error.base);
    jpeg_finish_decompress(&decoder);
    jpeg_destroy_decompress(&decoder);
    decoder_created = false;
    std::fclose(input);

    auto* decoded_pixels = const_cast<std::uint8_t*>(raw_pixels);
    raw_pixels = nullptr;
    std::unique_ptr<std::uint8_t, decltype(&std::free)> owned_pixels(decoded_pixels, &std::free);

    return make_video_frame(std::move(owned_pixels), width, height, value_count);
}

std::vector<Sam2HoiVideoFrame>
decode_jpeg_paths_bounded(const std::vector<std::string>& paths, std::size_t max_concurrency,
                          const std::function<Sam2HoiVideoFrame(const std::string&)>& decoder) {
    if (max_concurrency == 0U)
        throw std::invalid_argument("SAM2 HOI JPEG decode concurrency must be positive");
    if (!decoder)
        throw std::invalid_argument("SAM2 HOI JPEG decoder callback is empty");

    std::vector<Sam2HoiVideoFrame> frames(paths.size());
    if (paths.empty())
        return frames;

    std::vector<std::exception_ptr> failures(paths.size());
    std::atomic<std::size_t> next_index{0U};
    const auto worker = [&]() { decode_worker(paths, decoder, frames, failures, next_index); };

    const std::size_t worker_count = std::min(max_concurrency, paths.size());
    auto workers = launch_decode_workers(worker_count, worker);
    if (workers.empty())
        worker();
    const auto worker_failure = join_decode_workers(workers);
    rethrow_decode_failure(failures, worker_failure);
    return frames;
}

std::vector<Sam2HoiVideoFrame> decode_jpeg_pillow_rgb_batch(const std::vector<std::string>& paths) {
    return decode_jpeg_paths_bounded(paths, kMaxConcurrentJpegDecodes, decode_jpeg_pillow_rgb);
}

} // namespace trtmc::sam2_hoi
