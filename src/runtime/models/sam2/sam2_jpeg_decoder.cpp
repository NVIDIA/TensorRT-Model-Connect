/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2/sam2_jpeg_decoder.h"

#include <cerrno>
#include <climits>
#include <csetjmp>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>
#include <utility>
#include <vector>

extern "C" {
#include <jpeglib.h>
}

namespace trtmc::sam2 {

namespace {

constexpr std::size_t kRgbChannels = 3U;
constexpr std::uint64_t kMaximumEncodedBytes = std::uint64_t{64} * 1024U * 1024U;
constexpr std::size_t kExpectedRgbBytes = static_cast<std::size_t>(kOriginalImageHeight) *
                                          static_cast<std::size_t>(kOriginalImageWidth) *
                                          kRgbChannels;
static_assert(kOriginalImageHeight > 0 && kOriginalImageWidth > 0);
static_assert(static_cast<std::uint64_t>(kExpectedRgbBytes) ==
              static_cast<std::uint64_t>(kOriginalImageHeight) *
                  static_cast<std::uint64_t>(kOriginalImageWidth) * kRgbChannels);

class FileDescriptor {
  public:
    explicit FileDescriptor(int descriptor) : descriptor_(descriptor) {}
    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;
    ~FileDescriptor() {
        if (descriptor_ >= 0)
            (void)::close(descriptor_);
    }

    [[nodiscard]] int get() const noexcept { return descriptor_; }

  private:
    int descriptor_{-1};
};

struct JpegErrorManager {
    jpeg_error_mgr base;
    std::jmp_buf jump;
    char message[JMSG_LENGTH_MAX];
};

struct DecodeContext {
    jpeg_decompress_struct decoder;
    JpegErrorManager error;
    bool created;
};

struct DecodeFailure {
    char message[JMSG_LENGTH_MAX];
};

void copyFailure(DecodeFailure& failure, const char* message) noexcept {
    if (message == nullptr || *message == '\0')
        message = "unknown libjpeg error";
    (void)std::snprintf(failure.message, sizeof(failure.message), "%s", message);
}

[[noreturn]] void jpegErrorExit(j_common_ptr common) {
    auto* error = reinterpret_cast<JpegErrorManager*>(common->err);
    (*common->err->format_message)(common, error->message);
    std::longjmp(error->jump, 1);
}

void jpegEmitMessage(j_common_ptr common, int message_level) {
    // A negative level is a recoverable libjpeg warning. The qualification
    // contract is strict, so malformed/truncated inputs are rejected instead
    // of being partially decoded or written to stderr. Non-negative levels
    // are optional trace messages and remain silent.
    if (message_level < 0)
        jpegErrorExit(common);
}

void destroyDecodeContext(DecodeContext* context) noexcept {
    if (context == nullptr)
        return;
    if (context->created) {
        // Clear the flag first so an unexpected libjpeg failure during destroy
        // cannot recurse through the setjmp error path.
        context->created = false;
        jpeg_destroy_decompress(&context->decoder);
    }
    std::free(context);
}

bool failDecode(DecodeContext* context, DecodeFailure& failure, const char* message) noexcept {
    copyFailure(failure, message);
    destroyDecodeContext(context);
    return false;
}

// Keep the setjmp/longjmp boundary in a C-like leaf: no automatic object with a
// non-trivial destructor is created after setjmp. The libjpeg structs live in
// dynamically allocated storage, so their values remain usable after longjmp.
bool decodeInto(const std::uint8_t* encoded, std::size_t encoded_bytes, std::uint8_t* output,
                std::size_t output_bytes, DecodeFailure& failure) noexcept {
    auto* context = static_cast<DecodeContext*>(std::calloc(1U, sizeof(DecodeContext)));
    if (context == nullptr) {
        copyFailure(failure, "unable to allocate libjpeg decode context");
        return false;
    }
    context->decoder.err = jpeg_std_error(&context->error.base);
    context->error.base.error_exit = jpegErrorExit;
    context->error.base.emit_message = jpegEmitMessage;

    if (setjmp(context->error.jump) != 0) {
        copyFailure(failure, context->error.message);
        destroyDecodeContext(context);
        return false;
    }

    jpeg_create_decompress(&context->decoder);
    context->created = true;
    jpeg_mem_src(&context->decoder, encoded, static_cast<unsigned long>(encoded_bytes));
    if (jpeg_read_header(&context->decoder, TRUE) != JPEG_HEADER_OK)
        return failDecode(context, failure, "libjpeg did not return a complete header");

    if (context->decoder.image_height != static_cast<JDIMENSION>(kOriginalImageHeight) ||
        context->decoder.image_width != static_cast<JDIMENSION>(kOriginalImageWidth) ||
        context->decoder.data_precision != 8 || context->decoder.num_components != 3) {
        return failDecode(context, failure,
                          "JPEG must be 1088x1280 with 8-bit, three-component samples");
    }

    context->decoder.out_color_space = JCS_RGB;
    context->decoder.dct_method = JDCT_ISLOW;
    context->decoder.do_fancy_upsampling = TRUE;
    context->decoder.do_block_smoothing = TRUE;
    if (jpeg_start_decompress(&context->decoder) == FALSE)
        return failDecode(context, failure, "libjpeg suspended while starting decompression");

    if (context->decoder.output_height != static_cast<JDIMENSION>(kOriginalImageHeight) ||
        context->decoder.output_width != static_cast<JDIMENSION>(kOriginalImageWidth) ||
        context->decoder.output_components != static_cast<int>(kRgbChannels) ||
        context->decoder.out_color_space != JCS_RGB) {
        return failDecode(context, failure, "libjpeg output violated the SAM2 RGB contract");
    }

    const auto width = static_cast<std::size_t>(context->decoder.output_width);
    const auto height = static_cast<std::size_t>(context->decoder.output_height);
    const auto components = static_cast<std::size_t>(context->decoder.output_components);
    if (width == 0U || height == 0U || components != kRgbChannels ||
        width > std::numeric_limits<std::size_t>::max() / components ||
        width * components > std::numeric_limits<std::size_t>::max() / height ||
        width * components * height != output_bytes) {
        return failDecode(context, failure, "libjpeg output byte count overflowed or drifted");
    }
    const std::size_t stride = width * components;

    while (context->decoder.output_scanline < context->decoder.output_height) {
        const std::size_t row_index = static_cast<std::size_t>(context->decoder.output_scanline);
        if (row_index >= height)
            return failDecode(context, failure, "libjpeg produced an out-of-range scanline");
        JSAMPROW row = output + row_index * stride;
        if (jpeg_read_scanlines(&context->decoder, &row, 1U) != 1U)
            return failDecode(context, failure, "libjpeg returned a short scanline read");
    }
    if (jpeg_finish_decompress(&context->decoder) == FALSE)
        return failDecode(context, failure, "libjpeg suspended while finishing decompression");

    destroyDecodeContext(context);
    return true;
}

std::vector<std::uint8_t> readOwnedRegularFile(const std::filesystem::path& path) {
    const auto& native_path = path.native();
    if (native_path.empty() || native_path.find('\0') != std::string::npos)
        throw std::invalid_argument("SAM2 JPEG path must be non-empty and contain no NUL byte");

    int flags = O_RDONLY;
#ifdef O_CLOEXEC
    flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
    flags |= O_NOFOLLOW;
#endif
    const int raw_descriptor = ::open(native_path.c_str(), flags);
    if (raw_descriptor < 0)
        throw std::system_error(errno, std::generic_category(),
                                "unable to open SAM2 JPEG regular file");
    FileDescriptor descriptor(raw_descriptor);

    struct stat initial{};
    if (::fstat(descriptor.get(), &initial) != 0)
        throw std::system_error(errno, std::generic_category(), "unable to stat SAM2 JPEG");
    if (!S_ISREG(initial.st_mode))
        throw std::invalid_argument("SAM2 JPEG input must be a regular file");
    if (initial.st_size <= 0)
        throw std::invalid_argument("SAM2 JPEG input must not be empty");
    const auto file_bytes = static_cast<std::uint64_t>(initial.st_size);
    if (file_bytes > kMaximumEncodedBytes ||
        file_bytes > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::length_error("SAM2 JPEG input exceeds the 64 MiB safety limit");
    }

    std::vector<std::uint8_t> encoded(static_cast<std::size_t>(file_bytes));
    std::size_t offset = 0U;
    while (offset < encoded.size()) {
        const ssize_t count = ::pread(descriptor.get(), encoded.data() + offset,
                                      encoded.size() - offset, static_cast<off_t>(offset));
        if (count < 0) {
            if (errno == EINTR)
                continue;
            throw std::system_error(errno, std::generic_category(), "unable to read SAM2 JPEG");
        }
        if (count == 0)
            throw std::runtime_error("SAM2 JPEG regular-file snapshot was truncated");
        offset += static_cast<std::size_t>(count);
    }

    struct stat final_status{};
    if (::fstat(descriptor.get(), &final_status) != 0)
        throw std::system_error(errno, std::generic_category(), "unable to restat SAM2 JPEG");
    if (!S_ISREG(final_status.st_mode) || final_status.st_dev != initial.st_dev ||
        final_status.st_ino != initial.st_ino || final_status.st_size != initial.st_size) {
        throw std::runtime_error("SAM2 JPEG regular file changed during its owned snapshot");
    }
    std::uint8_t extra = 0U;
    ssize_t trailing = 0;
    do {
        trailing = ::pread(descriptor.get(), &extra, 1U, final_status.st_size);
    } while (trailing < 0 && errno == EINTR);
    if (trailing < 0)
        throw std::system_error(errno, std::generic_category(), "unable to verify SAM2 JPEG EOF");
    if (trailing != 0)
        throw std::runtime_error("SAM2 JPEG regular file grew during its owned snapshot");
    return encoded;
}

} // namespace

DecodedSam2Jpeg decodeSam2JpegBytes(std::vector<std::uint8_t> encoded_jpeg) {
    if (encoded_jpeg.empty())
        throw std::invalid_argument("SAM2 JPEG byte input must not be empty");
    if (encoded_jpeg.size() > kMaximumEncodedBytes)
        throw std::length_error("SAM2 JPEG byte input exceeds the 64 MiB safety limit");
    if constexpr (sizeof(std::size_t) > sizeof(unsigned long)) {
        if (encoded_jpeg.size() > static_cast<std::size_t>(ULONG_MAX))
            throw std::overflow_error("SAM2 JPEG byte count exceeds the libjpeg API range");
    }

    DecodedSam2Jpeg decoded;
    decoded.rgb_hwc.resize(kExpectedRgbBytes);
    DecodeFailure failure{};
    if (!decodeInto(encoded_jpeg.data(), encoded_jpeg.size(), decoded.rgb_hwc.data(),
                    decoded.rgb_hwc.size(), failure)) {
        throw std::runtime_error("SAM2 JPEG decode failed: " + std::string(failure.message));
    }
    return decoded;
}

DecodedSam2Jpeg decodeSam2JpegFile(const std::filesystem::path& path) {
    return decodeSam2JpegBytes(readOwnedRegularFile(path));
}

} // namespace trtmc::sam2
