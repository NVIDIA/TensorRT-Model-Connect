/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/core.hpp"
#include "trtmc/image.h"

namespace trtmc {

// Image inputs borrow pixels through the synchronous call. No decoding or
// model preprocessing is performed by the SDK wrapper.
struct ImageInput {
    ImageInput(Span<const float> pixels, std::uint32_t height, std::uint32_t width,
               std::uint32_t channels = 3)
        : wire{pixels.data(), float_bytes(pixels.size()), height, width,
               channels,      TRTMC_IMAGE_FLOAT32} {}
    ImageInput(Span<const std::uint8_t> pixels, std::uint32_t height, std::uint32_t width,
               std::uint32_t channels = 3) noexcept
        : wire{pixels.data(), pixels.size(), height, width, channels, TRTMC_IMAGE_UINT8} {}
    trtmc_image_input_v1 wire;

  private:
    static std::uint64_t float_bytes(std::size_t count) {
        if (count > std::numeric_limits<std::uint64_t>::max() / sizeof(float))
            throw std::overflow_error("image pixel byte size overflows uint64");
        return static_cast<std::uint64_t>(count) * sizeof(float);
    }
};

struct ImageMask {
    Span<const float> values;
    std::uint32_t height;
    std::uint32_t width;
};

struct TextToImageRequest {
    std::string prompt;
    Span<const float> initial_latents{}; // Borrowed through run; family-defined layout.
};
struct ImagesTextToImageEditRequest {
    std::vector<ImageInput> images;
    std::string prompt;
    Span<const float> initial_latents{};
};
struct MaskedImageTextToImageRequest {
    ImageInput source;
    ImageMask mask;
    std::string prompt;
};
struct BatchTextToImageItem {
    TextToImageRequest input;
    Config config;
};

struct ImageResultView {
    Span<const float> pixels;
    std::uint32_t height;
    std::uint32_t width;
    std::uint32_t channels;
    bool is_worker() const noexcept {
        return pixels.empty() && height == 0 && width == 0 && channels == 3;
    }
};

class ImageGenerationResult {
  public:
    ImageGenerationResult(const ImageGenerationResult&) = delete;
    ImageGenerationResult& operator=(const ImageGenerationResult&) = delete;
    ImageGenerationResult(ImageGenerationResult&& other) noexcept
        : owner_(std::move(other.owner_)), view_(std::exchange(other.view_, {})) {}
    ImageGenerationResult& operator=(ImageGenerationResult&& other) noexcept {
        if (this != &other) {
            owner_ = std::move(other.owner_);
            view_ = std::exchange(other.view_, {});
        }
        return *this;
    }
    Span<const float> pixels() const noexcept {
        return {view_.pixels, static_cast<std::size_t>(view_.pixel_count)};
    }
    std::uint32_t height() const noexcept { return view_.height; }
    std::uint32_t width() const noexcept { return view_.width; }
    std::uint32_t channels() const noexcept { return view_.channels; }
    bool is_worker() const noexcept {
        return view_.pixel_count == 0 && view_.height == 0 && view_.width == 0 &&
               view_.channels == 3;
    }

  private:
    friend class TextToImage;
    friend class ImagesTextToImageEdit;
    friend class MaskedImageTextToImage;
    ImageGenerationResult(std::shared_ptr<detail::ModelState> state, trtmc_result* result) noexcept
        : owner_(std::move(state), result) {}
    detail::ResultOwner owner_;
    trtmc_image_result_view_v1 view_{};
};

namespace detail {
template <class Table>
void validate_image_table(const trtmc_api_header* table) {
    if (!table || table->major != 1 || table->minor != 0 || table->byte_size < sizeof(Table))
        throw Error(TRTMC_VERSION_MISMATCH, "incompatible image Task table");
}
} // namespace detail

class TextToImage {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_TEXT_TO_IMAGE;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_image_table<trtmc_text_to_image_api_v1>(table);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    ImageGenerationResult run(const TextToImageRequest& request, const Config& config = {}) const {
        auto entries = config.c_entries();
        auto options = entries.view();
        trtmc_text_to_image_request_v1 input{
            detail::c_string(request.prompt),
            {request.initial_latents.data(), request.initial_latents.size()}};
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        auto status = api_->run(state_->handle, &input, &options, &raw, &error);
        ImageGenerationResult result(state_, raw);
        detail::check(state_->api, status, error);
        error = nullptr;
        status = api_->result_view(raw, &result.view_, &error);
        detail::check(state_->api, status, error);
        return result;
    }

  private:
    friend class Model;
    TextToImage(std::shared_ptr<detail::ModelState> state, const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_text_to_image_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_text_to_image_api_v1* api_;
};

class ImagesTextToImageEdit {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_IMAGES_TEXT_TO_IMAGE_EDIT;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_image_table<trtmc_images_text_to_image_edit_api_v1>(table);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    ImageGenerationResult run(const ImagesTextToImageEditRequest& request,
                              const Config& config = {}) const {
        std::vector<trtmc_image_input_v1> images;
        images.reserve(request.images.size());
        for (const auto& image : request.images)
            images.push_back(image.wire);
        auto entries = config.c_entries();
        auto options = entries.view();
        trtmc_images_text_to_image_edit_request_v1 input{
            images.data(),
            images.size(),
            detail::c_string(request.prompt),
            {request.initial_latents.data(), request.initial_latents.size()}};
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        auto status = api_->run(state_->handle, &input, &options, &raw, &error);
        ImageGenerationResult result(state_, raw);
        detail::check(state_->api, status, error);
        error = nullptr;
        status = api_->result_view(raw, &result.view_, &error);
        detail::check(state_->api, status, error);
        return result;
    }

  private:
    friend class Model;
    ImagesTextToImageEdit(std::shared_ptr<detail::ModelState> state,
                          const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_images_text_to_image_edit_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_images_text_to_image_edit_api_v1* api_;
};

class MaskedImageTextToImage {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_MASKED_IMAGE_TEXT_TO_IMAGE;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_image_table<trtmc_masked_image_text_to_image_api_v1>(table);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    ImageGenerationResult run(const MaskedImageTextToImageRequest& request,
                              const Config& config = {}) const {
        auto entries = config.c_entries();
        auto options = entries.view();
        trtmc_masked_image_text_to_image_request_v1 input{request.source.wire,
                                                          {request.mask.values.data(),
                                                           request.mask.values.size(),
                                                           request.mask.height, request.mask.width},
                                                          detail::c_string(request.prompt)};
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        auto status = api_->run(state_->handle, &input, &options, &raw, &error);
        ImageGenerationResult result(state_, raw);
        detail::check(state_->api, status, error);
        error = nullptr;
        status = api_->result_view(raw, &result.view_, &error);
        detail::check(state_->api, status, error);
        return result;
    }

  private:
    friend class Model;
    MaskedImageTextToImage(std::shared_ptr<detail::ModelState> state,
                           const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_masked_image_text_to_image_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_masked_image_text_to_image_api_v1* api_;
};

class ImageBatchResult {
  public:
    ImageBatchResult(const ImageBatchResult&) = delete;
    ImageBatchResult& operator=(const ImageBatchResult&) = delete;
    ImageBatchResult(ImageBatchResult&& other) noexcept
        : owner_(std::move(other.owner_)), api_(other.api_),
          count_(std::exchange(other.count_, 0)) {}
    ImageBatchResult& operator=(ImageBatchResult&& other) noexcept {
        if (this != &other) {
            owner_ = std::move(other.owner_);
            api_ = other.api_;
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }
    std::uint64_t size() const noexcept { return count_; }
    // The returned view borrows this batch result, not the model.
    ImageResultView at(std::uint64_t index) const {
        trtmc_image_result_view_v1 view{};
        trtmc_error* error = nullptr;
        const auto status = api_->result_item(owner_.get(), index, &view, &error);
        detail::check(owner_.api(), status, error);
        return {{view.pixels, static_cast<std::size_t>(view.pixel_count)},
                view.height,
                view.width,
                view.channels};
    }
    ImageResultView operator[](std::uint64_t index) const { return at(index); }

  private:
    friend class BatchTextToImage;
    ImageBatchResult(std::shared_ptr<detail::ModelState> state, trtmc_result* result,
                     const trtmc_batch_text_to_image_api_v1* api) noexcept
        : owner_(std::move(state), result), api_(api) {}
    detail::ResultOwner owner_;
    const trtmc_batch_text_to_image_api_v1* api_;
    std::uint64_t count_{0};
};

class BatchTextToImage {
  public:
    static constexpr std::string_view kTask = TRTMC_TASK_BATCH_TEXT_TO_IMAGE;
    static constexpr std::uint32_t kMajor = 1, kMinor = 0;
    static void validate_table(const trtmc_api_header* table) {
        detail::validate_image_table<trtmc_batch_text_to_image_api_v1>(table);
    }
    std::vector<ConfigField> config_fields() const {
        return detail::config_fields(state_, kTask, kMajor, kMinor);
    }
    ImageBatchResult run(const std::vector<BatchTextToImageItem>& requests) const {
        std::vector<Config::CEntries> configs;
        std::vector<trtmc_batch_text_to_image_item_v1> items;
        configs.reserve(requests.size());
        items.reserve(requests.size());
        for (const auto& item : requests) {
            configs.push_back(item.config.c_entries());
            items.push_back(
                {{detail::c_string(item.input.prompt),
                  {item.input.initial_latents.data(), item.input.initial_latents.size()}},
                 configs.back().view()});
        }
        trtmc_batch_text_to_image_request_v1 input{items.data(), items.size()};
        trtmc_result* raw = nullptr;
        trtmc_error* error = nullptr;
        auto status = api_->run_batch(state_->handle, &input, &raw, &error);
        ImageBatchResult result(state_, raw, api_);
        detail::check(state_->api, status, error);
        error = nullptr;
        status = api_->result_count(raw, &result.count_, &error);
        detail::check(state_->api, status, error);
        return result;
    }

  private:
    friend class Model;
    BatchTextToImage(std::shared_ptr<detail::ModelState> state,
                     const trtmc_api_header* table) noexcept
        : state_(std::move(state)),
          api_(reinterpret_cast<const trtmc_batch_text_to_image_api_v1*>(table)) {}
    std::shared_ptr<detail::ModelState> state_;
    const trtmc_batch_text_to_image_api_v1* api_;
};

} // namespace trtmc
