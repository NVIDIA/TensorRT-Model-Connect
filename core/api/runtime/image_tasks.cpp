/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "api_internal.h"
#include "trtmc/image.h"
#include "trtmc/internal/image.h"

#include <mutex>
#include <vector>

namespace trtmc::api {
namespace {

std::size_t image_elements(std::uint32_t height, std::uint32_t width, std::uint32_t channels) {
    require(height != 0 && width != 0 && channels != 0, "image dimensions must be positive");
    const auto rows = checked_size(height, width);
    const auto pixels = rows * width;
    checked_size(pixels, channels);
    return pixels * channels;
}

} // namespace

internal::ImageView image_input(const trtmc_image_input_v1& input) {
    const auto count = image_elements(input.height, input.width, input.channels);
    const bool floating = input.format == TRTMC_IMAGE_FLOAT32;
    require(floating || input.format == TRTMC_IMAGE_UINT8, "unknown image pixel format");
    const auto element_size = floating ? sizeof(float) : sizeof(std::uint8_t);
    checked_size(count, element_size);
    const auto bytes = count * element_size;
    require(input.data != nullptr && input.byte_size == bytes,
            "image byte size must match its contiguous HWC shape");
    if (floating)
        require(reinterpret_cast<std::uintptr_t>(input.data) % alignof(float) == 0,
                "float image data is not aligned");
    return {
        input.data,     bytes,
        input.height,   input.width,
        input.channels, floating ? internal::ImageFormat::Float32 : internal::ImageFormat::UInt8};
}

namespace {

trtmc_image_result_view_v1 image_output(const internal::ImageResult& image) {
    if (internal::is_worker_completion(image))
        return {nullptr, 0, 0, 0, 3};
    if (image.height <= 0 || image.width <= 0 || image.channels <= 0 || image.num_frames != 1)
        throw ApiFailure{TRTMC_INTERNAL_ERROR, "family returned invalid image dimensions"};
    std::size_t count;
    try {
        count = image_elements(image.height, image.width, image.channels);
    } catch (const ApiFailure&) {
        throw ApiFailure{TRTMC_INTERNAL_ERROR, "family image dimensions overflow host storage"};
    }
    if (image.pixels.size() != count)
        throw ApiFailure{TRTMC_INTERNAL_ERROR, "family image pixels do not match its HWC shape"};
    return {image.pixels.data(), count, static_cast<std::uint32_t>(image.height),
            static_cast<std::uint32_t>(image.width), static_cast<std::uint32_t>(image.channels)};
}

struct ImageStorage final : ResultStorage {
    explicit ImageStorage(internal::ImageResult value)
        : image(std::move(value)), view(image_output(image)) {}
    internal::ImageResult image;
    trtmc_image_result_view_v1 view;
};

struct ImageBatchStorage final : ResultStorage {
    explicit ImageBatchStorage(std::vector<internal::ImageResult> value)
        : images(std::move(value)) {
        views.reserve(images.size());
        for (const auto& image : images)
            views.push_back(image_output(image));
    }
    std::vector<internal::ImageResult> images;
    std::vector<trtmc_image_result_view_v1> views;
};

trtmc_status TRTMC_CALL image_result_view(const trtmc_result* result,
                                          trtmc_image_result_view_v1* out,
                                          trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(out != nullptr, "image view output is null");
        *out = require_result<ImageStorage>(result).view;
    });
}

trtmc_status TRTMC_CALL generate_image(trtmc_model* model,
                                       const trtmc_text_to_image_request_v1* request,
                                       const trtmc_config_view_v1* config, trtmc_result** out,
                                       trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(request != nullptr && out != nullptr, "image request or result output is null");
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& family =
            require_interface<internal::ITextToImage>(model, internal::ITextToImage::kTask);
        const ConvertedConfig options(config);
        validate_task_config(model_owner(model), internal::contract_key<internal::ITextToImage>(),
                             options.view());
        *out = make_result<ImageStorage>(
            family.run({string_view(request->prompt),
                        checked_span(request->initial_latents.data, request->initial_latents.size)},
                       options.view()));
    });
}

trtmc_status TRTMC_CALL edit_images(trtmc_model* model,
                                    const trtmc_images_text_to_image_edit_request_v1* request,
                                    const trtmc_config_view_v1* config, trtmc_result** out,
                                    trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(request != nullptr && out != nullptr, "edit request or result output is null");
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& family = require_interface<internal::IImagesTextToImageEdit>(
            model, internal::IImagesTextToImageEdit::kTask);
        const auto source = checked_span(request->images, request->image_count);
        require(!source.empty(), "image edit requires at least one source image");
        std::vector<internal::ImageView> images;
        images.reserve(source.size());
        for (const auto& image : source)
            images.push_back(image_input(image));
        const ConvertedConfig options(config);
        validate_task_config(model_owner(model),
                             internal::contract_key<internal::IImagesTextToImageEdit>(),
                             options.view());
        const internal::ImagesTextToImageEditRequest input{
            {images.data(), images.size()},
            string_view(request->prompt),
            checked_span(request->initial_latents.data, request->initial_latents.size)};
        *out = make_result<ImageStorage>(family.run(input, options.view()));
    });
}

trtmc_status TRTMC_CALL edit_masked_image(
    trtmc_model* model, const trtmc_masked_image_text_to_image_request_v1* request,
    const trtmc_config_view_v1* config, trtmc_result** out, trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(request != nullptr && out != nullptr,
                "masked edit request or result output is null");
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& family = require_interface<internal::IMaskedImageTextToImage>(
            model, internal::IMaskedImageTextToImage::kTask);
        const auto source = image_input(request->source);
        const auto mask = checked_span(request->mask.data, request->mask.count);
        require(request->mask.height == source.height && request->mask.width == source.width &&
                    mask.size() == image_elements(source.height, source.width, 1),
                "edit mask must align with the source image");
        const internal::MaskedImageTextToImageRequest input{
            source, {mask, source.height, source.width}, string_view(request->prompt)};
        const ConvertedConfig options(config);
        validate_task_config(model_owner(model),
                             internal::contract_key<internal::IMaskedImageTextToImage>(),
                             options.view());
        *out = make_result<ImageStorage>(family.run(input, options.view()));
    });
}

trtmc_status TRTMC_CALL generate_batch(trtmc_model* model,
                                       const trtmc_batch_text_to_image_request_v1* request,
                                       trtmc_result** out, trtmc_error** error) noexcept {
    if (out)
        *out = nullptr;
    return guarded(error, [&] {
        require(request != nullptr && out != nullptr, "batch request or result output is null");
        std::lock_guard<std::mutex> lock(model_mutex(model));
        auto& family = require_interface<internal::IBatchTextToImage>(
            model, internal::IBatchTextToImage::kTask);
        const auto source = checked_span(request->items, request->count);
        require(!source.empty(), "image batch must contain at least one item");
        std::vector<ConvertedConfig> configs;
        std::vector<internal::BatchTextToImageItem> items;
        configs.reserve(source.size());
        items.reserve(source.size());
        for (const auto& item : source) {
            configs.emplace_back(&item.config);

            items.push_back(
                {{string_view(item.input.prompt),
                  checked_span(item.input.initial_latents.data, item.input.initial_latents.size)},
                 configs.back().view()});
        }
        validate_batch_configs(model_owner(model),
                               internal::contract_key<internal::IBatchTextToImage>(), configs);
        auto results = family.run_batch({{items.data(), items.size()}});
        if (results.size() != items.size())
            throw ApiFailure{TRTMC_INTERNAL_ERROR,
                             "family batch result count does not match input"};
        *out = make_result<ImageBatchStorage>(std::move(results));
    });
}

trtmc_status TRTMC_CALL batch_count(const trtmc_result* result, std::uint64_t* out,
                                    trtmc_error** error) noexcept {
    if (out)
        *out = 0;
    return guarded(error, [&] {
        require(out != nullptr, "batch count output is null");
        *out = require_result<ImageBatchStorage>(result).views.size();
    });
}

trtmc_status TRTMC_CALL batch_item(const trtmc_result* result, std::uint64_t index,
                                   trtmc_image_result_view_v1* out, trtmc_error** error) noexcept {
    if (out)
        *out = {};
    return guarded(error, [&] {
        require(out != nullptr, "batch image output is null");
        const auto& batch = require_result<ImageBatchStorage>(result);
        require(index < batch.views.size(), "batch image index is out of range");
        *out = batch.views[static_cast<std::size_t>(index)];
    });
}

const trtmc_text_to_image_api_v1 generate_api{
    {1, 0, sizeof(trtmc_text_to_image_api_v1)}, generate_image, image_result_view};
const trtmc_images_text_to_image_edit_api_v1 edit_api{
    {1, 0, sizeof(trtmc_images_text_to_image_edit_api_v1)}, edit_images, image_result_view};
const trtmc_masked_image_text_to_image_api_v1 masked_api{
    {1, 0, sizeof(trtmc_masked_image_text_to_image_api_v1)}, edit_masked_image, image_result_view};
const trtmc_batch_text_to_image_api_v1 batch_api{
    {1, 0, sizeof(trtmc_batch_text_to_image_api_v1)}, generate_batch, batch_count, batch_item};

const TaskBinding bindings[] = {
    {internal::ITextToImage::kTask, 1, 0, &generate_api.header},
    {internal::IImagesTextToImageEdit::kTask, 1, 0, &edit_api.header},
    {internal::IMaskedImageTextToImage::kTask, 1, 0, &masked_api.header},
    {internal::IBatchTextToImage::kTask, 1, 0, &batch_api.header},
};

} // namespace

Span<const TaskBinding> image_task_bindings() noexcept {
    return bindings;
}

} // namespace trtmc::api
