/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/c_api.h"
#include "trtmc/runtime/family_loader.h"
#include "trtmc/task.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef TRTMC_VERSION_STRING
#define TRTMC_VERSION_STRING "0.1.0"
#endif

struct trtmc_task {
    std::unique_ptr<trtmc::ITask> value;
};

namespace {

thread_local char g_last_error[1024] = {};

void clear_error() noexcept {
    g_last_error[0] = '\0';
}

void set_error(const char* message) noexcept {
    std::snprintf(g_last_error, sizeof(g_last_error), "%s",
                  message != nullptr ? message : "unknown error");
}

trtmc_status_t fail(trtmc_status_t status, const char* message) noexcept {
    set_error(message);
    return status;
}

template <typename Function>
trtmc_status_t guarded(Function&& function) noexcept {
    clear_error();
    try {
        return function();
    } catch (const std::invalid_argument& error) {
        return fail(TRTMC_STATUS_INVALID_ARGUMENT, error.what());
    } catch (const std::bad_alloc&) {
        return fail(TRTMC_STATUS_OUT_OF_MEMORY, "out of memory");
    } catch (const std::exception& error) {
        return fail(TRTMC_STATUS_RUNTIME_ERROR, error.what());
    } catch (...) {
        return fail(TRTMC_STATUS_RUNTIME_ERROR, "unknown runtime error");
    }
}

void zero_text_result(trtmc_text_result_t* result) noexcept {
    *result = trtmc_text_result_t{};
}

void zero_image_result(trtmc_image_result_t* result) noexcept {
    *result = trtmc_image_result_t{};
}

bool checked_product(size_t left, size_t right, size_t* result) noexcept {
    if (right != 0 && left > std::numeric_limits<size_t>::max() / right)
        return false;
    *result = left * right;
    return true;
}

trtmc_status_t copy_text_result(const trtmc::TextResult& source,
                                trtmc_text_result_t* output) noexcept {
    trtmc_text_result_t result{};
    if (source.text.size() == std::numeric_limits<size_t>::max())
        return fail(TRTMC_STATUS_OUT_OF_MEMORY, "text result is too large");
    result.text = static_cast<char*>(std::malloc(source.text.size() + 1));
    if (result.text == nullptr)
        return fail(TRTMC_STATUS_OUT_OF_MEMORY, "unable to allocate text result");
    std::memcpy(result.text, source.text.data(), source.text.size());
    result.text[source.text.size()] = '\0';
    result.text_length = source.text.size();

    if (!source.token_ids.empty()) {
        size_t bytes = 0;
        if (!checked_product(source.token_ids.size(), sizeof(int32_t), &bytes)) {
            std::free(result.text);
            return fail(TRTMC_STATUS_OUT_OF_MEMORY, "token result is too large");
        }
        result.token_ids = static_cast<int32_t*>(std::malloc(bytes));
        if (result.token_ids == nullptr) {
            std::free(result.text);
            return fail(TRTMC_STATUS_OUT_OF_MEMORY, "unable to allocate token result");
        }
        std::memcpy(result.token_ids, source.token_ids.data(), bytes);
        result.token_count = source.token_ids.size();
    }

    result.setup_ms = source.setup_ms;
    result.prefill_ms = source.prefill_ms;
    result.decode_ms = source.decode_ms;
    *output = result;
    return TRTMC_STATUS_OK;
}

bool image_dimensions_are_positive(const trtmc::ImageResult& source) noexcept {
    return source.height > 0 && source.width > 0 && source.channels > 0 && source.num_frames > 0;
}

trtmc_status_t image_element_count(const trtmc::ImageResult& source, size_t* count) noexcept {
    if (!image_dimensions_are_positive(source))
        return fail(TRTMC_STATUS_RUNTIME_ERROR, "image result has invalid dimensions");
    *count = static_cast<size_t>(source.height);
    if (!checked_product(*count, static_cast<size_t>(source.width), count))
        return fail(TRTMC_STATUS_RUNTIME_ERROR, "image result dimensions overflow");
    if (!checked_product(*count, static_cast<size_t>(source.channels), count))
        return fail(TRTMC_STATUS_RUNTIME_ERROR, "image result dimensions overflow");
    if (!checked_product(*count, static_cast<size_t>(source.num_frames), count))
        return fail(TRTMC_STATUS_RUNTIME_ERROR, "image result dimensions overflow");
    if (*count != source.pixels.size())
        return fail(TRTMC_STATUS_RUNTIME_ERROR, "image result size does not match its dimensions");
    return TRTMC_STATUS_OK;
}

trtmc_status_t copy_image_result(const trtmc::ImageResult& source,
                                 trtmc_image_result_t* output) noexcept {
    size_t expected = 0;
    const trtmc_status_t count_status = image_element_count(source, &expected);
    if (count_status != TRTMC_STATUS_OK)
        return count_status;

    size_t bytes = 0;
    if (!checked_product(expected, sizeof(float), &bytes))
        return fail(TRTMC_STATUS_OUT_OF_MEMORY, "image result is too large");
    float* pixels = static_cast<float*>(std::malloc(bytes));
    if (pixels == nullptr)
        return fail(TRTMC_STATUS_OUT_OF_MEMORY, "unable to allocate image result");
    std::memcpy(pixels, source.pixels.data(), bytes);

    output->pixels = pixels;
    output->pixel_count = expected;
    output->height = source.height;
    output->width = source.width;
    output->channels = source.channels;
    output->num_frames = source.num_frames;
    return TRTMC_STATUS_OK;
}

trtmc_status_t prepare_image_outputs(trtmc_image_result_t* outputs, size_t count) noexcept {
    if (outputs == nullptr)
        return fail(TRTMC_STATUS_INVALID_ARGUMENT, "out_results must not be null");
    if (count == 0)
        return fail(TRTMC_STATUS_INVALID_ARGUMENT, "count is outside its valid range");
    if (count > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
        return fail(TRTMC_STATUS_INVALID_ARGUMENT, "count is outside its valid range");
    for (size_t index = 0; index < count; ++index)
        zero_image_result(&outputs[index]);
    return TRTMC_STATUS_OK;
}

trtmc_status_t validate_image_config(const trtmc_task_t* task, int32_t num_steps,
                                     float guidance_scale) noexcept {
    if (task == nullptr || task->value == nullptr)
        return fail(TRTMC_STATUS_INVALID_ARGUMENT, "task must not be null");
    if (num_steps <= 0)
        return fail(TRTMC_STATUS_INVALID_ARGUMENT, "num_steps must be positive");
    if (!std::isfinite(guidance_scale))
        return fail(TRTMC_STATUS_INVALID_ARGUMENT,
                    "guidance_scale must be finite and non-negative");
    if (guidance_scale < 0.0F)
        return fail(TRTMC_STATUS_INVALID_ARGUMENT,
                    "guidance_scale must be finite and non-negative");
    return TRTMC_STATUS_OK;
}

trtmc_status_t copy_prompts(const char* const* prompts, size_t count,
                            std::vector<std::string>* output) {
    if (prompts == nullptr)
        return fail(TRTMC_STATUS_INVALID_ARGUMENT, "prompts must not be null");
    output->reserve(count);
    for (size_t index = 0; index < count; ++index) {
        if (prompts[index] == nullptr)
            return fail(TRTMC_STATUS_INVALID_ARGUMENT, "every prompt must be non-null");
        output->emplace_back(prompts[index]);
    }
    return TRTMC_STATUS_OK;
}

trtmc_status_t copy_image_results(const std::vector<trtmc::ImageResult>& source,
                                  trtmc_image_result_t* output) noexcept {
    for (size_t index = 0; index < source.size(); ++index) {
        const trtmc_status_t status = copy_image_result(source[index], &output[index]);
        if (status != TRTMC_STATUS_OK) {
            for (size_t completed = 0; completed < index; ++completed)
                trtmc_image_result_free(&output[completed]);
            return status;
        }
    }
    return TRTMC_STATUS_OK;
}

} // namespace

extern "C" {

const char* trtmc_version(void) {
    return TRTMC_VERSION_STRING;
}

const char* trtmc_last_error(void) {
    return g_last_error;
}

trtmc_status_t trtmc_task_load(const char* bundle_path, const char* runtime_root,
                               trtmc_task_t** out_task) {
    return guarded([&]() {
        if (out_task == nullptr)
            return fail(TRTMC_STATUS_INVALID_ARGUMENT, "out_task must not be null");
        *out_task = nullptr;
        if (bundle_path == nullptr || bundle_path[0] == '\0')
            return fail(TRTMC_STATUS_INVALID_ARGUMENT, "bundle_path must not be null or empty");
        if (runtime_root == nullptr || runtime_root[0] == '\0')
            return fail(TRTMC_STATUS_INVALID_ARGUMENT, "runtime_root must not be null or empty");

        std::unique_ptr<trtmc::ITask> task = trtmc::load_task(bundle_path, runtime_root);
        if (task == nullptr)
            return fail(TRTMC_STATUS_RUNTIME_ERROR, "runtime returned a null task");
        auto handle = std::make_unique<trtmc_task>();
        handle->value = std::move(task);
        *out_task = handle.release();
        return TRTMC_STATUS_OK;
    });
}

void trtmc_task_destroy(trtmc_task_t* task) {
    delete task;
}

trtmc_status_t trtmc_task_name(const trtmc_task_t* task, const char** out_name) {
    return guarded([&]() {
        if (out_name == nullptr)
            return fail(TRTMC_STATUS_INVALID_ARGUMENT, "out_name must not be null");
        *out_name = nullptr;
        if (task == nullptr || task->value == nullptr)
            return fail(TRTMC_STATUS_INVALID_ARGUMENT, "task must not be null");
        const char* name = task->value->task();
        if (name == nullptr || name[0] == '\0')
            return fail(TRTMC_STATUS_RUNTIME_ERROR, "loaded task has no name");
        *out_name = name;
        return TRTMC_STATUS_OK;
    });
}

trtmc_status_t trtmc_text_generate(trtmc_task_t* task, const char* prompt, int32_t max_new_tokens,
                                   trtmc_text_result_t* out_result) {
    return guarded([&]() {
        if (out_result == nullptr)
            return fail(TRTMC_STATUS_INVALID_ARGUMENT, "out_result must not be null");
        zero_text_result(out_result);
        if (task == nullptr || task->value == nullptr)
            return fail(TRTMC_STATUS_INVALID_ARGUMENT, "task must not be null");
        if (prompt == nullptr)
            return fail(TRTMC_STATUS_INVALID_ARGUMENT, "prompt must not be null");
        if (max_new_tokens <= 0)
            return fail(TRTMC_STATUS_INVALID_ARGUMENT, "max_new_tokens must be positive");
        auto* text = dynamic_cast<trtmc::ITextGeneration*>(task->value.get());
        if (text == nullptr) {
            return fail(TRTMC_STATUS_WRONG_TASK, "task does not implement text_generation");
        }
        trtmc::TextGenerationConfig config;
        config.max_new_tokens = max_new_tokens;
        return copy_text_result(text->generate(prompt, config), out_result);
    });
}

void trtmc_text_result_free(trtmc_text_result_t* result) {
    if (result == nullptr)
        return;
    std::free(result->text);
    std::free(result->token_ids);
    zero_text_result(result);
}

trtmc_status_t trtmc_image_generate_batch(trtmc_task_t* task, const char* const* prompts,
                                          const uint32_t* seeds, size_t count, int32_t num_steps,
                                          float guidance_scale, trtmc_image_result_t* out_results) {
    return guarded([&]() {
        const trtmc_status_t output_status = prepare_image_outputs(out_results, count);
        if (output_status != TRTMC_STATUS_OK)
            return output_status;
        const trtmc_status_t config_status = validate_image_config(task, num_steps, guidance_scale);
        if (config_status != TRTMC_STATUS_OK)
            return config_status;
        if (seeds == nullptr)
            return fail(TRTMC_STATUS_INVALID_ARGUMENT, "seeds must not be null");

        std::vector<std::string> prompt_values;
        const trtmc_status_t prompt_status = copy_prompts(prompts, count, &prompt_values);
        if (prompt_status != TRTMC_STATUS_OK)
            return prompt_status;
        std::vector<std::uint32_t> seed_values(seeds, seeds + count);
        auto* image = dynamic_cast<trtmc::IImageBatchGeneration*>(task->value.get());
        if (image == nullptr) {
            return fail(TRTMC_STATUS_WRONG_TASK, "task does not implement image_generation_batch");
        }

        trtmc::ImageGenerationConfig config;
        config.num_steps = num_steps;
        config.guidance_scale = guidance_scale;
        const auto results = image->generate_image_batch(prompt_values, seed_values, config);
        if (results.size() != count)
            return fail(TRTMC_STATUS_RUNTIME_ERROR, "image task returned the wrong result count");
        return copy_image_results(results, out_results);
    });
}

void trtmc_image_result_free(trtmc_image_result_t* result) {
    if (result == nullptr)
        return;
    std::free(result->pixels);
    zero_image_result(result);
}

} // extern "C"
