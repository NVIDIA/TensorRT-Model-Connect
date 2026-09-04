/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "trtmc/bundle.h"
#include "trtmc/pipeline.h"
#include "trtmc/runtime/pipeline_factory.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#ifndef TRTMC_VERSION_STRING
#define TRTMC_VERSION_STRING "0.1.0"
#endif

namespace {

thread_local std::string g_last_error;

struct PipelineCreateArgs {
    std::string hf_python;
    std::string runtime_cache;
    bool cuda_graphs{false};
};

void set_last_error(const std::string& msg) {
    g_last_error = msg;
}

void clear_last_error() {
    g_last_error.clear();
}

PipelineCreateArgs parse_pipeline_options(const TrtmcPipelineOptions* options) {
    PipelineCreateArgs args;
    if (options != nullptr && options->hf_python != nullptr)
        args.hf_python = options->hf_python;
    if (options != nullptr && options->runtime_cache != nullptr)
        args.runtime_cache = options->runtime_cache;
    args.cuda_graphs = (options != nullptr && options->cuda_graphs != 0);
    return args;
}

// Convert a single C++ ImageResult into the C-ABI POD form. Allocates the
// pixel buffer with std::malloc so callers can free it from C (or via
// trtmc_image_result_free). Returns false (and leaves `out` zero-initialised)
// if the allocation fails.
bool to_c_image_result(const trtmc::ImageResult& src, trtmc_image_result_t* out) {
    std::memset(out, 0, sizeof(*out));
    const std::size_t count = src.pixels.size();
    if (count == 0) {
        out->height = src.height;
        out->width = src.width;
        out->channels = src.channels;
        out->num_frames = src.num_frames;
        out->num_pixels = 0;
        out->pixels = nullptr;
        return true;
    }
    auto* buf = static_cast<float*>(std::malloc(count * sizeof(float)));
    if (buf == nullptr) {
        return false;
    }
    std::memcpy(buf, src.pixels.data(), count * sizeof(float));
    out->pixels = buf;
    out->height = src.height;
    out->width = src.width;
    out->channels = src.channels;
    out->num_frames = src.num_frames;
    out->num_pixels = static_cast<std::uint64_t>(count);
    return true;
}

} // namespace

extern "C" {

trtmc::IPipeline* trtmc_create_pipeline(const char* bundle_path, int flags) {
    (void)flags;
    TrtmcPipelineOptions opts{};
    opts.max_new_tokens = 0;
    opts.hf_python = nullptr;
    opts.image_path = nullptr;
    opts.runtime_cache = nullptr;
    opts.cuda_graphs = 0;
    return trtmc_create_pipeline_ex(bundle_path, &opts);
}

trtmc::IPipeline* trtmc_create_pipeline_ex(const char* bundle_path,
                                           const TrtmcPipelineOptions* options) {
    clear_last_error();

    if (bundle_path == nullptr || bundle_path[0] == '\0') {
        set_last_error("bundle_path must not be null or empty");
        return nullptr;
    }

    try {
        const std::string path(bundle_path);
        if (!trtmc::IsBundle(path)) {
            set_last_error("Not a valid .bundle artifact: " + path);
            return nullptr;
        }

        const PipelineCreateArgs args = parse_pipeline_options(options);

        auto t0 = std::chrono::steady_clock::now();

        auto pipeline = trtmc::PipelineFactory::from_bundle(path, args.hf_python,
                                                            args.runtime_cache, args.cuda_graphs);

        auto t1 = std::chrono::steady_clock::now();
        std::cerr << "[trtmc] Runtime ready (strategy=" << pipeline->pipeline_type() << ") ["
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                  << " ms]" << std::endl;

        return pipeline.release();
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return nullptr;
    } catch (...) {
        set_last_error("Unknown error creating pipeline");
        return nullptr;
    }
}

const char* trtmc_last_error(void) {
    return g_last_error.c_str();
}

const char* trtmc_version(void) {
    return TRTMC_VERSION_STRING;
}

int trtmc_has_trt(void) {
    return 1;
}

void trtmc_image_result_free(trtmc_image_result_t* result) {
    if (result == nullptr) {
        return;
    }
    if (result->pixels != nullptr) {
        std::free(result->pixels);
        result->pixels = nullptr;
    }
    result->num_pixels = 0;
}

namespace {

// Validate the raw C-ABI inputs in one place; sets last_error on failure.
int validate_batch_args(trtmc_pipeline_t handle, const char* const* prompts, int num_prompts,
                        const std::uint32_t* seeds, int num_seeds,
                        const trtmc_image_result_t* out_results) {
    const char* reason = nullptr;
    if (handle == nullptr)
        reason = "trtmc_generate_batch: pipeline handle must not be null";
    else if (prompts == nullptr || num_prompts <= 0)
        reason = "trtmc_generate_batch: prompts array must be non-empty";
    else if (seeds == nullptr)
        reason = "trtmc_generate_batch: seeds array must not be null";
    else if (num_prompts != num_seeds)
        reason = "trtmc_generate_batch: num_prompts must equal num_seeds";
    else if (out_results == nullptr)
        reason = "trtmc_generate_batch: out_results must not be null";
    if (reason) {
        set_last_error(reason);
        return TRTMC_ERR_INVALID_ARG;
    }
    return TRTMC_OK;
}

// Copy results into caller-owned C structs. On OOM, frees what was already
// produced and returns TRTMC_ERR_RUNTIME with last_error set.
int copy_results_to_c(const std::vector<trtmc::ImageResult>& results,
                      trtmc_image_result_t* out_results, int num_prompts) {
    for (int i = 0; i < num_prompts; ++i) {
        std::memset(&out_results[i], 0, sizeof(trtmc_image_result_t));
    }
    for (int i = 0; i < num_prompts; ++i) {
        if (!to_c_image_result(results[static_cast<std::size_t>(i)], &out_results[i])) {
            for (int j = 0; j < i; ++j)
                trtmc_image_result_free(&out_results[j]);
            set_last_error("trtmc_generate_batch: out-of-memory copying pixels");
            return TRTMC_ERR_RUNTIME;
        }
    }
    return TRTMC_OK;
}

} // namespace

int trtmc_generate_batch(trtmc_pipeline_t handle, const char* const* prompts, int num_prompts,
                         const std::uint32_t* seeds, int num_seeds, int num_inference_steps,
                         float guidance_scale, trtmc_image_result_t* out_results) {
    clear_last_error();
    if (int rc = validate_batch_args(handle, prompts, num_prompts, seeds, num_seeds, out_results);
        rc != TRTMC_OK) {
        return rc;
    }

    try {
        std::vector<std::string> prompt_vec;
        prompt_vec.reserve(static_cast<std::size_t>(num_prompts));
        for (int i = 0; i < num_prompts; ++i) {
            if (prompts[i] == nullptr) {
                set_last_error("trtmc_generate_batch: prompts[" + std::to_string(i) + "] is null");
                return TRTMC_ERR_INVALID_ARG;
            }
            prompt_vec.emplace_back(prompts[i]);
        }
        std::vector<std::uint32_t> seed_vec(seeds, seeds + num_seeds);

        trtmc::GenerateConfig cfg;
        if (num_inference_steps > 0)
            cfg.num_steps = num_inference_steps;
        cfg.guidance_scale = guidance_scale;

        auto results = handle->generate_image_batch(prompt_vec, seed_vec, cfg);
        if (static_cast<int>(results.size()) != num_prompts) {
            set_last_error("trtmc_generate_batch: pipeline returned " +
                           std::to_string(results.size()) + " results for " +
                           std::to_string(num_prompts) + " prompts");
            return TRTMC_ERR_RUNTIME;
        }
        return copy_results_to_c(results, out_results, num_prompts);
    } catch (const std::invalid_argument& e) {
        set_last_error(e.what());
        return TRTMC_ERR_INVALID_ARG;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return TRTMC_ERR_RUNTIME;
    } catch (...) {
        set_last_error("trtmc_generate_batch: unknown error");
        return TRTMC_ERR_RUNTIME;
    }
}

} // extern "C"
