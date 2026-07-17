/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Sam3Plugin: SAM3-owned text-prompted segmentation strategy.

#include "plugin_helpers.h"
#include "runtime/models/sam3/sam3_pipeline.h"
#include "runtime/models/sam3/sam3_tracker_step_runtime.h"
#include "runtime/models/sam3/sam3_video_c_api.h"
#include "trtmc/runtime/pipeline_registry.h"
#include "utils/json_helpers.h"

#include <chrono>
#include <exception>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc {

namespace {

thread_local std::string sam3_video_last_error;

struct Sam3CustomerFrame {
    const float* pixels{nullptr};
    int32_t height{0};
    int32_t width{0};
};

struct Sam3CustomerPredictor {
    std::unique_ptr<IPipeline> pipeline;
    std::unique_ptr<Sam3VideoSegmentationSession> session;
    std::vector<Sam3CustomerFrame> frames;
    std::optional<Sam3VideoFrameResult> prompt_result;
    int32_t prompt_object_count{-1};
};

template <typename Function, typename Result>
Result translate_sam3_video_errors(Function&& function, Result failure) noexcept {
    try {
        sam3_video_last_error.clear();
        return function();
    } catch (const std::exception& error) {
        sam3_video_last_error = error.what();
    } catch (...) {
        sam3_video_last_error = "unknown native exception";
    }
    return failure;
}

Sam3CustomerPredictor& require_sam3_customer_predictor(void* opaque) {
    if (opaque == nullptr)
        throw std::invalid_argument("null Model Connect SAM3 video predictor");
    return *static_cast<Sam3CustomerPredictor*>(opaque);
}

Sam3Config make_sam3_config(const std::string& json) {
    Sam3Config cfg;
    cfg.text_max_position_embeddings = extract_json_int(json, "sam3_text_max_position_embeddings",
                                                        cfg.text_max_position_embeddings);
    cfg.text_pad_token_id = extract_json_int(json, "sam3_text_pad_token_id", cfg.text_pad_token_id);
    cfg.image_size = extract_json_int(json, "sam3_image_size",
                                      extract_json_int(json, "input_image_h", cfg.image_size));
    cfg.low_res_mask_size = extract_json_int(json, "sam3_low_res_mask_size", cfg.low_res_mask_size);
    cfg.num_queries = extract_json_int(json, "sam3_num_queries", cfg.num_queries);
    cfg.score_threshold = extract_json_float(json, "sam3_score_threshold", cfg.score_threshold);
    cfg.mask_threshold = extract_json_float(json, "sam3_mask_threshold", cfg.mask_threshold);
    cfg.detection_threshold =
        extract_json_float(json, "sam3_detection_threshold", cfg.detection_threshold);
    cfg.detection_nms_threshold =
        extract_json_float(json, "sam3_detection_nms_threshold", cfg.detection_nms_threshold);
    cfg.association_iou_threshold =
        extract_json_float(json, "sam3_assoc_iou_threshold", cfg.association_iou_threshold);
    cfg.tracker_association_iou_threshold = extract_json_float(
        json, "sam3_tracker_assoc_iou_threshold", cfg.tracker_association_iou_threshold);
    cfg.new_detection_threshold =
        extract_json_float(json, "sam3_new_detection_threshold", cfg.new_detection_threshold);
    cfg.high_confidence_threshold =
        extract_json_float(json, "sam3_high_confidence_threshold", cfg.high_confidence_threshold);
    cfg.high_iou_threshold =
        extract_json_float(json, "sam3_high_iou_threshold", cfg.high_iou_threshold);
    cfg.overlap_suppression_threshold = extract_json_float(
        json, "sam3_overlap_suppression_threshold", cfg.overlap_suppression_threshold);
    cfg.hotstart_delay = extract_json_int(json, "sam3_hotstart_delay", cfg.hotstart_delay);
    cfg.hotstart_unmatch_threshold =
        extract_json_int(json, "sam3_hotstart_unmatch_threshold", cfg.hotstart_unmatch_threshold);
    cfg.hotstart_duplicate_threshold = extract_json_int(json, "sam3_hotstart_duplicate_threshold",
                                                        cfg.hotstart_duplicate_threshold);
    cfg.suppress_unmatched_only_within_hotstart =
        extract_json_bool(json, "sam3_suppress_unmatched_only_within_hotstart",
                          cfg.suppress_unmatched_only_within_hotstart);
    cfg.initial_tracker_keep_alive =
        extract_json_int(json, "sam3_initial_tracker_keep_alive", cfg.initial_tracker_keep_alive);
    cfg.max_tracker_keep_alive =
        extract_json_int(json, "sam3_max_tracker_keep_alive", cfg.max_tracker_keep_alive);
    cfg.min_tracker_keep_alive =
        extract_json_int(json, "sam3_min_tracker_keep_alive", cfg.min_tracker_keep_alive);
    cfg.decrease_keep_alive_for_empty_masks = extract_json_bool(
        json, "sam3_decrease_keep_alive_for_empty_masks", cfg.decrease_keep_alive_for_empty_masks);
    cfg.recondition_every_nth_frame =
        extract_json_int(json, "sam3_recondition_every_nth_frame", cfg.recondition_every_nth_frame);
    cfg.fill_hole_area = extract_json_int(json, "sam3_fill_hole_area", cfg.fill_hole_area);
    cfg.max_tracked_objects =
        extract_json_int(json, "sam3_max_tracked_objects", cfg.max_tracked_objects);
    cfg.num_mask_memory_frames =
        extract_json_int(json, "sam3_num_mask_memory_frames", cfg.num_mask_memory_frames);
    cfg.max_conditioning_frames =
        extract_json_int(json, "sam3_max_conditioning_frames", cfg.max_conditioning_frames);
    cfg.max_object_pointers =
        extract_json_int(json, "sam3_max_object_pointers", cfg.max_object_pointers);
    cfg.max_video_frames = extract_json_int(json, "sam3_max_video_frames", cfg.max_video_frames);
    cfg.max_conditioning_pointers =
        extract_json_int(json, "sam3_max_conditioning_pointers", cfg.max_conditioning_pointers);
    cfg.max_pointer_inputs =
        extract_json_int(json, "sam3_max_pointer_inputs", cfg.max_pointer_inputs);
    auto mean = extract_json_float_array(json, "image_mean", 3);
    if (mean.size() == 3)
        cfg.image_mean = std::move(mean);
    auto stdv = extract_json_float_array(json, "image_std", 3);
    if (stdv.size() == 3)
        cfg.image_std = std::move(stdv);
    return cfg;
}

struct Sam3TrackerInitModules {
    std::unique_ptr<ITrtModule> canonical;
    std::unique_ptr<ITrtModule> parallel_sibling;
};

struct Sam3TrackerStepModules {
    std::unique_ptr<ITrtModule> batch1;
    std::unique_ptr<ITrtModule> batch2;
};

struct Sam3TrackerMemoryModules {
    std::unique_ptr<ITrtModule> soft_batch1;
    std::unique_ptr<ITrtModule> soft_batch2;
    std::unique_ptr<ITrtModule> hard_batch1;
    std::unique_ptr<ITrtModule> hard_batch2;
};

struct Sam3HardMaskResizeModules {
    std::unique_ptr<ITrtModule> batch1;
    std::unique_ptr<ITrtModule> batch2;
};

std::vector<std::unique_ptr<ITrtModule>>
load_sam3_parallel_modules(IBackend& backend, const std::vector<char>& plan, const char* label,
                           const ModuleCreateOptions& options, std::size_t count) {
    ModuleCreateOptions independent_options = options;
    // create_module() gives each module a backend-owned stream when no stream
    // is supplied. SAM3 deliberately pays for duplicate engine ownership here
    // so its concurrent fixed-shape lanes do not require a generic backend
    // capability or share execution-context state.
    independent_options.stream = nullptr;
    independent_options.cuda_graphs = false;

    std::vector<std::unique_ptr<ITrtModule>> modules;
    modules.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        auto loaded = load_trt_module_from_plan(&backend, &plan, label, independent_options);
        if (loaded.module == nullptr || !loaded.module->ok() || loaded.module->stream() == nullptr)
            throw std::runtime_error(std::string("SAM3 failed to create independent ") + label);
        for (const auto& existing : modules) {
            if (existing->stream() == loaded.module->stream()) {
                throw std::runtime_error(std::string("SAM3 backend reused a stream for ") + label);
            }
        }
        modules.push_back(std::move(loaded.module));
    }
    return modules;
}

Sam3TrackerInitModules load_sam3_tracker_init_modules(const PipelineContext& ctx,
                                                      const ModuleCreateOptions& options) {
    constexpr const char* kLabel = "sam3 tracker_init_engine";
    const auto* plan = find_section(ctx.bundle, "sam3_tracker_init_engine_plan");
    if (plan == nullptr || plan->empty() || ctx.backend == nullptr) {
        return {extract_optional_module(ctx.backend, plan, kLabel, options), nullptr};
    }

    try {
        const auto t0 = std::chrono::steady_clock::now();
        auto loaded = load_sam3_parallel_modules(*ctx.backend, *plan, kLabel, options, 2);
        const auto t1 = std::chrono::steady_clock::now();
        log_trt_load_timing(kLabel, std::chrono::duration<double, std::milli>(t1 - t0).count(),
                            plan->size());
        if (loaded.size() == 2 && loaded[0]->stream() != nullptr &&
            loaded[1]->stream() != nullptr && loaded[0]->stream() != loaded[1]->stream()) {
            loaded[0]->set_timing_label(std::string(kLabel) + ":parallel0");
            loaded[1]->set_timing_label(std::string(kLabel) + ":parallel1");
            return {std::move(loaded[0]), std::move(loaded[1])};
        }
        std::cerr << "[trtmc] WARNING: " << kLabel
                  << " did not create two profile-0 contexts on distinct streams; falling back "
                     "to serial tracker init"
                  << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "[trtmc] WARNING: failed to create parallel contexts for " << kLabel << ": "
                  << e.what() << "; falling back to serial tracker init" << std::endl;
    } catch (...) {
        std::cerr << "[trtmc] WARNING: failed to create parallel contexts for " << kLabel
                  << "; falling back to serial tracker init" << std::endl;
    }

    return {extract_optional_module(ctx.backend, plan, kLabel, options), nullptr};
}

Sam3TrackerMemoryModules load_sam3_tracker_memory_modules(const PipelineContext& ctx,
                                                          const ModuleCreateOptions& options,
                                                          bool video_tracking_supported) {
    if (!video_tracking_supported)
        return {};

    auto soft_batch1 = load_trt_module_from_plan(
        ctx.backend, find_section(ctx.bundle, "sam3_tracker_memory_engine_plan"),
        "sam3 tracker_memory_engine", options);
    auto soft_batch2 = load_trt_module_from_plan(
        ctx.backend, find_section(ctx.bundle, "sam3_tracker_memory_batch2_engine_plan"),
        "sam3 tracker_memory_batch2_engine", options);
    auto hard_batch1 = load_trt_module_from_plan(
        ctx.backend, find_section(ctx.bundle, "sam3_tracker_hard_memory_engine_plan"),
        "sam3 tracker_hard_memory_engine", options);
    auto hard_batch2 = load_trt_module_from_plan(
        ctx.backend, find_section(ctx.bundle, "sam3_tracker_hard_memory_batch2_engine_plan"),
        "sam3 tracker_hard_memory_batch2_engine", options);
    return {std::move(soft_batch1.module), std::move(soft_batch2.module),
            std::move(hard_batch1.module), std::move(hard_batch2.module)};
}

Sam3HardMaskResizeModules load_sam3_hard_mask_resize_modules(const PipelineContext& ctx,
                                                             const ModuleCreateOptions& options,
                                                             bool video_tracking_supported) {
    if (!video_tracking_supported)
        return {};

    auto batch1 = load_trt_module_from_plan(
        ctx.backend, find_section(ctx.bundle, "sam3_hard_mask_resize_engine_plan"),
        "sam3 hard_mask_resize_engine", options);
    auto batch2 = load_trt_module_from_plan(
        ctx.backend, find_section(ctx.bundle, "sam3_hard_mask_resize_batch2_engine_plan"),
        "sam3 hard_mask_resize_batch2_engine", options);
    return {std::move(batch1.module), std::move(batch2.module)};
}

Sam3TrackerStepModules load_sam3_tracker_step_modules(const PipelineContext& ctx,
                                                      const ModuleCreateOptions& options,
                                                      bool video_tracking_supported) {
    if (!video_tracking_supported)
        return {};

    // The model-owned DSO registers both FFI plugin creators, the two split
    // tracker-step functions, and all four fixed tracker-memory functions.
    // This must complete before TensorRT sees any serialized step or memory
    // plugin layer; the memory plans are deserialized below only after this
    // helper returns.
    load_sam3_tracker_step_runtime(ctx.bundle);
    auto batch1 = load_trt_module_from_plan(
        ctx.backend, find_section(ctx.bundle, "sam3_tracker_step_engine_plan"),
        "sam3 tracker_step_engine", options);
    auto batch2 = load_trt_module_from_plan(
        ctx.backend, find_section(ctx.bundle, "sam3_tracker_step_batch2_engine_plan"),
        "sam3 tracker_step_batch2_engine", options);
    return {std::move(batch1.module), std::move(batch2.module)};
}

} // namespace

class Sam3Plugin final : public IPipelinePlugin {
  public:
    std::unique_ptr<IPipeline> create(const PipelineContext& ctx) override {
        load_ffi_kernels_from_bundle(ctx.bundle);

        ModuleCreateOptions opts;
        opts.runtime_cache_path = ctx.runtime_cache_path.c_str();
        // CUDA graphs regressed the qualified SAM3 L4 path and rebinding
        // device-resident vision features would invalidate captured addresses.
        // Keep them disabled for this model even when requested globally.
        opts.cuda_graphs = false;

        const auto sam3_config = make_sam3_config(ctx.config_json);
        const bool video_tracking_supported =
            extract_json_bool(ctx.config_json, "sam3_video_tracking_supported", false);
        auto text_encoder = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "engine_plan"), "sam3 text_encoder", opts);
        auto vision_encoder =
            load_trt_module_from_plan(ctx.backend, find_section(ctx.bundle, "vision_engine_plan"),
                                      "sam3 vision_encoder", opts);
        auto core_engine = load_trt_module_from_plan(
            ctx.backend, find_section(ctx.bundle, "sam3_core_engine_plan"), "sam3 core_engine",
            opts);
        auto tracker_init_engines = load_sam3_tracker_init_modules(ctx, opts);
        auto tracker_step_engines =
            load_sam3_tracker_step_modules(ctx, opts, video_tracking_supported);
        auto tracker_memory_engines =
            load_sam3_tracker_memory_modules(ctx, opts, video_tracking_supported);
        auto hard_mask_resize_engines =
            load_sam3_hard_mask_resize_modules(ctx, opts, video_tracking_supported);
        auto tokenizer = create_tokenizer_from_bundle(ctx.bundle);
        return std::make_unique<Sam3Pipeline>(
            std::move(text_encoder.module), std::move(vision_encoder.module),
            std::move(core_engine.module), std::move(tokenizer), sam3_config,
            ctx.bundle.info.model_id, std::move(tracker_init_engines.canonical),
            std::move(tracker_step_engines.batch1), std::move(tracker_memory_engines.soft_batch1),
            std::move(tracker_step_engines.batch2), std::move(tracker_memory_engines.soft_batch2),
            std::move(tracker_init_engines.parallel_sibling),
            std::move(tracker_memory_engines.hard_batch1),
            std::move(tracker_memory_engines.hard_batch2),
            std::move(hard_mask_resize_engines.batch1), std::move(hard_mask_resize_engines.batch2));
    }
};

extern "C" {

const char* trtmc_sam3_video_last_error() noexcept {
    return sam3_video_last_error.c_str();
}

void* trtmc_sam3_video_create(const char* bundle_path, const char* plugin_dir,
                              const char* backend_dir) noexcept {
    return translate_sam3_video_errors(
        [&]() -> void* {
            if (bundle_path == nullptr || plugin_dir == nullptr || backend_dir == nullptr)
                throw std::invalid_argument("bundle, plugin, and backend paths are required");
            LoadOptions options;
            options.model_plugin_search_paths.emplace_back(plugin_dir);
            options.backend_search_paths.emplace_back(backend_dir);
            auto predictor = std::make_unique<Sam3CustomerPredictor>();
            predictor->pipeline = load(bundle_path, options);
            auto* sam3_pipeline = dynamic_cast<Sam3Pipeline*>(predictor->pipeline.get());
            if (sam3_pipeline == nullptr) {
                throw std::runtime_error(
                    "loaded bundle did not create a SAM3 prompted-segmentation pipeline");
            }
            return predictor.release();
        },
        static_cast<void*>(nullptr));
}

void trtmc_sam3_video_destroy(void* opaque) noexcept {
    delete static_cast<Sam3CustomerPredictor*>(opaque);
}

int32_t trtmc_sam3_video_begin(void* opaque, int32_t expected_frames) noexcept {
    return translate_sam3_video_errors(
        [&]() -> int32_t {
            auto& predictor = require_sam3_customer_predictor(opaque);
            if (expected_frames <= 0)
                throw std::invalid_argument("expected frame count must be positive");
            predictor.session.reset();
            predictor.prompt_result.reset();
            predictor.prompt_object_count = -1;
            predictor.frames.clear();
            predictor.frames.reserve(static_cast<std::size_t>(expected_frames));
            return 0;
        },
        -1);
}

int32_t trtmc_sam3_video_append_frame(void* opaque, const float* pixels, int32_t height,
                                      int32_t width) noexcept {
    return translate_sam3_video_errors(
        [&]() -> int32_t {
            auto& predictor = require_sam3_customer_predictor(opaque);
            if (pixels == nullptr || height <= 0 || width <= 0)
                throw std::invalid_argument("invalid decoded customer frame");
            predictor.frames.push_back({pixels, height, width});
            return 0;
        },
        -1);
}

int32_t trtmc_sam3_video_add_prompt(void* opaque, const char* prompt) noexcept {
    return translate_sam3_video_errors(
        [&]() -> int32_t {
            auto& predictor = require_sam3_customer_predictor(opaque);
            if (prompt == nullptr || *prompt == '\0')
                throw std::invalid_argument("customer prompt must be non-empty");
            if (predictor.frames.empty())
                throw std::runtime_error("customer session has no decoded frames");
            auto* sam3_pipeline = dynamic_cast<Sam3Pipeline*>(predictor.pipeline.get());
            if (sam3_pipeline == nullptr)
                throw std::runtime_error("loaded pipeline is not owned by the SAM3 plugin");
            predictor.session = sam3_pipeline->create_sam3_video_session(prompt);
            const auto& frame = predictor.frames.front();
            auto result =
                predictor.session->accept_prompt_frame(frame.pixels, frame.height, frame.width);
            predictor.prompt_object_count = static_cast<int32_t>(result.object_ids.size());
            predictor.prompt_result = std::move(result);
            return predictor.prompt_object_count;
        },
        -1);
}

int32_t trtmc_sam3_video_propagate(void* opaque, int32_t* object_counts,
                                   int32_t capacity) noexcept {
    return translate_sam3_video_errors(
        [&]() -> int32_t {
            auto& predictor = require_sam3_customer_predictor(opaque);
            if (!predictor.session)
                throw std::runtime_error("customer prompt must run before propagation");
            if (object_counts == nullptr || capacity < 0 ||
                static_cast<std::size_t>(capacity) < predictor.frames.size()) {
                throw std::invalid_argument("object-count output buffer is too small");
            }
            if (predictor.prompt_object_count < 0 || !predictor.prompt_result.has_value())
                throw std::runtime_error("sequential B1 propagation has no prompt-frame result");

            std::vector<Sam3VideoFrameView> views;
            views.reserve(predictor.frames.size());
            for (const auto& frame : predictor.frames)
                views.push_back({frame.pixels, frame.height, frame.width});
            auto results = predictor.session->propagate_borrowed_continuation(
                std::move(*predictor.prompt_result), views.data(), views.size());
            predictor.prompt_result.reset();
            if (results.size() != predictor.frames.size()) {
                throw std::runtime_error(
                    "stateful offline continuation returned the wrong frame count");
            }
            for (std::size_t index = 0; index < results.size(); ++index)
                object_counts[index] = static_cast<int32_t>(results[index].object_ids.size());
            return static_cast<int32_t>(results.size());
        },
        -1);
}

int32_t trtmc_sam3_video_close_session(void* opaque) noexcept {
    return translate_sam3_video_errors(
        [&]() -> int32_t {
            auto& predictor = require_sam3_customer_predictor(opaque);
            predictor.session.reset();
            predictor.prompt_result.reset();
            predictor.prompt_object_count = -1;
            predictor.frames.clear();
            return 0;
        },
        -1);
}

} // extern "C"

REGISTER_PIPELINE_PLUGIN_WITH_MANIFEST(register_sam3_plugin, Sam3Plugin,
                                       "sam3_prompted_segmentation");

} // namespace trtmc
