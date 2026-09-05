/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/sam3/runtime/plugin_helpers.h"
#include "families/sam3/runtime/sam3_pipeline.h"
#include "trtmc/runtime/family_factory.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::sam3_factory {
namespace {

std::vector<char> require_section(const BundleReader& bundle, const char* name) {
    const auto* section = bundle.find_section(name);
    if (section == nullptr || section->length == 0)
        throw std::runtime_error("bundle section is missing or empty: " + std::string(name));
    return bundle.read_section(name);
}

Sam3Config parse_config(const nlohmann::json& json) {
    Sam3Config config;
#define SAM3_INT(field) config.field = json.at(#field).get<std::int32_t>()
    SAM3_INT(text_max_position_embeddings);
    SAM3_INT(text_pad_token_id);
    SAM3_INT(image_size);
    SAM3_INT(low_res_mask_size);
    SAM3_INT(num_queries);
    SAM3_INT(hotstart_delay);
    SAM3_INT(hotstart_unmatch_threshold);
    SAM3_INT(hotstart_duplicate_threshold);
    SAM3_INT(initial_tracker_keep_alive);
    SAM3_INT(max_tracker_keep_alive);
    SAM3_INT(min_tracker_keep_alive);
    SAM3_INT(recondition_every_nth_frame);
    SAM3_INT(fill_hole_area);
    SAM3_INT(max_tracked_objects);
    SAM3_INT(num_mask_memory_frames);
    SAM3_INT(max_conditioning_frames);
    SAM3_INT(max_object_pointers);
    SAM3_INT(max_video_frames);
    SAM3_INT(max_conditioning_pointers);
    SAM3_INT(max_pointer_inputs);
#undef SAM3_INT
#define SAM3_FLOAT(field) config.field = json.at(#field).get<float>()
    SAM3_FLOAT(score_threshold);
    SAM3_FLOAT(mask_threshold);
    SAM3_FLOAT(detection_threshold);
    SAM3_FLOAT(detection_nms_threshold);
    SAM3_FLOAT(association_iou_threshold);
    SAM3_FLOAT(tracker_association_iou_threshold);
    SAM3_FLOAT(new_detection_threshold);
    SAM3_FLOAT(high_confidence_threshold);
    SAM3_FLOAT(high_iou_threshold);
    SAM3_FLOAT(overlap_suppression_threshold);
#undef SAM3_FLOAT
    config.suppress_unmatched_only_within_hotstart =
        json.at("suppress_unmatched_only_within_hotstart").get<bool>();
    config.decrease_keep_alive_for_empty_masks =
        json.at("decrease_keep_alive_for_empty_masks").get<bool>();
    config.image_mean = json.at("image_mean").get<std::vector<float>>();
    config.image_std = json.at("image_std").get<std::vector<float>>();
    if (config.text_max_position_embeddings <= 0 || config.image_size <= 0 ||
        config.low_res_mask_size <= 0 || config.num_queries <= 0 || config.max_video_frames <= 0 ||
        config.image_mean.size() != 3 || config.image_std.size() != 3)
        throw std::runtime_error("SAM3 runtime.json does not match its runtime contract");
    return config;
}

std::unique_ptr<ITrtModule> load(IBackend& backend, const BundleReader& bundle, const char* name,
                                 cudaStream_t stream = nullptr) {
    ModuleCreateOptions options{};
    options.stream = stream;
    const auto& plan = require_section(bundle, name);
    return load_trt_module_from_plan(&backend, &plan, name, options).module;
}

} // namespace
} // namespace trtmc::sam3_factory

extern "C" trtmc::ITask* trtmc_create_family(const trtmc::FamilyContext& context) {
    if (context.kv_cache_size_bytes != 0)
        throw std::invalid_argument("sam3 does not support --kv-cache-size");
    using namespace trtmc;
    const auto& runtime = sam3_factory::require_section(context.reader, "runtime.json");
    const auto config =
        sam3_factory::parse_config(nlohmann::json::parse(runtime.begin(), runtime.end()));
    auto text = sam3_factory::load(context.backend, context.reader, "engine.plan");
    auto vision = sam3_factory::load(context.backend, context.reader, "vision.plan");
    auto core = sam3_factory::load(context.backend, context.reader, "core.plan");
    auto tracker_init = sam3_factory::load(context.backend, context.reader, "tracker.init.plan");
    auto parallel_init = sam3_factory::load(context.backend, context.reader, "tracker.init.plan");
    auto tracker_step = sam3_factory::load(context.backend, context.reader, "tracker.step.plan");
    auto tracker_step_batch2 =
        sam3_factory::load(context.backend, context.reader, "tracker.step.batch2.plan");
    auto tracker_memory =
        sam3_factory::load(context.backend, context.reader, "tracker.memory.plan");
    auto tracker_memory_batch2 =
        sam3_factory::load(context.backend, context.reader, "tracker.memory.batch2.plan");
    auto hard_memory =
        sam3_factory::load(context.backend, context.reader, "tracker.hard_memory.plan");
    auto hard_memory_batch2 =
        sam3_factory::load(context.backend, context.reader, "tracker.hard_memory.batch2.plan");
    auto mask_resize = sam3_factory::load(context.backend, context.reader, "mask_resize.plan");
    auto mask_resize_batch2 =
        sam3_factory::load(context.backend, context.reader, "mask_resize.batch2.plan");
    auto tokenizer = create_tokenizer_from_bundle(context.reader);
    if (!tokenizer)
        throw std::runtime_error("SAM3 bundle does not contain its required tokenizer");
    return new Sam3Pipeline(
        std::move(text), std::move(vision), std::move(core), std::move(tokenizer), config, "",
        std::move(tracker_init), std::move(tracker_step), std::move(tracker_memory),
        std::move(tracker_step_batch2), std::move(tracker_memory_batch2), std::move(parallel_init),
        std::move(hard_memory), std::move(hard_memory_batch2), std::move(mask_resize),
        std::move(mask_resize_batch2));
}
