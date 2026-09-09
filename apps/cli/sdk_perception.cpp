/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/sdk_dispatch.h"
#include "trtmc/perception.hpp"
#include "trtmc/tracking.hpp"

#include <cmath>
#include <nlohmann/json.hpp>

namespace trtmc::cli {
namespace {
using detail::has_option;
using detail::require_option;

template <class T>
nlohmann::json values(const T* data, std::uint64_t count) {
    nlohmann::json result = nlohmann::json::array();
    for (std::uint64_t i = 0; i < count; ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            if (!std::isfinite(data[i]))
                throw std::runtime_error("perception output contains a non-finite value");
        }
        result.push_back(data[i]);
    }
    return result;
}
nlohmann::json box_json(const trtmc_pixel_box_v1& box) {
    const float corners[] = {box.x_min, box.y_min, box.x_max, box.y_max};
    return values(corners, 4);
}
nlohmann::json boxes_json(const trtmc_pixel_box_v1* boxes, std::uint64_t count) {
    nlohmann::json result = nlohmann::json::array();
    for (std::uint64_t i = 0; i < count; ++i)
        for (const auto& coordinate : box_json(boxes[i]))
            result.push_back(coordinate);
    return result;
}
const char* mask_kind(std::uint32_t kind) {
    switch (kind) {
    case TRTMC_MASK_LOGITS:
        return "logits";
    case TRTMC_MASK_PROBABILITY:
        return "probability";
    case TRTMC_MASK_BINARY:
        return "binary";
    default:
        throw std::runtime_error("unknown mask interpretation");
    }
}
nlohmann::json confidence_json(bool present, float score) {
    if (present && !std::isfinite(score))
        throw std::runtime_error("perception confidence is not finite");
    return present ? nlohmann::json(score) : nlohmann::json(nullptr);
}
const char* score_kind(std::uint32_t kind) {
    switch (kind) {
    case TRTMC_SCORE_LOGIT:
        return "logit";
    case TRTMC_SCORE_PROBABILITY:
        return "probability";
    case TRTMC_SCORE_UNBOUNDED:
        return "unbounded";
    default:
        throw std::runtime_error("unknown score interpretation");
    }
}

ImageInput image_view(const io::LoadedImage& image) {
    return {{image.pixels.data(), image.pixels.size()},
            static_cast<std::uint32_t>(image.height),
            static_cast<std::uint32_t>(image.width)};
}

void disparity(const Command& command, const Model& model, std::ostream& output) {
    const auto left = detail::read_image(require_option(command, "--left"));
    const auto right = detail::read_image(require_option(command, "--right"));
    if (left.height != right.height || left.width != right.width)
        throw std::invalid_argument("--left and --right images must have the same dimensions");
    const auto task = model.task<StereoImagesToDisparity>();
    const auto config = detail::task_config(command, task.config_fields(), {"--left", "--right"});
    const auto result = task.run({image_view(left), image_view(right)}, config);
    const auto& view = result.view().disparity;
    nlohmann::json values = nlohmann::json::array();
    for (std::uint64_t i = 0; i < view.count; ++i) {
        if (!std::isfinite(view.data[i]))
            throw std::runtime_error("disparity contains a non-finite value");
        values.push_back(view.data[i]);
    }
    detail::write_json(output, {{"disparity", std::move(values)},
                                {"height", view.rows},
                                {"width", view.columns},
                                {"units", "pixels"},
                                {"grid", "left_image"},
                                {"convention", "x_left_minus_x_right"}});
}

void geometry(const Command& command, const Model& model, std::ostream& output) {
    const auto image = detail::read_image(require_option(command, "--image"));
    const auto task = model.task<ImageToMetricGeometry>();
    const auto config = detail::task_config(command, task.config_fields(), {"--image", "--output"});
    const auto result = task.run({image_view(image)}, config);
    const auto& view = result.view();
    const std::filesystem::path directory = require_option(command, "--output");
    std::filesystem::create_directories(directory);
    // Invalid geometry may intentionally contain infinity. Preserve the raw
    // maps and authoritative validity mask rather than fabricating zero points.
    detail::write_binary(
        directory / "points.f32",
        Span<const float>{view.points, static_cast<std::size_t>(view.point_value_count)});
    detail::write_binary(directory / "depth.f32",
                         Span<const float>{view.depth, static_cast<std::size_t>(view.pixel_count)});
    detail::write_binary(
        directory / "mask.u8",
        Span<const std::uint8_t>{view.valid, static_cast<std::size_t>(view.pixel_count)});
    nlohmann::json intrinsics = nlohmann::json::array();
    for (std::size_t row = 0; row < 3; ++row) {
        intrinsics.push_back({view.normalized_intrinsics[row * 3],
                              view.normalized_intrinsics[row * 3 + 1],
                              view.normalized_intrinsics[row * 3 + 2]});
    }
    std::ofstream matrix_output(directory / "intrinsics.json");
    matrix_output.exceptions(std::ios::badbit | std::ios::failbit);
    matrix_output << nlohmann::json{{"height", view.height},
                                    {"width", view.width},
                                    {"intrinsics", intrinsics},
                                    {"normalized", true}}
                         .dump(2)
                  << '\n';
    matrix_output.close();
    detail::write_json(output, {{"output", directory.string()},
                                {"height", view.height},
                                {"width", view.width},
                                {"points", (directory / "points.f32").string()},
                                {"depth", (directory / "depth.f32").string()},
                                {"mask", (directory / "mask.u8").string()},
                                {"intrinsics", (directory / "intrinsics.json").string()},
                                {"units", "meters"},
                                {"coordinate_frame", "camera_x_right_y_down_z_forward"}});
}

nlohmann::json masks_json(const trtmc_masks_view_v1& view) {
    nlohmann::json proposals = nlohmann::json::array();
    for (std::uint64_t i = 0; i < view.proposal_count; ++i) {
        const auto& proposal = view.proposals[i];
        nlohmann::json seeds = nlohmann::json::array();
        for (std::uint64_t j = 0; j < proposal.seed_point_count; ++j) {
            const float point[] = {proposal.seed_points[j].x, proposal.seed_points[j].y};
            seeds.push_back(values(point, 2));
        }
        proposals.push_back({{"area", proposal.area},
                             {"crop_box", box_json(proposal.crop_box)},
                             {"seed_points", std::move(seeds)}});
    }
    return {{"masks", values(view.masks, view.value_count)},
            {"iou_scores", values(view.predicted_iou, view.iou_count)},
            {"confidence", values(view.confidence, view.confidence_count)},
            {"stability_scores", values(view.stability, view.stability_count)},
            {"boxes", boxes_json(view.boxes, view.box_count)},
            {"num_masks", view.mask_count},
            {"height", view.height},
            {"width", view.width},
            {"mask_kind", mask_kind(view.kind)},
            {"box_coordinates", "original_image_pixels_xyxy"},
            {"object_ids", values(view.object_ids.data, view.object_ids.size)},
            {"proposals", std::move(proposals)},
            {"low_res_logits", values(view.low_res_logits, view.low_res_count)},
            {"low_res_height", view.low_res_height},
            {"low_res_width", view.low_res_width}};
}

void semantic_segment(const Command& command, const Model& model, std::ostream& output) {
    const auto image = detail::read_image(require_option(command, "--image"));
    const auto task = model.task<ImageToSemanticSegmentation>();
    const auto config = detail::task_config(command, task.config_fields(), {"--image"});
    const auto result = task.run({image_view(image)}, config);
    const auto& view = result.view();
    nlohmann::json names = nlohmann::json::array();
    for (std::uint64_t i = 0; i < view.class_names.size; ++i)
        names.push_back(std::string(trtmc::detail::string_view(view.class_names.data[i])));
    nlohmann::json output_value{
        {"mask", values(view.labels, view.pixel_count)},
        {"height", view.height},
        {"width", view.width},
        {"class_ids", values(view.class_ids.data, view.class_ids.size)},
        {"class_names", std::move(names)},
        {"vocabulary_id", std::string(trtmc::detail::string_view(view.vocabulary_id))},
        {"class_scores", values(view.class_scores, view.score_count)},
        {"score_height", view.score_height},
        {"score_width", view.score_width},
        {"score_kind", score_kind(view.score_kind)},
        {"class_score_axes", {"class", "height", "width"}}};
    output_value["ignore_label"] =
        view.has_ignore_label ? nlohmann::json(view.ignore_label) : nlohmann::json(nullptr);
    output_value["background_label"] =
        view.has_background_label ? nlohmann::json(view.background_label) : nlohmann::json(nullptr);
    detail::write_json(output, output_value);
}

void point_masks(const Command& command, const Model& model, bool center_helper,
                 std::ostream& output) {
    if (has_option(command, "--prompt"))
        throw std::invalid_argument("text prompts require the text-instance mask Task");
    const auto image = detail::read_image(require_option(command, "--image"));
    const auto task = model.task<ImagePointsToMasks>();
    const auto config = detail::task_config(command, task.config_fields(),
                                            {"--image", "--point-x", "--point-y", "--foreground"});
    const float x = center_helper ? 0.5F : detail::float_option(command, "--point-x", 0.5F);
    const float y = center_helper ? 0.5F : detail::float_option(command, "--point-y", 0.5F);
    const bool foreground = center_helper || !has_option(command, "--foreground") ||
                            detail::parse_bool(command.options.at("--foreground"), "--foreground");
    // Existing flags express fractional image coordinates quantized to the
    // original pixel grid. Family resizing/prompt embeddings remain family-owned.
    const PointPrompt point{{std::floor(x * image.width), std::floor(y * image.height)},
                            foreground};
    const auto result = task.run({image_view(image), {&point, 1}}, config);
    const auto& view = result.view();
    auto output_value = masks_json(view);
    output_value["point"] = {{"x", point.point.x},
                             {"y", point.point.y},
                             {"foreground", foreground},
                             {"coordinates", "original_image_pixels"}};
    if (center_helper) {
        nlohmann::json mask = nlohmann::json::array();
        if (view.mask_count != 0) {
            const auto area = static_cast<std::uint64_t>(view.height) * view.width;
            for (std::uint64_t i = 0; i < area; ++i)
                mask.push_back(view.masks[i] > 0 ? 1 : 0);
        }
        // Preserve the old caller helper: first already-selected family mask,
        // threshold >0. Never sort by IoU or recreate the decoder's last-K rule.
        output_value["mask"] = std::move(mask);
        output_value["selected_mask_index"] =
            view.mask_count ? nlohmann::json(0) : nlohmann::json(nullptr);
        output_value["selected_mask_kind"] = "binary";
    }
    detail::write_json(output, output_value);
}

void text_masks(const Command& command, const Model& model, std::ostream& output) {
    if (has_option(command, "--point-x") || has_option(command, "--point-y") ||
        has_option(command, "--foreground"))
        throw std::invalid_argument("text-instance masks do not accept point-prompt options");
    const auto image = detail::read_image(require_option(command, "--image"));
    const auto task = model.task<ImageTextToInstanceMasks>();
    const auto config = detail::task_config(command, task.config_fields(), {"--image", "--prompt"});
    const auto result = task.run({image_view(image), require_option(command, "--prompt")}, config);
    // Detection confidence is not predicted IoU, even if an older JSON field
    // used that misleading name. Preserve the two typed score arrays separately.
    detail::write_json(output, masks_json(result.view()));
}

void grounding(const Command& command, const Model& model, std::string_view id,
               std::ostream& output) {
    const auto image = detail::read_image(require_option(command, "--image"));
    const auto prompt = require_option(command, "--prompt");
    nlohmann::json coordinates = nlohmann::json::array(), labels = nlohmann::json::array(),
                   scores = nlohmann::json::array();
    nlohmann::json result{{"image_height", image.height},
                          {"image_width", image.width},
                          {"coordinate_space", "original_image_pixels"}};
    if (id == ImageTextToBoxes::kTask) {
        const auto task = model.task<ImageTextToBoxes>();
        const auto config =
            detail::task_config(command, task.config_fields(), {"--image", "--prompt"});
        const auto output_value = task.run({image_view(image), prompt}, config);
        const auto& view = output_value.view();
        for (std::uint64_t i = 0; i < view.count; ++i) {
            const auto& box = view.boxes[i];
            for (const auto& coordinate : box_json(box.box))
                coordinates.push_back(coordinate);
            labels.push_back(std::string(trtmc::detail::string_view(box.label)));
            scores.push_back(confidence_json(box.has_confidence != 0, box.confidence));
        }
        result["boxes"] = std::move(coordinates);
        result["box_format"] = "xyxy";
        result["raw_response"] = std::string(trtmc::detail::string_view(view.raw_response));
        result["parse_complete"] = view.parse_complete != 0;
    } else {
        const auto task = model.task<ImageTextToPoints>();
        const auto config =
            detail::task_config(command, task.config_fields(), {"--image", "--prompt"});
        const auto output_value = task.run({image_view(image), prompt}, config);
        const auto& view = output_value.view();
        for (std::uint64_t i = 0; i < view.count; ++i) {
            const auto& point = view.points[i];
            const float xy[] = {point.point.x, point.point.y};
            for (const auto& coordinate : values(xy, 2))
                coordinates.push_back(coordinate);
            labels.push_back(std::string(trtmc::detail::string_view(point.label)));
            scores.push_back(confidence_json(point.has_confidence != 0, point.confidence));
        }
        result["points"] = std::move(coordinates);
        result["raw_response"] = std::string(trtmc::detail::string_view(view.raw_response));
        result["parse_complete"] = view.parse_complete != 0;
    }
    result["labels"] = std::move(labels);
    result["scores"] = std::move(scores); // Unknown confidence stays null; no invented class IDs.
    detail::write_json(output, result);
}

nlohmann::json tracks_json(const trtmc_track_clip_view_v1& clip) {
    nlohmann::json frames = nlohmann::json::array();
    for (std::uint64_t i = 0; i < clip.frame_count; ++i) {
        const auto& frame = clip.frames[i];
        if (frame.memory_kind != TRTMC_TRACK_HOST)
            throw std::runtime_error(
                "file video-segment requires host masks, not borrowed CUDA pointers");
        nlohmann::json masks;
        if (frame.element_type == TRTMC_TRACK_UINT8)
            masks = values(static_cast<const std::uint8_t*>(frame.masks), frame.mask_byte_size);
        else if (frame.element_type == TRTMC_TRACK_FLOAT32)
            masks = values(static_cast<const float*>(frame.masks),
                           frame.mask_byte_size / sizeof(float));
        else
            throw std::runtime_error("unknown host track-mask element type");
        frames.push_back(
            {{"masks", std::move(masks)},
             {"frame_index", frame.frame_index},
             {"timestamp_seconds", nullptr},
             {"mask_kind", mask_kind(frame.mask_kind)},
             {"mask_element_type", frame.element_type == TRTMC_TRACK_UINT8 ? "uint8" : "float32"},
             {"object_ids", values(frame.object_ids.data, frame.object_ids.size)},
             {"detection_scores", values(frame.detection_scores, frame.detection_score_count)},
             {"tracking_scores", values(frame.tracker_scores, frame.tracker_score_count)},
             {"boxes", boxes_json(frame.boxes, frame.box_count)},
             {"box_coordinates", "original_image_pixels_xyxy"},
             {"num_objects", frame.object_ids.size},
             {"height", frame.height},
             {"width", frame.width},
             {"class_ids", values(frame.class_ids.data, frame.class_ids.size)},
             {"removed_object_ids",
              values(frame.removed_object_ids.data, frame.removed_object_ids.size)},
             {"suppressed_object_ids",
              values(frame.suppressed_object_ids.data, frame.suppressed_object_ids.size)}});
    }
    nlohmann::json detections = nlohmann::json::array();
    for (std::uint64_t i = 0; i < clip.detection_count; ++i) {
        const auto& detection = clip.initial_detections[i];
        detections.push_back({{"frame_index", detection.frame_index},
                              {"object_id", detection.object_id},
                              {"class_id", detection.class_id},
                              {"score", confidence_json(true, detection.score)},
                              {"prompt_box", box_json(detection.prompt_box)}});
    }
    return {{"frames", std::move(frames)}, {"initial_detections", std::move(detections)}};
}

void video_segment(const Command& command, const Model& model, std::string_view id,
                   std::ostream& output) {
    if (command.frames.empty())
        throw std::invalid_argument("video-segment requires at least one --frame");
    std::vector<io::LoadedImage> images;
    images.reserve(command.frames.size());
    for (const auto& path : command.frames)
        images.push_back(detail::read_image(path));
    VideoInput clip;
    for (const auto& image : images)
        clip.frames.push_back(image_view(image));
    // The existing frame-file command carries ordering but no physical times.
    // It delegates one complete clip; family owns any two-stage composition.
    if (id == FramesTextToMaskTracks::kTask) {
        const auto task = model.task<FramesTextToMaskTracks>();
        const auto config = detail::task_config(command, task.config_fields(), {"--prompt"});
        auto session = task.create(config);
        const auto result = session.segment(clip, require_option(command, "--prompt"));
        detail::write_json(output, tracks_json(result.view()));
    } else {
        if (has_option(command, "--prompt"))
            throw std::invalid_argument("text prompts require the text-driven clip Task");
        const auto task = model.task<FramesToDetectedMaskTracks>();
        const auto config = detail::task_config(command, task.config_fields(), {});
        auto session = task.create(config);
        const auto result = session.segment(clip);
        detail::write_json(output, tracks_json(result.view()));
    }
}
} // namespace

std::string_view perception_task_for_command(const Command& command, const Model& model) {
    if (!command.selected_task.empty())
        return {};
    if (command.kind == CommandKind::kDisparity)
        return StereoImagesToDisparity::kTask;
    if (command.kind == CommandKind::kGeometry)
        return ImageToMetricGeometry::kTask;
    if (command.kind == CommandKind::kSegment) {
        const auto primary = model.info().bundle_task;
        if (primary == ImagePointsToMasks::kTask)
            return ImagePointsToMasks::kTask;
        if (primary == ImageToMaskProposals::kTask)
            return ImageToMaskProposals::kTask;
        if (model.supports<ImageToSemanticSegmentation>())
            return ImageToSemanticSegmentation::kTask;
        return ImagePointsToMasks::kTask;
    }
    if (command.kind == CommandKind::kSegmentPrompted)
        return has_option(command, "--prompt") ? ImageTextToInstanceMasks::kTask
                                               : ImagePointsToMasks::kTask;
    if (command.kind == CommandKind::kDetect)
        return model.info().bundle_task == ImageTextToPoints::kTask ? ImageTextToPoints::kTask
                                                                    : ImageTextToBoxes::kTask;
    if (command.kind == CommandKind::kVideoSegment) {
        const auto primary = model.info().bundle_task;
        if (primary == PromptFrameTextToMaskTracks::kTask)
            return PromptFrameTextToMaskTracks::kTask; // Not silently adapted to a whole clip.
        return has_option(command, "--prompt") || primary == FramesTextToMaskTracks::kTask
                   ? FramesTextToMaskTracks::kTask
                   : FramesToDetectedMaskTracks::kTask;
    }
    return {};
}

bool dispatch_sdk_perception(const Command& command, const Model& model, std::string_view id,
                             std::ostream& output) {
    if (command.kind == CommandKind::kDisparity && id == StereoImagesToDisparity::kTask) {
        disparity(command, model, output);
        return true;
    }
    if (command.kind == CommandKind::kGeometry && id == ImageToMetricGeometry::kTask) {
        geometry(command, model, output);
        return true;
    }
    if (command.kind == CommandKind::kSegment && id == ImageToSemanticSegmentation::kTask) {
        semantic_segment(command, model, output);
        return true;
    }
    if ((command.kind == CommandKind::kSegment || command.kind == CommandKind::kSegmentPrompted) &&
        id == ImagePointsToMasks::kTask) {
        point_masks(command, model, command.kind == CommandKind::kSegment, output);
        return true;
    }
    if (command.kind == CommandKind::kSegmentPrompted && id == ImageTextToInstanceMasks::kTask) {
        text_masks(command, model, output);
        return true;
    }
    if (command.kind == CommandKind::kSegment && id == ImageToMaskProposals::kTask) {
        const auto image = detail::read_image(require_option(command, "--image"));
        const auto task = model.task<ImageToMaskProposals>();
        const auto config = detail::task_config(command, task.config_fields(), {"--image"});
        const auto result = task.run({image_view(image)}, config);
        detail::write_json(output, masks_json(result.view()));
        return true;
    }
    if (command.kind == CommandKind::kDetect &&
        (id == ImageTextToBoxes::kTask || id == ImageTextToPoints::kTask)) {
        grounding(command, model, id, output);
        return true;
    }
    if (command.kind == CommandKind::kVideoSegment &&
        (id == FramesTextToMaskTracks::kTask || id == FramesToDetectedMaskTracks::kTask)) {
        video_segment(command, model, id, output);
        return true;
    }
    return false;
}

} // namespace trtmc::cli
