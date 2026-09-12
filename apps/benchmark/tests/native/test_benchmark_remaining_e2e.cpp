/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace {
using Json = nlohmann::json;
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}
std::string quote(const std::string& value) {
    std::string output{"'"};
    for (char c : value)
        output += c == '\'' ? "'\\''" : std::string(1, c);
    return output + "'";
}
void bundle(const std::filesystem::path& path, const std::string& task, const std::string& family) {
    const auto header = Json{
        {"format", 1},
        {"family", family},
        {"task", task},
        {"backend", "fake"},
        {"sections",
         Json::object()}}.dump();
    std::ofstream output(path, std::ios::binary);
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output.write("BUNDLE\x01\x00", 8);
    for (unsigned shift = 0; shift < 64; shift += 8)
        output.put(static_cast<char>((static_cast<std::uint64_t>(header.size()) >> shift) & 255U));
    output << header;
}
void image(const std::filesystem::path& path, int width, unsigned char red, int height = 1) {
    std::ofstream output(path, std::ios::binary);
    output.exceptions(std::ios::failbit | std::ios::badbit);
    output << "P6\n" << width << ' ' << height << "\n255\n";
    for (int i = 0; i < width * height; ++i) {
        output.put(static_cast<char>(red));
        output.put(0);
        output.put(0);
    }
}
} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: test_benchmark_remaining_e2e WORKER SDK_RUNTIME_ROOT\n";
        return 2;
    }
    try {
        const std::filesystem::path runtime(argv[2]);
        const auto model = runtime / "benchmark_remaining.bundle";
        const auto input = runtime / "benchmark_remaining_request.json";
        const auto output = runtime / "benchmark_remaining_result.json";
        auto artifact = output;
        artifact.replace_extension(".disparity.f32");
        const auto left = runtime / "benchmark_remaining_left.ppm";
        const auto right = runtime / "benchmark_remaining_right.ppm";
        image(left, 2, 255);
        image(right, 2, 0);
        Json request{
            {"schema_version", 2},
            {"case_name", "remaining-sdk"},
            {"bundle", model.string()},
            {"runtime_root", runtime.string()},
            {"operation", "disparity"},
            {"request", {{"left_image_path", left.string()}, {"right_image_path", right.string()}}},
            {"measurement",
             {{"warmup", 1}, {"iterations", 2}, {"timing_scope", "public_task_call_wall"}}}};
        const auto command = quote(argv[1]) + " --request " + quote(input.string()) + " --output " +
                             quote(output.string());
        auto run = [&](bool success = true) {
            {
                std::ofstream file(input);
                file << request;
            }
            const int status = std::system(command.c_str());
            check(success ? status == 0 : status != 0, "worker process status");
            std::ifstream file(output);
            Json result;
            file >> result;
            check(result.at("status") == (success ? "completed" : "failed"), "receipt status");
            if (!success)
                check(!result.contains("observations"), "failure has no completed observations");
            return result;
        };
        bundle(model, "stereo_images_to_disparity", "perception_fixture");
        for (bool include_assets : {false, true}) {
            request["measurement"]["asset_loading_included"] = include_assets;
            const auto result = run();
            const auto& summary = result.at("output_summary");
            check(result.at("observations").size() == 2 &&
                      result.at("observation_serialization_included") == false &&
                      summary.at("disparity_pixels") == 2 && summary.at("height") == 1 &&
                      summary.at("width") == 2 && summary.at("element_count") == 2 &&
                      summary.at("units") == "pixels" &&
                      summary.at("convention") == "x_left_minus_x_right",
                  "stereo metadata, call count and timing boundary preserved");
            std::ifstream file(artifact, std::ios::binary);
            float values[2]{};
            file.read(reinterpret_cast<char*>(values), sizeof(values));
            check(file.good() && values[0] == 1 && values[1] == 1 &&
                      std::filesystem::file_size(artifact) == sizeof(values),
                  "final actual disparity map written exactly once as F32");
        }
        if (std::filesystem::exists("/dev/full")) {
            std::filesystem::remove(artifact);
            std::filesystem::create_symlink("/dev/full", artifact);
            run(false);
            std::filesystem::remove(artifact);
        }
        std::filesystem::create_directory(artifact);
        run(false);
        std::filesystem::remove(artifact);
        request["request"]["config"] = {{"unknown", 1}};
        run(false);
        request["request"].erase("config");
        image(right, 1, 0);
        run(false);
        image(right, 2, 0);
        auto select = [&](const char* task, const char* family, const char* operation, Json args) {
            bundle(model, task, family);
            request["operation"] = operation;
            request["request"] = std::move(args);
        };
        select("text_to_head_scores", "features_fixture", "head_scores",
               {{"token_ids", {7, 8}}, {"config", {{"scale", 2.0}}}});
        auto head = run();
        const auto& head_output = head.at("output_summary");
        check(head.at("observations").size() == 2 &&
                  head.at("observation_serialization_included") == false &&
                  head_output.at("values") == Json::array({-4, 2, 6, -8}) &&
                  head_output.at("shape") == Json::array({1, 2, 2}) &&
                  head_output.at("score_kind") == "logit" &&
                  head_output.at("head_score_values") == 4 &&
                  !head_output.contains("embedding_vectors") && !head_output.contains("tokens"),
              "head score benchmark measures the public call and retains typed raw values/shape");
        request["request"] = {{"prompt", "abc"}, {"config", {{"representation", "unit"}}}};
        bundle(model, "text_query_documents_to_relevance", "features_fixture");
        head = run();
        check(head.at("task") == "text_query_documents_to_relevance" &&
                  head.at("selected_task") == "text_to_head_scores",
              "implicit unique head-score operation records the actual secondary Task");
        check(head.at("output_summary").at("score_kind") == "unbounded" &&
                  head.at("output_summary").at("shape") == Json::array({2}) &&
                  head.at("output_summary").at("normalization") == "l2",
              "secondary head-score binding is callable and preserves family-transformed metadata");
        request["request"]["token_ids"] = {1};
        run(false);
        request["request"] = {{"token_ids", {2147483648LL}}};
        run(false);
        request["request"] = {{"prompt", "x"}, {"config", {{"representation", "unknown"}}}};
        run(false);
        request["request"] = {{"prompt", "abc"}};
        request["selected_task"] = "text_to_pooled_features";
        check(run(false).at("error").get<std::string>().find(
                  "operation cannot execute selected Task") != std::string::npos,
              "a unique head-score operation never ignores an explicit different bound Task");
        request["selected_task"] = "text_to_head_scores";
        check(run().at("selected_task") == "text_to_head_scores",
              "explicit secondary head-score selection also succeeds");
        request.erase("selected_task");

        select("text_to_embedding", "features_fixture", "encode", {{"token_ids", {77, 88}}});
        request["expected_family"] = "features_fixture";
        request["expected_task"] = "text_to_embedding";
        request["selected_task"] = "text_to_token_features";
        auto selected = run();
        check(selected.at("task") == "text_to_embedding" &&
                  selected.at("selected_task") == "text_to_token_features" &&
                  selected.at("output_summary").at("feature_kind") == "token" &&
                  selected.at("output_summary").at("tokens")[0].at("token_id") == 77,
              "one physical embedding bundle can explicitly execute token features");
        request["selected_task"] = "text_to_pooled_features";
        selected = run();
        check(selected.at("task") == "text_to_embedding" &&
                  selected.at("selected_task") == "text_to_pooled_features" &&
                  selected.at("output_summary").at("feature_kind") == "pooled",
              "same primary and operation can explicitly execute a different bound Task");
        request["expected_task"] = "text_to_pooled_features";
        check(run(false).at("error").get<std::string>().find("bundle identity mismatch") !=
                  std::string::npos,
              "selected Task never replaces physical bundle identity validation");
        request["expected_task"] = "text_to_embedding";
        request["selected_task"] = "text_to_embedding";
        run(false); // Explicit embedding cannot take the implicit encode -> pooled alias.
        for (const auto& invalid : {Json("not_a_task"), Json("image_to_metric_geometry"), Json(""),
                                    Json(7), Json(nullptr)}) {
            request["selected_task"] = invalid;
            run(false);
        }
        request.erase("selected_task");
        selected = run();
        check(selected.at("selected_task") == "text_to_pooled_features" &&
                  selected.at("output_summary").at("feature_kind") == "pooled",
              "existing implicit embedding encode alias remains unchanged");
        request.erase("expected_family");
        request.erase("expected_task");
        const auto tracking_frame = runtime / "benchmark_tracking_frame.ppm";
        image(tracking_frame, 3, 128, 2);
        const Json tracking_input{
            {"frame_paths", std::vector<std::string>(5, tracking_frame.string())},
            {"timestamps_seconds", {0.0, 0.2, 0.5, 0.7, 1.0}}};
        select("frames_to_detected_mask_tracks", "tracking_fixture", "track_masks", tracking_input);
        for (const bool include_assets : {false, true}) {
            request["measurement"]["asset_loading_included"] = include_assets;
            const auto value = run().at("output_summary");
            const auto& first = value.at("frames").at(0);
            check(value.at("tracked_frames") == 5 && value.at("mask_elements") == 30 &&
                      first.at("masks") == Json::array({1, 1, 1, 1, 1, 1}) &&
                      first.at("object_ids") == Json::array({7}) &&
                      first.at("class_ids") == Json::array({2}) &&
                      first.at("source_memory") == "host" &&
                      first.at("mask_element_type") == "uint8" &&
                      value.at("frames").at(2).at("timestamp_seconds") == 0.5 &&
                      value.at("initial_detections").at(0).at("prompt_box") ==
                          Json::array({0, 0, 3, 2}),
                  "tracking preserves all host mask pixels, identities, initial detections and "
                  "explicit timestamps");
        }
        request["request"]["device_masks"] = true;
        check(run(false).at("error").get<std::string>().find("CUDA mask") != std::string::npos,
              "CPU fixture CUDA placeholders are rejected by real pointer validation, never copied "
              "as host memory");
        request["request"] = tracking_input;
        request["request"]["frame_paths"].erase(4);
        request["request"].erase("timestamps_seconds");
        run(false);
        for (const auto* task :
             {"frames_text_to_mask_tracks", "prompt_frame_text_to_mask_tracks"}) {
            auto input = tracking_input;
            input["prompt"] = "bird";
            select(task, "tracking_fixture", "track_masks", input);
            const auto value = run().at("output_summary");
            const auto& first = value.at("frames").at(0);
            check(value.at("tracked_frames") == 5 && first.at("mask_element_type") == "float32" &&
                      first.at("object_ids") == Json::array({13}) &&
                      first.at("detection_scores") == Json::array({0.875}) &&
                      first.at("tracking_scores") == Json::array({0.625}) &&
                      first.at("removed_object_ids") == Json::array({19}) &&
                      first.at("suppressed_object_ids") == Json::array({23}),
                  "text tracking preserves float masks and all per-frame metadata");
            if (std::string(task) == "prompt_frame_text_to_mask_tracks")
                check(value.at("prompt_snapshot").at("frames").at(0).at("masks").at(0) == 1.0 &&
                          first.at("masks").at(0) == 0.0 && value.at("frames").size() == 5,
                      "prompt-frame lifecycle retains the original prompt and replaces frame zero "
                      "with consolidation without prepending");
            request["request"]["prompt"] = "cat";
            run(false);
        }
        std::filesystem::remove(tracking_frame);

        std::vector<std::filesystem::path> pose_files;
        auto floats = [&](const std::vector<float>& values) {
            const auto path =
                runtime / ("benchmark_pose_" + std::to_string(pose_files.size()) + ".f32");
            std::ofstream file(path, std::ios::binary);
            file.exceptions(std::ios::badbit | std::ios::failbit);
            file.write(reinterpret_cast<const char*>(values.data()),
                       static_cast<std::streamsize>(values.size() * sizeof(float)));
            file.close();
            pose_files.push_back(path);
            return path.string();
        };
        auto poses = [](std::initializer_list<float> xs, float y = 0) {
            std::vector<float> values(xs.size() * 16, 0);
            std::size_t n = 0;
            for (const auto x : xs) {
                for (int k = 0; k < 4; ++k)
                    values[n * 16 + k * 5] = 1;
                values[n * 16 + 3] = x;
                values[n * 16 + 7] = y;
                ++n;
            }
            return values;
        };
        auto crop = [&](const char* stage, int iteration, const std::vector<float>& query,
                        float rendered, std::initializer_list<float> observed) {
            const auto count = query.size() / 16;
            std::vector<float> a(count * 6, rendered), b(count * 6, 0.25F);
            std::size_t n = 0;
            for (const auto value : observed)
                b[n++ * 6] = value;
            return Json{{"stage", stage},
                        {"iteration", iteration},
                        {"shape", {count, 1, 1, 6}},
                        {"query_poses_path", floats(query)},
                        {"rendered_path", floats(a)},
                        {"observed_path", floats(b)}};
        };
        const auto initial_poses = poses({0.5F, 0.75F});
        const auto candidate_path = floats(initial_poses);
        const Json refine_input{
            {"candidate_poses_path", candidate_path},
            {"hypothesis_count", 2},
            {"mesh_diameter_meters", 2.0},
            {"crop_batches",
             Json::array({crop("refinement", 0, initial_poses, 0.25F, {0.25F, 0.25F}),
                          crop("refinement", 1, poses({1, 1.25F}, 0.25F), 0.5F, {0.25F, 0.25F}),
                          crop("scoring", 2, poses({2, 2.25F}, 0.5F), 0.0F, {0.25F, 0.5F})})},
            {"config", {{"refinement_iterations", 2}, {"score_hypotheses", true}}}};
        select("pose_hypotheses_crops_to_refined_poses", "perception_fixture", "refine_pose",
               refine_input);
        for (const bool include_assets : {false, true}) {
            request["measurement"]["asset_loading_included"] = include_assets;
            const auto value = run().at("output_summary");
            check(value.at("refined_hypotheses") == 2 &&
                      value.at("refined_poses") == poses({2, 2.25F}, 0.5F) &&
                      value.at("scores") == Json::array({1, 2}) && value.at("best_index") == 1 &&
                      value.at("all_poses_rigid") == true && value.at("crop_queries").size() == 3 &&
                      value.at("crop_queries").at(2).at("iteration") == 2 &&
                      value.at("refinement_ms") == 1.5 && value.at("scoring_ms") == 0.5,
                  "pose refinement consumes exact prepared callbacks and preserves full poses, "
                  "scores, selection and timing");
        }
        request["request"]["crop_batches"].erase(2);
        run(false);
        request["request"] = refine_input;
        request["request"]["crop_batches"][0]["iteration"] = 1;
        run(false);
        request["request"] = refine_input;
        request["request"]["crop_batches"].push_back(refine_input.at("crop_batches").back());
        run(false);

        const Json initialize{
            {"candidate_poses_path", candidate_path},
            {"hypothesis_count", 2},
            {"mesh_diameter_meters", 2.0},
            {"crop_batches",
             Json::array({crop("refinement", 0, initial_poses, 0.25F, {0.25F, 0.25F}),
                          crop("scoring", 0, poses({1, 1.25F}), 0.0F, {0.25F, 0.5F})})}};
        const Json pose_tracking{
            {"initialization", initialize},
            {"updates",
             Json::array({{{"crop_batches",
                            Json::array({crop("refinement", 0, poses({1.25F}), 0.5F, {0.25F}),
                                         crop("scoring", 0, poses({2.25F}), 0.0F, {0.75F})})}},
                          {{"crop_batches",
                            Json::array({crop("refinement", 0, poses({2.25F}), 0.25F, {0.25F}),
                                         crop("scoring", 0, poses({2.75F}), 0.0F, {0.5F})})}}})}};
        select("crop_pose_tracking", "tracking_fixture", "track_pose", pose_tracking);
        const auto pose_measurement = run();
        for (const auto& value : pose_measurement.at("observations")) {
            check(value.at("pose_updates") == 2 &&
                      value.at("initialization").at("best_index") == 1 &&
                      value.at("updates").at(0).at("refined_poses") == poses({2.25F}) &&
                      value.at("updates").at(1).at("refined_poses") == poses({2.75F}) &&
                      value.at("lifecycle_scope") == "fresh_create_initialize_track_all_close",
                  "each fresh pose session initializes, tracks from its selected pose and closes "
                  "with owned snapshots");
        }
        request["request"]["updates"][0]["crop_batches"][0]["query_poses_path"] = candidate_path;
        run(false);
        request["request"] = pose_tracking;
        request["request"]["updates"] = Json::array();
        run(false);
        for (const auto& path : pose_files)
            std::filesystem::remove(path);

        select("image_to_metric_geometry", "perception_fixture", "geometry",
               {{"image_path", left.string()}, {"config", {{"fov_x", 42.0}}}});
        std::filesystem::path geometry_prefix(output);
        geometry_prefix.replace_extension(".geometry");
        const auto points_path = geometry_prefix.string() + ".points.f32";
        const auto depth_path = geometry_prefix.string() + ".depth.f32";
        const auto mask_path = geometry_prefix.string() + ".mask.u8";
        for (const bool include_assets : {false, true}) {
            request["measurement"]["asset_loading_included"] = include_assets;
            const auto value = run();
            const auto& geometry = value.at("output_summary");
            check(geometry.at("geometry_pixels") == 2 && geometry.at("valid_pixels") == 1 &&
                      geometry.at("point_shape") == Json::array({1, 2, 3}) &&
                      geometry.at("units") == "meters" &&
                      geometry.at("camera_axes") == Json::array({"right", "down", "forward"}) &&
                      geometry.at("intrinsics_coordinates") == "normalized_uv" &&
                      std::abs(geometry.at("normalized_intrinsics").at(0).at(0).get<double>() -
                               0.42) < 1e-6,
                  "geometry preserves metric axes, normalized intrinsics and actual output grid");
            float points[6]{}, depth[2]{};
            unsigned char mask[2]{};
            std::ifstream(points_path, std::ios::binary)
                .read(reinterpret_cast<char*>(points), sizeof(points));
            std::ifstream(depth_path, std::ios::binary)
                .read(reinterpret_cast<char*>(depth), sizeof(depth));
            std::ifstream(mask_path, std::ios::binary)
                .read(reinterpret_cast<char*>(mask), sizeof(mask));
            check(
                std::isinf(points[0]) && std::isinf(points[1]) && std::isinf(points[2]) &&
                    points[3] == 2 && std::isinf(depth[0]) && depth[1] == 2 && mask[0] == 0 &&
                    mask[1] == 1 && std::filesystem::file_size(points_path) == sizeof(points) &&
                    std::filesystem::file_size(depth_path) == sizeof(depth) &&
                    std::filesystem::file_size(mask_path) == sizeof(mask),
                "raw geometry artifacts retain invalid infinities and authoritative validity mask");
            check(value.at("asset_loading_included") == include_assets &&
                      value.at("observation_serialization_included") == false &&
                      value.at("observations").size() == 2 &&
                      value.at("observations").at(0).at("geometry_images") == 1,
                  "geometry maps are serialized after measured typed calls");
        }
        request["request"]["config"] = {{"fov_x", "wrong"}};
        run(false);
        for (const auto& path : {points_path, depth_path, mask_path})
            std::filesystem::remove(path);

        select("image_to_boxes", "image_boxes_fixture", "detect", {{"image_path", left.string()}});
        for (const bool include_assets : {false, true}) {
            request["measurement"]["asset_loading_included"] = include_assets;
            const auto value = run();
            const auto& detected = value.at("output_summary");
            check(detected.at("detected_images") == 1 && detected.at("detections") == 2 &&
                      detected.at("boxes") == Json::array({-2, 1, 5, 1, 0, 0, 1, 1}) &&
                      detected.at("scores") == Json::array({0.75, 0.0}) &&
                      detected.at("class_ids") == Json::array({42, 7}) &&
                      detected.at("image_height") == 1 && detected.at("image_width") == 2 &&
                      detected.at("coordinates") == "xyxy" && detected.at("units") == "pixels",
                  "image-only detection preserves out-of-frame boxes, zero scores and numeric "
                  "class IDs");
            check(value.at("observations").size() == 2 &&
                      value.at("asset_loading_included") == include_assets,
                  "detection preserves per-call observations and asset timing policy");
        }
        request["request"]["config"] = {{"empty", true}};
        const auto empty_detection = run().at("output_summary");
        check(empty_detection.at("detections") == 0 && empty_detection.at("boxes").empty() &&
                  empty_detection.at("scores").empty() && empty_detection.at("image_width") == 2,
              "empty detections remain a valid result with original image dimensions");
        request["request"]["config"] = {{"score", 0.0}};
        check(run().at("output_summary").at("scores") == Json::array({0.0, 0.0}),
              "explicit zero detector score is not replaced by a default");
        request["request"]["prompt"] = "";
        run(false);
        request["request"].erase("prompt");
        request["request"]["config"] = {{"score", "wrong"}};
        run(false);

        const auto document = runtime / "benchmark_structure.b2rq";
        {
            std::ofstream file(document, std::ios::binary);
            file.write("B2RQ\0\x7f", 6);
        }
        auto read_document = [](const std::string& path) {
            std::ifstream file(path, std::ios::binary);
            if (!file)
                throw std::runtime_error("cannot read structure artifact");
            return std::string{std::istreambuf_iterator<char>(file),
                               std::istreambuf_iterator<char>()};
        };
        select("molecular_document_to_structure", "structure_fixture", "predict_structure",
               {{"document_path", document.string()}, {"source_path", "relative/original.yaml"}});
        for (const bool include_assets : {false, true}) {
            request["measurement"]["asset_loading_included"] = include_assets;
            const auto value = run();
            const auto& structure = value.at("output_summary");
            const auto contents = read_document(structure.at("structure_artifact"));
            const auto metadata = Json::parse(read_document(structure.at("metadata_artifact")));
            check(contents == "data_fixture\n# b2rq:42325251007f\n" &&
                      structure.at("document_bytes") == 6 &&
                      structure.at("structure_bytes") == contents.size() &&
                      structure.at("format") == "mmcif" &&
                      structure.at("input_encoding") == "b2rq" &&
                      structure.at("source_path") == "relative/original.yaml" &&
                      metadata.at("source_path") == "relative/original.yaml" &&
                      metadata.at("seed") == 42 && metadata.at("sampling_steps") == 200,
                  "structure input bytes including NUL, source provenance and family defaults are "
                  "preserved");
            const auto& confidence = structure.at("confidence");
            check(confidence.size() == 8 && confidence.at("plddt") == Json::array({88, 0, 99}) &&
                      confidence.at("complex_plddt") == 66 &&
                      confidence.at("complex_iplddt") == 77 &&
                      std::abs(confidence.at("confidence_score").get<double>() - 0.1) < 1e-6,
                  "all confidence fields retain family units and valid zero entries");
            check(value.at("observations").size() == 2 &&
                      value.at("observations").at(0).at("structures") == 1 &&
                      value.at("asset_loading_included") == include_assets &&
                      value.at("observation_serialization_included") == false,
                  "structure bytes are written after the measured public Task calls");
        }
        request["request"]["input_encoding"] = "yaml";
        request["request"]["source_path"] = "";
        request["request"]["config"] = {{"seed", 0},
                                        {"include_confidence", false},
                                        {"output_format", "pdb"},
                                        {"sampling_steps", 7}};
        auto structure = run().at("output_summary");
        check(structure.at("format") == "pdb" && structure.at("confidence").is_null() &&
                  structure.at("source_path") == "" && structure.at("input_encoding") == "yaml" &&
                  read_document(structure.at("structure_artifact")) ==
                      "HEADER fixture\nREMARK yaml:42325251007f\n",
              "explicit encoding and empty provenance are not inferred again; absent confidence is "
              "null");
        auto metadata = Json::parse(read_document(structure.at("metadata_artifact")));
        check(metadata.at("seed") == 0 && metadata.at("sampling_steps") == 7,
              "structure Config preserves explicit zero and sampling-step values");
        const std::string provenance("source\0path", 11);
        request["request"]["source_path"] = provenance;
        structure = run().at("output_summary");
        const auto raw_metadata = read_document(structure.at("metadata_artifact"));
        check(raw_metadata.find(provenance) != std::string::npos &&
                  raw_metadata.size() == structure.at("metadata_bytes") &&
                  structure.at("source_path") == provenance,
              "metadata result bytes and source-path view are never truncated at NUL");
        request["request"]["source_path"] = "";
        request["request"]["input_encoding"] = "";
        run(false);
        request["request"]["input_encoding"] = "other";
        run(false);
        request["request"]["input_encoding"] = "b2rq";
        request["request"]["seed"] = 0;
        run(false); // A duplicate flat/nested Config value must not be overwritten.
        request["request"].erase("seed");
        request["request"]["config"] = {{"seed", "0"}};
        run(false);
        std::filesystem::path structure_prefix(output);
        structure_prefix.replace_extension(".structure");
        for (const auto* suffix : {".cif", ".pdb", ".metadata.json"})
            std::filesystem::remove(structure_prefix.string() + suffix);
        std::filesystem::remove(document);

        select("series_to_regression_values", "numeric_fixture", "regress",
               {{"past_values", {1, 2, 3, 4}},
                {"shape", {2, 2}},
                {"observed_mask", {1, 0, 1, 1}},
                {"config", {{"scale", 2.0}}}});
        auto summary = run().at("output_summary");
        check(summary.at("kind") == "regression_values" && summary.at("values").size() == 2 &&
                  summary.at("values")[0] == 16 && summary.at("target_count") == 2 &&
                  summary.at("regression_targets") == 2 && summary.at("parameter_elements") == 0 &&
                  summary.at("axes") == Json::array({"target"}) &&
                  summary.at("target_names").empty() && summary.at("target_units").empty() &&
                  !summary.contains("distribution") && !summary.contains("horizon_steps"),
              "deterministic target regression retains full target values, mask and semantic axes");
        request["request"]["batch_size"] = 2;
        run(false);
        request["request"].erase("batch_size");
        request["request"]["config"]["distribution"] = "normal";
        run(false);

        select("series_to_point_forecast", "numeric_fixture", "regress",
               {{"past_values", {1, 2, 3, 4}},
                {"shape", {2, 2}},
                {"observed_mask", {1, 0, 1, 1}},
                {"config", {{"scale", 2.0}}}});
        request["selected_task"] = "series_to_regression_values";
        auto target_values = run();
        check(target_values.at("task") == "series_to_point_forecast" &&
                  target_values.at("selected_task") == "series_to_regression_values" &&
                  target_values.at("output_summary").at("kind") == "regression_values" &&
                  target_values.at("output_summary").at("values")[0] == 16,
              "deterministic regression is selectable independently of physical forecast primary");
        request["selected_task"] = "series_to_regression_distribution";
        request["request"]["config"] = {{"distribution", "normal"}};
        auto target_distribution = run();
        check(target_distribution.at("task") == "series_to_point_forecast" &&
                  target_distribution.at("selected_task") == "series_to_regression_distribution" &&
                  target_distribution.at("output_summary").at("distribution") == "normal",
              "same physical numeric model can select its other regression contract");
        request["selected_task"] = "series_to_point_forecast";
        run(false); // No fallback to distribution when an explicit Task conflicts with regress.
        request.erase("selected_task");

        select("image_to_class_scores", "features_fixture", "classify",
               {{"image_path", left.string()}});
        auto result = run();
        summary = result.at("output_summary");
        check(summary.at("scores") == Json::array({18, 1}) && summary.at("score_kind") == "logit" &&
                  summary.at("top_class") == 0 && summary.at("top_score") == 18 &&
                  summary.at("labels") == Json::array({"left", "right"}),
              "classification retains raw labeled logits without softmax");
        request["request"] = {{"image_path", right.string()}, {"config", {{"scale", 0.0}}}};
        check(run().at("output_summary").at("top_class") == 0,
              "classification ties choose first index");
        request["request"]["batch_size"] = 2;
        run(false);

        select("image_to_token_and_pooled_features", "features_fixture", "extract_features",
               {{"image_path", left.string()}, {"config", {{"scale", 0.5}}}});
        for (bool include_assets : {false, true}) {
            request["measurement"]["asset_loading_included"] = include_assets;
            summary = run().at("output_summary");
            check(summary.at("last_hidden_state") == Json::array({10, 3}) &&
                      summary.at("pooler_output") == Json::array({10, 3, 1}) &&
                      summary.at("feature_elements") == 5 &&
                      summary.at("last_hidden_state_shape") == Json::array({1, 1, 2}) &&
                      summary.at("pooler_output_shape") == Json::array({1, 3}),
                  "one joint feature call returns both arrays with a matching invocation marker");
        }
        for (const auto* task :
             {"image_to_token_features", "image_to_pooled_features", "image_to_spatial_features"}) {
            select(task, "features_fixture", "extract_features", {{"image_path", left.string()}});
            summary = run().at("output_summary");
            check(summary.at("processed_images") == 1 &&
                      summary.at("feature_elements").get<int>() > 0,
                  "distinct image feature representation has actual element count");
            if (std::string(task) == "image_to_token_features")
                check(summary.at("tokens").size() == 3 &&
                          summary.at("tokens").at(2).at("role") == "patch",
                      "image token roles and grid metadata retained");
            else if (std::string(task) == "image_to_spatial_features")
                check(summary.at("maps").at(0).at("shape") == Json::array({2, 1, 1}) &&
                          summary.at("source_to_processed").at("offset_x") == -3,
                      "spatial maps preserve their axes and source-image transform");
        }
        select("text_to_pooled_features", "features_fixture", "encode", {{"prompt", "Hello"}});
        summary = run().at("output_summary");
        check(summary.at("values") == Json::array({3, 5}) && summary.at("dim") == 2 &&
                  summary.at("feature_kind") == "pooled" && summary.at("embedding_vectors") == 1,
              "pooled encoding stays a pooled result rather than token features");
        select("text_to_token_features", "features_fixture", "encode", {{"prompt", "Hello"}});
        summary = run().at("output_summary");
        check(summary.at("shape") == Json::array({1, 2}) && summary.at("dim") == 2 &&
                  summary.at("feature_kind") == "token" && summary.at("tokens").size() == 1,
              "explicit token features retain token metadata and feature dimension");
        request["request"] = {{"token_ids", {77, 88}}, {"config", {{"scale", 2.0}}}};
        summary = run().at("output_summary");
        check(summary.at("values") == Json::array({2, 2}) &&
                  summary.at("tokens").at(0).at("token_id") == 77,
              "token feature benchmark preserves caller token IDs without retokenizing");
        select("text_to_embedding", "features_fixture", "encode", {{"token_ids", {77, 88}}});
        summary = run().at("output_summary");
        check(summary.at("values") == Json::array({3, 2}) && summary.at("feature_kind") == "pooled",
              "embedding bundle's existing encode operation uses its actual pooled Task");
        request["request"]["prompt"] = "";
        run(false);
        select("text_to_embedding", "features_fixture", "embed",
               {{"prompt", "Hello"}, {"role", "document"}});
        summary = run().at("output_summary");
        check(summary.at("values") == Json::array({4, 2}) && summary.contains("embedding_space"),
              "embedding role is typed input and embedding space is not discarded");
        request["request"]["role"] = "unsupported";
        run(false);
        request["request"] = {{"prompt", "Hello"}, {"scale", 2.0}, {"config", {{"scale", 3.0}}}};
        run(false);
        select("text_query_documents_to_relevance", "features_fixture", "rerank",
               {{"query", "q"}, {"documents", {"ab", "c"}}});
        summary = run().at("output_summary");
        check(summary.at("documents") == 2 && summary.at("scores") == Json::array({3102, 3111}) &&
                  summary.at("order") == "input_documents",
              "rerank list is one family call and preserves document order without claiming native "
              "batching");
        request["request"]["documents"] = Json::array();
        check(run().at("output_summary").at("scores").empty(), "empty document list stays empty");
        request["request"]["documents"] = Json::array({17});
        run(false);

        select("image_to_semantic_segmentation", "perception_fixture", "segment",
               {{"image_path", left.string()}});
        summary = run().at("output_summary");
        check(summary.at("mask") == Json::array({255, 5}) && summary.at("ignore_label") == 255 &&
                  summary.at("class_ids") == Json::array({0, 5}) &&
                  summary.at("class_scores").size() == 4,
              "semantic segmentation preserves labels, vocabulary and separate class-score grid");
        select("image_points_to_masks", "perception_fixture", "segment",
               {{"image_path", left.string()}, {"config", {{"benchmark_masks", "first_vs_best"}}}});
        summary = run().at("output_summary");
        check(summary.at("mask") == Json::array({0, 1}) && summary.at("num_masks") == 1 &&
                  summary.at("returned_mask_count") == 2 && summary.at("mask_pixels") == 2 &&
                  summary.at("selected_mask_index") == 0 &&
                  summary.at("iou_scores").at(0) < summary.at("iou_scores").at(1) &&
                  summary.at("masks").at(2) == 1 && summary.at("point").at("x") == 1,
              "center helper takes first family mask, not higher-IoU mask, and thresholds at zero");
        request["request"]["config"]["benchmark_masks"] = "empty";
        summary = run().at("output_summary");
        check(summary.at("mask").empty() && summary.at("num_masks") == 0 &&
                  summary.at("selected_mask_index").is_null() && summary.at("mask_pixels") == 0,
              "empty masks are not replaced by zero images");
        request["request"]["point_x"] = 0.5;
        run(false); // Fixed center helper does not silently ignore explicit point controls.
        request["operation"] = "segment_prompted";
        request["request"] = {{"image_path", left.string()},
                              {"point_x", 0.75},
                              {"point_y", 0.25},
                              {"is_foreground", false}};
        summary = run().at("output_summary");
        check(summary.at("generated_masks") == 2 && summary.at("mask_pixels") == 4 &&
                  summary.at("point").at("x") == 1 && summary.at("point").at("y") == 0 &&
                  summary.at("low_res_logits").at(0) == -7,
              "prompted masks retain all outputs, quantized original-pixel point and foreground "
              "flag");
        request["request"]["is_foreground"] = "false";
        run(false);
        request["request"] = {{"image_path", left.string()}, {"point_x", "0.5"}};
        run(false);
        select("image_text_to_instance_masks", "perception_fixture", "segment_prompted",
               {{"image_path", left.string()}, {"prompt", "object"}});
        summary = run().at("output_summary");
        check(summary.at("confidence").size() == 2 && summary.at("iou_scores").empty() &&
                  summary.at("object_ids") == Json::array({100, 101}),
              "instance confidence is not relabeled as IoU and object identities survive");

        const auto one = runtime / "benchmark_remaining_one.ppm";
        const auto zero = runtime / "benchmark_remaining_zero.ppm";
        image(one, 1, 255);
        image(zero, 1, 0);
        const auto state = runtime / "benchmark_remaining_state.f32";
        {
            const float values[]{1, 2};
            std::ofstream file(state, std::ios::binary);
            file.write(reinterpret_cast<const char*>(values), sizeof(values));
        }
        select("image_state_to_action_chunk", "action_fixture", "control",
               {{"image_path", one.string()},
                {"state_path", state.string()},
                {"config", {{"tag", "bench"}}}});
        summary = run().at("output_summary");
        check(summary.at("action_steps") == 2 && summary.at("action_dim") == 2 &&
                  summary.at("actions") == Json::array({2, -3, 2, 8}) &&
                  summary.at("within_training_bounds") == false &&
                  summary.at("inference_ms") == 103 &&
                  summary.at("schema").at("domain") == "fixture.bench" &&
                  summary.at("schema").at("normalization") == "unnormalized",
              "stateless action chunk retains values, schema, bounds and fresh-call timing");
        bundle(model, "image_state_action_queue", "action_fixture");
        run(false);

        const auto next_state = runtime / "benchmark_remaining_next_state.f32";
        {
            const float values[]{20, 40};
            std::ofstream file(next_state, std::ios::binary);
            file.write(reinterpret_cast<const char*>(values), sizeof(values));
        }
        const Json queue_input{
            {"observations",
             Json::array({{{"image_path", one.string()}, {"state_path", state.string()}},
                          {{"image_path", zero.string()}, {"state_path", next_state.string()}},
                          {{"image_path", one.string()}, {"state_path", next_state.string()}}})},
            {"config", {{"tag", "session"}}}};
        select("image_state_action_queue", "action_fixture", "control_queue", queue_input);
        for (const bool include_assets : {false, true}) {
            request["measurement"]["asset_loading_included"] = include_assets;
            const auto measured = run();
            for (const auto& observation : measured.at("observations")) {
                const auto& steps = observation.at("steps");
                check(observation.at("action_steps") == 3 &&
                          observation.at("lifecycle_scope") == "fresh_create_act_all_close" &&
                          steps.at(0).at("actions") == Json::array({2, -3}) &&
                          steps.at(1).at("actions") == Json::array({2, 8}) &&
                          steps.at(2).at("actions") == Json::array({21, -3}),
                      "action queue uses its queued prediction before refilling from the next "
                      "observation");
                check(steps.at(0).at("inference_ms") == 1 && steps.at(1).at("inference_ms") == 0 &&
                          steps.at(2).at("inference_ms") == 2 &&
                          steps.at(0).at("started_new_chunk") == true &&
                          steps.at(1).at("started_new_chunk") == false &&
                          steps.at(2).at("started_new_chunk") == true,
                      "every measured iteration starts with a fresh queue and releases the "
                      "previous session");
                check(steps.at(0).at("schema").at("domain") == "fixture.session" &&
                          steps.at(0).at("within_training_bounds") == true &&
                          steps.at(1).at("within_training_bounds") == false,
                      "per-step schema, timing and bounds remain owned after session close");
            }
        }
        request["request"]["observations"][0]["config"] = {{"tag", "not-an-act-option"}};
        run(false);
        request["request"] = queue_input;
        request["request"]["observations"] = Json::array();
        run(false);
        request["request"] = queue_input;
        request["request"]["observations"][1].erase("state_path");
        run(false);
        std::filesystem::remove(next_state);

        select("text_to_image", "image_fixture", "generate_image", {{"prompt", "Hello"}});
        summary = run().at("output_summary");
        check(summary.at("generated_images") == 1 && summary.at("generated_frames") == 1 &&
                  summary.at("output_elements") == 3 && summary.at("media_type") == "image",
              "scalar image counts an actual image");
        request["request"]["prompt"] = "benchmark-worker";
        run(false);
        request["request"] = {{"prompt", "Hello"}, {"media_type", "video"}};
        run(false);
        const auto latents = runtime / "benchmark_remaining_latents.f32";
        {
            const float values[]{0.1F, 0.2F, 0.3F};
            std::ofstream file(latents, std::ios::binary);
            file.write(reinterpret_cast<const char*>(values), sizeof(values));
        }
        request["request"] = {{"prompt", "Hello"}, {"initial_latents_path", latents.string()}};
        run();
        select("images_text_to_image_edit", "image_fixture", "generate_image",
               {{"prompt", "Edit"}, {"image_path", one.string()}});
        run();
        request["request"] = {{"prompt", "Edit"},
                              {"image_paths", {one.string(), zero.string()}},
                              {"initial_latents_path", latents.string()}};
        run();
        request["request"]["image_path"] = left.string();
        run(false);
        select("batch_text_to_image", "image_fixture", "generate_image",
               {{"prompt", {"first", "benchmark-wide"}},
                {"seeds", {0, 17}},
                {"batch_size", 2},
                {"item_configs", Json::array({Json::object(), Json{{"level", 0.5}}})}});
        summary = run().at("output_summary");
        check(summary.at("generated_images") == 2 && summary.at("generated_frames") == 2 &&
                  summary.at("output_elements") == 9 &&
                  summary.at("images").at(0).at("width") == 1 &&
                  summary.at("images").at(1).at("width") == 2,
              "one native batch transports exact per-item seed/config and unequal output shapes");
        request["request"]["seed"] = 0;
        run(false);
        request["request"].erase("seed");
        request["request"]["config"] = {{"level", 0.25}};
        run(false);
        request["request"].erase("config");
        request["request"]["seeds"] = {0};
        run(false);
        request["request"] = {{"prompt", {"first", "benchmark-worker"}}};
        run(false);
        request["request"] = {{"prompt", {"first"}}, {"initial_latents_path", latents.string()}};
        run(false);
        request["request"] = {{"prompt", {"first"}}, {"item_configs", Json::array({17})}};
        run(false);

        {
            const float values[]{0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F};
            std::ofstream file(latents, std::ios::binary);
            file.write(reinterpret_cast<const char*>(values), sizeof(values));
        }
        select("text_to_video", "video_fixture", "generate_image",
               {{"prompt", "Hello"},
                {"media_type", "video"},
                {"initial_latents_path", latents.string()}});
        summary = run().at("output_summary");
        check(summary.at("generated_images") == 1 && summary.at("generated_frames") == 3 &&
                  summary.at("output_elements") == 9 &&
                  summary.at("timestamps_seconds") == Json::array({0, 0.1, 0.3}) &&
                  summary.at("media_type") == "video",
              "video retains actual nonuniform timeline, frame count and clip-count metric");
        request["request"] = {{"prompt", "benchmark-worker"}, {"media_type", "video"}};
        run(false);
        select("image_text_action_to_video", "video_fixture", "generate_image",
               {{"prompt", "Drive"},
                {"image_path", one.string()},
                {"action", "forward"},
                {"camera_intrinsics", {100, 100, 0.5, 0.5}},
                {"media_type", "video"}});
        run();
        request["request"]["camera_intrinsics"] = Json::array({100, 0, 0.5, 0, 100, 0.5, 0, 0, 1});
        run();
        request["request"]["camera_intrinsics"] = Json::array({1, 2, 3});
        run(false);
        request["request"]["camera_intrinsics"] = Json::array({true, 100, 0.5, 0.5});
        run(false);
        request["request"]["camera_intrinsics"] = Json::array({100, 100, 0.5, 0.5});
        request["request"]["action"] = "";
        run(false);
        std::cout << (failures ? "FAILED\n" : "ALL PASSED\n");
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
