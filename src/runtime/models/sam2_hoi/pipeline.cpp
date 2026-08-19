/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/sam2_hoi/pipeline.h"

#include "runtime/models/sam2_hoi/hoi_postprocess.h"
#include "runtime/models/sam2_hoi/jpeg_decoder.h"
#include "runtime/models/sam2_hoi/ordered_async_mask_postprocessor.h"
#include "runtime/models/sam2_hoi/pafpn_composite.h"
#include "runtime/models/sam2_hoi/rolling_async_preprocessor.h"
#include "runtime/models/sam2_hoi/sam2_hoi_io.h"
#include "trtmc/trtmc_io.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trtmc::sam2_hoi {
namespace {

constexpr int32_t kObjectBatch = 2;
constexpr int32_t kLowMaskSize = 256;
constexpr int32_t kMemoryChannels = 64;
constexpr int32_t kMemorySpatialSize = 64;
constexpr int32_t kPointerWidth = 256;
constexpr int32_t kMaxMemoryFrames = 7;
constexpr int32_t kMaxPointers = 16;

bool is_jpeg_path(const std::string& path) {
    std::string extension = std::filesystem::path(path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
    return extension == ".jpg" || extension == ".jpeg";
}

constexpr std::size_t kMaskValues =
    static_cast<std::size_t>(kObjectBatch) * kLowMaskSize * kLowMaskSize;
constexpr std::size_t kPointerValues = static_cast<std::size_t>(kObjectBatch) * kPointerWidth;
constexpr std::size_t kMemoryValuesPerObject =
    static_cast<std::size_t>(kMemoryChannels) * kMemorySpatialSize * kMemorySpatialSize;

struct TrackerOutput {
    std::vector<float> masks;
    std::vector<float> pointers;
    std::vector<float> object_scores;
};

struct MemoryRecord {
    std::vector<std::uint8_t> features;
    std::vector<std::uint8_t> position;
    DType dtype{DType::kFloat32};
};

struct FrameResult {
    HoiPostprocessResult hoi;
    std::vector<int32_t> object_ids;
    std::vector<std::uint8_t> binary_masks;
    int32_t frame_index{0};
    int32_t height{0};
    int32_t width{0};
};

const Tensor& require_output(const TensorMap& outputs, const std::string& name) {
    const auto iterator = outputs.find(name);
    if (iterator == outputs.end() || iterator->second.data == nullptr)
        throw std::runtime_error("SAM2 HOI engine did not produce output '" + name + "'");
    return iterator->second;
}

void require_float_output(const TensorMap& outputs, const std::string& name,
                          std::size_t expected_values, std::vector<float>& destination) {
    const auto& output = require_output(outputs, name);
    if (output.dtype != DType::kFloat32 ||
        static_cast<std::size_t>(output.numel()) != expected_values) {
        throw std::runtime_error("SAM2 HOI output '" + name + "' has an invalid contract");
    }
    const auto* values = static_cast<const float*>(output.data);
    destination.assign(values, values + expected_values);
}

std::vector<std::uint8_t> copy_raw_output(const TensorMap& outputs, const std::string& name,
                                          DType& dtype, std::size_t expected_values) {
    const auto& output = require_output(outputs, name);
    if (static_cast<std::size_t>(output.numel()) != expected_values)
        throw std::runtime_error("SAM2 HOI output '" + name + "' has an invalid shape");
    dtype = output.dtype;
    std::vector<std::uint8_t> bytes(output.nbytes());
    std::memcpy(bytes.data(), output.data, bytes.size());
    return bytes;
}

void bind_output_to_input(ITrtModule& producer, const char* output_name, ITrtModule& consumer,
                          const char* input_name) {
    if (!producer.has_output(output_name) || !consumer.has_input(input_name))
        throw std::runtime_error(std::string("SAM2 HOI split-plan binding is missing: ") +
                                 output_name + " -> " + input_name);
    const auto shape = producer.tensor_shape(output_name);
    if (shape.empty() || shape != consumer.tensor_shape(input_name) ||
        producer.tensor_dtype(output_name) != consumer.tensor_dtype(input_name)) {
        throw std::runtime_error(std::string("SAM2 HOI split-plan binding is incompatible: ") +
                                 output_name + " -> " + input_name);
    }
    void* pointer = producer.device_ptr(output_name);
    if (pointer == nullptr)
        throw std::runtime_error(std::string("SAM2 HOI producer has no device buffer for '") +
                                 output_name + "'");
    consumer.bind_external(input_name, pointer, shape);
    if (consumer.device_ptr(input_name) != pointer)
        throw std::runtime_error(std::string("SAM2 HOI split-plan binding failed: ") + output_name +
                                 " -> " + input_name);
}

std::vector<float> interaction_probabilities(ITrtModule& interaction,
                                             const std::vector<float>& pair_embeddings,
                                             std::size_t pair_count) {
    if (pair_count == 0)
        return {};
    if (pair_embeddings.size() != pair_count * 2 * kHoiEmbeddingSize)
        throw std::runtime_error("SAM2 HOI interaction input has an invalid size");
    Tensor pairs;
    pairs.data = const_cast<float*>(pair_embeddings.data());
    pairs.shape = {static_cast<int64_t>(pair_count), 2 * static_cast<int64_t>(kHoiEmbeddingSize)};
    pairs.dtype = DType::kFloat32;
    const TensorMap outputs = interaction.forward({{"pair_features", pairs}});
    const auto iterator = outputs.find("interaction_probabilities");
    if (iterator == outputs.end() || iterator->second.data == nullptr ||
        iterator->second.dtype != DType::kFloat32 || iterator->second.numel() != pair_count * 2) {
        throw std::runtime_error("SAM2 HOI interaction engine returned an invalid output");
    }
    const auto* values = static_cast<const float*>(iterator->second.data);
    std::vector<float> probabilities(pair_count);
    for (std::size_t index = 0; index < pair_count; ++index)
        probabilities[index] = values[index * 2 + 1];
    return probabilities;
}

HoiPostprocessResult run_detector(ITrtModule& detector, ITrtModule& interaction) {
    const auto outputs = detector.forward({});
    std::vector<float> scores;
    std::vector<float> boxes;
    std::vector<float> embeddings;
    require_float_output(outputs, "class_scores", kHoiQueryCount * kHoiClassCount, scores);
    require_float_output(outputs, "boxes_cxcywh", kHoiQueryCount * 4, boxes);
    require_float_output(outputs, "query_embeddings", kHoiQueryCount * kHoiEmbeddingSize,
                         embeddings);

    HoiPostprocessResult result;
    const auto status = postprocess_hoi(
        scores, boxes, embeddings,
        [&](const std::vector<float>& pair_embeddings, std::size_t pair_count) {
            return interaction_probabilities(interaction, pair_embeddings, pair_count);
        },
        result);
    if (status != HoiPostprocessStatus::kOk) {
        throw std::runtime_error("SAM2 HOI detector postprocess failed with status " +
                                 std::to_string(static_cast<int>(status)));
    }
    return result;
}

std::vector<int32_t> selected_prompt_detection_indices(const HoiPostprocessResult& hoi) {
    std::set<int32_t> paired;
    for (const auto& pair : hoi.interaction_pairs) {
        paired.insert(pair.source_detection_index);
        paired.insert(pair.target_detection_index);
    }
    std::vector<int32_t> selected;
    for (std::size_t index = 0; index < hoi.detections.size(); ++index) {
        if (hoi.detections[index].label <= 1 || paired.count(static_cast<int32_t>(index)) != 0)
            selected.push_back(static_cast<int32_t>(index));
    }
    return selected;
}

TrackerOutput run_prompt_tracker(ITrtModule& tracker, const HoiPostprocessResult& hoi,
                                 const std::vector<int32_t>& selected) {
    if (selected.size() != kObjectBatch)
        throw std::runtime_error("SAM2 HOI fixed B2 prompt engine requires exactly two prompts");
    std::array<float, kObjectBatch * 3 * 2> point_coordinates{};
    std::array<int32_t, kObjectBatch * 3> point_labels{};
    for (int32_t object = 0; object < kObjectBatch; ++object) {
        const auto detection_index = static_cast<std::size_t>(selected[object]);
        if (detection_index >= hoi.detections.size())
            throw std::runtime_error("SAM2 HOI selected prompt index is out of range");
        const auto& box = hoi.detections[detection_index].box_xyxy;
        const std::size_t coordinate_offset = static_cast<std::size_t>(object) * 6;
        point_coordinates[coordinate_offset] = box[0];
        point_coordinates[coordinate_offset + 1] = box[1];
        point_coordinates[coordinate_offset + 2] = box[2];
        point_coordinates[coordinate_offset + 3] = box[3];
        point_coordinates[coordinate_offset + 4] = (box[0] + box[2]) * 0.5F;
        point_coordinates[coordinate_offset + 5] = (box[1] + box[3]) * 0.5F;
        const std::size_t label_offset = static_cast<std::size_t>(object) * 3;
        point_labels[label_offset] = 2;
        point_labels[label_offset + 1] = 3;
        point_labels[label_offset + 2] = 1;
    }

    TensorMap inputs{
        {"point_coords", Tensor{point_coordinates.data(), {kObjectBatch, 3, 2}, DType::kFloat32}},
        {"point_labels", Tensor{point_labels.data(), {kObjectBatch, 3}, DType::kInt32}},
    };
    const auto outputs = tracker.forward(inputs);
    TrackerOutput result;
    require_float_output(outputs, "pred_masks", kMaskValues, result.masks);
    require_float_output(outputs, "object_pointer", kPointerValues, result.pointers);
    require_float_output(outputs, "object_score_logits", kObjectBatch, result.object_scores);
    return result;
}

std::vector<const MemoryRecord*> select_spatial_memories(const std::vector<MemoryRecord>& records,
                                                         int32_t frame_index,
                                                         std::vector<int32_t>& temporal_positions) {
    if (frame_index <= 0 || records.empty())
        throw std::runtime_error("SAM2 HOI recurrent frame has no conditioning memory");
    std::vector<const MemoryRecord*> selected{&records.front()};
    temporal_positions = {0};
    const int32_t first_non_conditioning = std::max<int32_t>(1, frame_index - 6);
    for (int32_t prior = first_non_conditioning; prior < frame_index; ++prior) {
        if (static_cast<std::size_t>(prior) >= records.size())
            throw std::runtime_error("SAM2 HOI memory bank is incomplete");
        selected.push_back(&records[static_cast<std::size_t>(prior)]);
        temporal_positions.push_back(kMaxMemoryFrames - (frame_index - prior));
    }
    return selected;
}

std::vector<std::uint8_t> pack_memory_rows(const std::vector<const MemoryRecord*>& records,
                                           bool position, DType expected_dtype) {
    if (records.empty())
        throw std::runtime_error("SAM2 HOI cannot pack an empty memory bank");
    const std::size_t bytes_per_object = kMemoryValuesPerObject * dtype_size(expected_dtype);
    const std::size_t memory_count = records.size();
    std::vector<std::uint8_t> packed(static_cast<std::size_t>(kObjectBatch) * memory_count *
                                     bytes_per_object);
    for (int32_t object = 0; object < kObjectBatch; ++object) {
        for (std::size_t memory = 0; memory < memory_count; ++memory) {
            if (records[memory]->dtype != expected_dtype)
                throw std::runtime_error("SAM2 HOI memory dtype does not match recurrent input");
            const auto& source = position ? records[memory]->position : records[memory]->features;
            if (source.size() != static_cast<std::size_t>(kObjectBatch) * bytes_per_object)
                throw std::runtime_error("SAM2 HOI stored memory has an invalid byte size");
            const std::size_t source_offset = static_cast<std::size_t>(object) * bytes_per_object;
            const std::size_t destination_offset =
                (static_cast<std::size_t>(object) * memory_count + memory) * bytes_per_object;
            std::memcpy(packed.data() + destination_offset, source.data() + source_offset,
                        bytes_per_object);
        }
    }
    return packed;
}

std::vector<const std::vector<float>*>
select_pointers(const std::vector<std::vector<float>>& pointer_records, int32_t frame_index,
                std::vector<float>& temporal_offsets) {
    if (frame_index <= 0 || pointer_records.empty())
        throw std::runtime_error("SAM2 HOI recurrent frame has no conditioning pointer");
    std::vector<const std::vector<float>*> selected{&pointer_records.front()};
    temporal_offsets = {static_cast<float>(frame_index)};
    const int32_t available = std::min<int32_t>(frame_index - 1, kMaxPointers - 1);
    for (int32_t difference = 1; difference <= available; ++difference) {
        const int32_t prior = frame_index - difference;
        selected.push_back(&pointer_records[static_cast<std::size_t>(prior)]);
        temporal_offsets.push_back(static_cast<float>(difference));
    }
    return selected;
}

std::vector<float>
pack_pointer_rows(const std::vector<const std::vector<float>*>& pointer_records) {
    const std::size_t pointer_count = pointer_records.size();
    std::vector<float> packed(static_cast<std::size_t>(kObjectBatch) * pointer_count *
                              kPointerWidth);
    for (int32_t object = 0; object < kObjectBatch; ++object) {
        for (std::size_t pointer = 0; pointer < pointer_count; ++pointer) {
            const auto& source = *pointer_records[pointer];
            if (source.size() != kPointerValues)
                throw std::runtime_error("SAM2 HOI stored object pointer has an invalid size");
            const auto* begin = source.data() + static_cast<std::size_t>(object) * kPointerWidth;
            auto* destination =
                packed.data() +
                (static_cast<std::size_t>(object) * pointer_count + pointer) * kPointerWidth;
            std::copy(begin, begin + kPointerWidth, destination);
        }
    }
    return packed;
}

TrackerOutput run_recurrent_tracker(ITrtModule& tracker,
                                    const std::vector<MemoryRecord>& memory_records,
                                    const std::vector<std::vector<float>>& pointer_records,
                                    int32_t frame_index, int32_t total_frames) {
    std::vector<int32_t> memory_temporal_positions;
    const auto memories =
        select_spatial_memories(memory_records, frame_index, memory_temporal_positions);
    const DType memory_dtype = tracker.tensor_dtype("memory_features");
    if (memory_dtype != tracker.tensor_dtype("memory_position"))
        throw std::runtime_error("SAM2 HOI recurrent memory inputs have inconsistent dtypes");
    auto packed_features = pack_memory_rows(memories, false, memory_dtype);
    auto packed_position = pack_memory_rows(memories, true, memory_dtype);
    const int64_t memory_count = static_cast<int64_t>(memories.size());
    std::vector<int32_t> memory_offsets(static_cast<std::size_t>(kObjectBatch) * memories.size());
    for (int32_t object = 0; object < kObjectBatch; ++object) {
        std::copy(memory_temporal_positions.begin(), memory_temporal_positions.end(),
                  memory_offsets.begin() + static_cast<std::ptrdiff_t>(object * memory_count));
    }

    std::vector<float> pointer_temporal_offsets;
    const auto pointers = select_pointers(pointer_records, frame_index, pointer_temporal_offsets);
    auto packed_pointers = pack_pointer_rows(pointers);
    const int64_t pointer_count = static_cast<int64_t>(pointers.size());
    std::vector<float> pointer_offsets(static_cast<std::size_t>(kObjectBatch) * pointers.size());
    for (int32_t object = 0; object < kObjectBatch; ++object) {
        std::copy(pointer_temporal_offsets.begin(), pointer_temporal_offsets.end(),
                  pointer_offsets.begin() + static_cast<std::ptrdiff_t>(object * pointer_count));
    }
    const std::array<float, 1> denominator{
        static_cast<float>(std::min(total_frames, kMaxPointers) - 1)};
    if (!(denominator[0] > 0.0F))
        throw std::runtime_error("SAM2 HOI recurrent pointer denominator must be positive");

    TensorMap inputs{
        {"memory_features", Tensor{packed_features.data(),
                                   {kObjectBatch, memory_count, kMemoryChannels, kMemorySpatialSize,
                                    kMemorySpatialSize},
                                   memory_dtype}},
        {"memory_position", Tensor{packed_position.data(),
                                   {kObjectBatch, memory_count, kMemoryChannels, kMemorySpatialSize,
                                    kMemorySpatialSize},
                                   memory_dtype}},
        {"memory_temporal_offsets",
         Tensor{memory_offsets.data(), {kObjectBatch, memory_count}, DType::kInt32}},
        {"object_pointers", Tensor{packed_pointers.data(),
                                   {kObjectBatch, pointer_count, kPointerWidth},
                                   DType::kFloat32}},
        {"object_pointer_temporal_offsets",
         Tensor{pointer_offsets.data(), {kObjectBatch, pointer_count}, DType::kFloat32}},
        {"object_pointer_time_denominator",
         Tensor{const_cast<float*>(denominator.data()), {1}, DType::kFloat32}},
    };
    const auto outputs = tracker.forward(inputs);
    TrackerOutput result;
    require_float_output(outputs, "pred_masks", kMaskValues, result.masks);
    require_float_output(outputs, "object_pointer", kPointerValues, result.pointers);
    require_float_output(outputs, "object_score_logits", kObjectBatch, result.object_scores);
    return result;
}

MemoryRecord encode_memory(ITrtModule& encoder, const std::vector<float>& masks,
                           const std::vector<float>& object_scores, bool from_points) {
    if (masks.size() != kMaskValues || object_scores.size() != kObjectBatch)
        throw std::runtime_error("SAM2 HOI memory encoder input has an invalid shape");
    const std::array<int32_t, kObjectBatch> point_flags{
        from_points ? 1 : 0,
        from_points ? 1 : 0,
    };
    TensorMap inputs{
        {"pred_masks", Tensor{const_cast<float*>(masks.data()),
                              {kObjectBatch, 1, kLowMaskSize, kLowMaskSize},
                              DType::kFloat32}},
        {"object_score_logits",
         Tensor{const_cast<float*>(object_scores.data()), {kObjectBatch, 1}, DType::kFloat32}},
        {"is_mask_from_points",
         Tensor{const_cast<int32_t*>(point_flags.data()), {kObjectBatch, 1}, DType::kInt32}},
    };
    const auto outputs = encoder.forward(inputs);
    MemoryRecord record;
    DType feature_dtype = DType::kFloat32;
    DType position_dtype = DType::kFloat32;
    const std::size_t expected_values =
        static_cast<std::size_t>(kObjectBatch) * kMemoryValuesPerObject;
    record.features =
        copy_raw_output(outputs, "new_memory_features", feature_dtype, expected_values);
    record.position =
        copy_raw_output(outputs, "new_memory_position", position_dtype, expected_values);
    if (feature_dtype != position_dtype)
        throw std::runtime_error("SAM2 HOI memory encoder outputs have inconsistent dtypes");
    record.dtype = feature_dtype;
    return record;
}

std::string escape_json(const std::string& value) {
    std::ostringstream escaped;
    for (const unsigned char character : value) {
        switch (character) {
        case '"':
            escaped << "\\\"";
            break;
        case '\\':
            escaped << "\\\\";
            break;
        case '\b':
            escaped << "\\b";
            break;
        case '\f':
            escaped << "\\f";
            break;
        case '\n':
            escaped << "\\n";
            break;
        case '\r':
            escaped << "\\r";
            break;
        case '\t':
            escaped << "\\t";
            break;
        default:
            if (character < 0x20U) {
                escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(character) << std::dec << std::setfill(' ');
            } else {
                escaped << static_cast<char>(character);
            }
        }
    }
    return escaped.str();
}

template <typename Value>
void write_scalar_array(std::ostream& output, const std::vector<Value>& values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0)
            output << ',';
        output << values[index];
    }
    output << ']';
}

void write_frame_json(std::ostream& output, const FrameResult& frame,
                      const std::string& mask_path) {
    output << "    {\n";
    output << "      \"frame_index\": " << frame.frame_index << ",\n";
    output << "      \"object_ids\": ";
    write_scalar_array(output, frame.object_ids);
    output << ",\n      \"binary_masks_path\": \"" << escape_json(mask_path) << "\",\n";
    output << "      \"det_bboxes\": [";
    const float scale_x = static_cast<float>(frame.width) / kImageSize;
    const float scale_y = static_cast<float>(frame.height) / kImageSize;
    for (std::size_t index = 0; index < frame.hoi.detections.size(); ++index) {
        if (index != 0)
            output << ',';
        const auto& box = frame.hoi.detections[index].box_xyxy;
        output << '[' << box[0] * scale_x << ',' << box[1] * scale_y << ',' << box[2] * scale_x
               << ',' << box[3] * scale_y << ']';
    }
    output << "],\n      \"det_labels\": [";
    for (std::size_t index = 0; index < frame.hoi.detections.size(); ++index) {
        if (index != 0)
            output << ',';
        output << frame.hoi.detections[index].label;
    }
    output << "],\n      \"det_scores\": [";
    for (std::size_t index = 0; index < frame.hoi.detections.size(); ++index) {
        if (index != 0)
            output << ',';
        output << frame.hoi.detections[index].score;
    }
    output << "],\n      \"interaction_pairs\": [";
    for (std::size_t index = 0; index < frame.hoi.interaction_pairs.size(); ++index) {
        if (index != 0)
            output << ',';
        const auto& pair = frame.hoi.interaction_pairs[index];
        output << '[' << pair.source_detection_index << ',' << pair.target_detection_index << ']';
    }
    output << "]\n    }";
}

void write_results(const std::vector<FrameResult>& results, const std::string& output_json,
                   const std::string& output_masks_dir) {
    namespace fs = std::filesystem;
    std::error_code error;
    const fs::path mask_root(output_masks_dir);
    fs::create_directories(mask_root, error);
    if (error)
        throw std::runtime_error("SAM2 HOI could not create mask output directory: " +
                                 error.message());
    const fs::path json_path(output_json);
    if (json_path.has_parent_path()) {
        fs::create_directories(json_path.parent_path(), error);
        if (error)
            throw std::runtime_error("SAM2 HOI could not create JSON output directory: " +
                                     error.message());
    }

    std::vector<std::string> mask_paths;
    mask_paths.reserve(results.size());
    for (std::size_t index = 0; index < results.size(); ++index) {
        std::ostringstream filename;
        filename << "frame_" << std::setw(6) << std::setfill('0') << results[index].frame_index
                 << ".npy";
        const fs::path path = fs::absolute(mask_root / filename.str());
        std::string detail;
        if (!write_uint8_npy(path.string(), results[index].binary_masks, kObjectBatch,
                             results[index].height, results[index].width, &detail)) {
            throw std::runtime_error("SAM2 HOI could not write mask output: " + detail);
        }
        mask_paths.push_back(path.string());
    }

    std::ofstream output(json_path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("SAM2 HOI could not open output JSON: " + output_json);
    output << std::setprecision(std::numeric_limits<float>::max_digits10);
    output << "{\n  \"schema_version\": 1,\n  \"frames\": [\n";
    for (std::size_t index = 0; index < results.size(); ++index) {
        if (index != 0)
            output << ",\n";
        write_frame_json(output, results[index], mask_paths[index]);
    }
    output << "\n  ]\n}\n";
    if (!output)
        throw std::runtime_error("SAM2 HOI failed while writing output JSON: " + output_json);
}

std::filesystem::path normalized_output_path(const std::filesystem::path& path) {
    std::error_code error;
    auto normalized = std::filesystem::weakly_canonical(path, error);
    if (!error)
        return normalized;
    error.clear();
    normalized = std::filesystem::absolute(path, error);
    return error ? path.lexically_normal() : normalized.lexically_normal();
}

void validate_output_paths(const std::string& output_json, const std::string& output_masks_dir,
                           std::size_t input_frame_count) {
    const auto json_path = normalized_output_path(output_json);
    const auto mask_root = normalized_output_path(output_masks_dir);
    if (json_path == mask_root)
        throw std::invalid_argument("SAM2 HOI JSON output must not replace the mask directory");
    for (std::size_t frame_index = 0; frame_index < input_frame_count; ++frame_index) {
        std::ostringstream filename;
        filename << "frame_" << std::setw(6) << std::setfill('0') << frame_index << ".npy";
        if (json_path == normalized_output_path(mask_root / filename.str())) {
            throw std::invalid_argument("SAM2 HOI JSON output must not replace a generated mask");
        }
    }
}

void validate_pipeline_modules(ITrtModule& image, ITrtModule& detector, ITrtModule& interaction,
                               ITrtModule& prompt_tracker, ITrtModule& recurrent_tracker,
                               ITrtModule& memory_encoder) {
    for (const auto* module :
         {&image, &detector, &interaction, &prompt_tracker, &recurrent_tracker, &memory_encoder}) {
        if (!module->ok() || module->stream() == nullptr)
            throw std::runtime_error("SAM2 HOI pipeline received an invalid TensorRT module");
    }
}

void bind_tracker_inputs(ITrtModule& image, ITrtModule& prompt_tracker,
                         ITrtModule& recurrent_tracker, ITrtModule& memory_encoder) {
    for (const char* name : {"tracker_feature_0", "tracker_feature_1", "tracker_feature_2"}) {
        bind_output_to_input(image, name, prompt_tracker, name);
        bind_output_to_input(image, name, recurrent_tracker, name);
    }
    bind_output_to_input(image, "tracker_position_2", recurrent_tracker, "tracker_position_2");
    bind_output_to_input(image, "tracker_feature_2", memory_encoder, "tracker_feature_2");
}

void require_dependencies(std::initializer_list<bool> dependencies, const char* message) {
    if (std::any_of(dependencies.begin(), dependencies.end(),
                    [](bool present) { return !present; }))
        throw std::runtime_error(message);
}

void validate_phase_a_streams(ITrtModule& image_front, IPafpnComposite& pafpn,
                              ITrtModule& detector) {
    if (image_front.stream() != pafpn.stream() || detector.stream() != pafpn.stream())
        throw std::runtime_error("SAM2 HOI front/PAFPN/detector stream contract failed");
}

void validate_video_frame(const Sam2HoiVideoFrameView& frame, int32_t height, int32_t width) {
    if (frame.pixels == nullptr || frame.height <= 0 || frame.width <= 0)
        throw std::invalid_argument("SAM2 HOI decoded frame is invalid");
    if (frame.height != height || frame.width != width)
        throw std::invalid_argument("SAM2 HOI video frames must have a fixed resolution");
}

bool validate_video_request(const std::vector<Sam2HoiVideoFrameView>& frames,
                            const std::string& output_json, const std::string& output_masks_dir) {
    if (frames.empty())
        throw std::invalid_argument("SAM2 HOI video must contain at least one frame");
    if (frames.size() > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
        throw std::invalid_argument("SAM2 HOI video frame count exceeds the runtime limit");
    const bool discard_outputs = output_json.empty() && output_masks_dir.empty();
    if (output_json.empty() != output_masks_dir.empty()) {
        throw std::invalid_argument(
            "SAM2 HOI output paths must both be empty or both be non-empty");
    }
    if (!discard_outputs)
        validate_output_paths(output_json, output_masks_dir, frames.size());
    for (const auto& frame : frames)
        validate_video_frame(frame, frames.front().height, frames.front().width);
    return discard_outputs;
}

void run_image_front(ITrtModule& image_front, IPafpnComposite* pafpn, std::vector<float>& pixels) {
    Tensor image{pixels.data(), {1, 3, kImageSize, kImageSize}, DType::kFloat32};
    image_front.forward_async({{"pixel_values", image}});
    if (pafpn != nullptr)
        pafpn->forward_async();
    else
        image_front.sync();
}

TrackerOutput run_tracker_frame(ITrtModule& prompt_tracker, ITrtModule& recurrent_tracker,
                                const HoiPostprocessResult& hoi,
                                const std::vector<MemoryRecord>& memory_records,
                                const std::vector<std::vector<float>>& pointer_records,
                                int32_t tracking_frame_index, int32_t total_frames,
                                bool& tracking_started, std::vector<int32_t>& object_ids) {
    TrackerOutput output;
    if (!tracking_started) {
        object_ids = selected_prompt_detection_indices(hoi);
        output = run_prompt_tracker(prompt_tracker, hoi, object_ids);
        tracking_started = true;
    } else {
        output = run_recurrent_tracker(recurrent_tracker, memory_records, pointer_records,
                                       tracking_frame_index, total_frames);
    }
    if (object_ids.size() != kObjectBatch)
        throw std::runtime_error("SAM2 HOI prompt frame did not select two tracked objects");
    return output;
}

std::shared_ptr<std::vector<float>>
prepare_postprocess_masks(TrackerOutput& tracker_output, bool prompt_frame, bool needs_memory) {
    std::shared_ptr<std::vector<float>> masks;
    if (prompt_frame || !needs_memory)
        masks = std::make_shared<std::vector<float>>(std::move(tracker_output.masks));
    else
        masks = std::make_shared<std::vector<float>>(tracker_output.masks);
    if (prompt_frame)
        fill_small_mask_holes(*masks, kObjectBatch, kLowMaskSize, kLowMaskSize);
    return masks;
}

void collect_mask_result(OrderedAsyncMaskPostprocessor& postprocessor,
                         std::vector<FrameResult>& results) {
    auto completed = postprocessor.take_next();
    if (completed.index >= results.size())
        throw std::logic_error("SAM2 HOI mask postprocess returned an invalid result index");
    results[completed.index].binary_masks = std::move(completed.output);
}

void submit_mask_result(OrderedAsyncMaskPostprocessor& postprocessor, std::size_t result_index,
                        const std::shared_ptr<std::vector<float>>& masks, bool prompt_frame,
                        int32_t height, int32_t width) {
    postprocessor.submit(
        result_index, [masks, fill_async = !prompt_frame, height, width]() mutable {
            if (fill_async) {
                fill_small_mask_holes(*masks, kObjectBatch, kLowMaskSize, kLowMaskSize);
            }
            return resize_and_threshold_masks(masks->data(), kObjectBatch, kLowMaskSize,
                                              kLowMaskSize, height, width, 0.01F);
        });
}

void record_tracker_memory(ITrtModule& memory_encoder, TrackerOutput& tracker_output,
                           const std::shared_ptr<std::vector<float>>& postprocess_masks,
                           bool prompt_frame, bool needs_memory,
                           std::vector<MemoryRecord>& memory_records,
                           std::vector<std::vector<float>>& pointer_records) {
    if (!needs_memory)
        return;
    const std::vector<float>& memory_masks =
        prompt_frame ? *postprocess_masks : tracker_output.masks;
    memory_records.push_back(
        encode_memory(memory_encoder, memory_masks, tracker_output.object_scores, prompt_frame));
    pointer_records.push_back(std::move(tracker_output.pointers));
}

void process_video_frame(std::size_t input_frame_index, std::size_t total_frames,
                         const Sam2HoiVideoFrameView& frame, std::vector<float>& pixels,
                         ITrtModule& image_front, IPafpnComposite* pafpn, ITrtModule& detector,
                         ITrtModule& interaction, ITrtModule& prompt_tracker,
                         ITrtModule& recurrent_tracker, ITrtModule& memory_encoder,
                         bool& tracking_started, std::vector<int32_t>& object_ids,
                         std::vector<MemoryRecord>& memory_records,
                         std::vector<std::vector<float>>& pointer_records,
                         OrderedAsyncMaskPostprocessor& mask_postprocessor,
                         std::vector<FrameResult>& results) {
    run_image_front(image_front, pafpn, pixels);
    FrameResult frame_result;
    frame_result.hoi = run_detector(detector, interaction);
    if (!tracking_started && frame_result.hoi.detections.empty())
        return;
    frame_result.frame_index = static_cast<int32_t>(input_frame_index);
    frame_result.height = frame.height;
    frame_result.width = frame.width;

    const int32_t tracking_frame_index = static_cast<int32_t>(results.size());
    auto tracker_output = run_tracker_frame(
        prompt_tracker, recurrent_tracker, frame_result.hoi, memory_records, pointer_records,
        tracking_frame_index, static_cast<int32_t>(total_frames), tracking_started, object_ids);
    frame_result.object_ids = object_ids;

    const bool prompt_frame = tracking_frame_index == 0;
    const bool needs_memory = input_frame_index + 1 < total_frames;
    auto postprocess_masks = prepare_postprocess_masks(tracker_output, prompt_frame, needs_memory);
    if (mask_postprocessor.full())
        collect_mask_result(mask_postprocessor, results);
    const std::size_t result_index = results.size();
    results.push_back(std::move(frame_result));
    submit_mask_result(mask_postprocessor, result_index, postprocess_masks, prompt_frame,
                       frame.height, frame.width);
    record_tracker_memory(memory_encoder, tracker_output, postprocess_masks, prompt_frame,
                          needs_memory, memory_records, pointer_records);
}

} // namespace

void validateVideoOutputPaths(const std::string& output_json, const std::string& output_masks_dir,
                              std::size_t input_frame_count) {
    validate_output_paths(output_json, output_masks_dir, input_frame_count);
}

Sam2HoiPipeline::Sam2HoiPipeline(std::unique_ptr<ITrtModule> image_features,
                                 std::unique_ptr<ITrtModule> detector,
                                 std::unique_ptr<ITrtModule> interaction,
                                 std::unique_ptr<ITrtModule> prompt_tracker,
                                 std::unique_ptr<ITrtModule> recurrent_tracker,
                                 std::unique_ptr<ITrtModule> memory_encoder, std::string model_id)
    : image_front_(std::move(image_features)), detector_(std::move(detector)),
      interaction_(std::move(interaction)), prompt_tracker_(std::move(prompt_tracker)),
      recurrent_tracker_(std::move(recurrent_tracker)), memory_encoder_(std::move(memory_encoder)),
      model_id_(std::move(model_id)) {
    require_dependencies({image_front_ != nullptr, detector_ != nullptr, interaction_ != nullptr,
                          prompt_tracker_ != nullptr, recurrent_tracker_ != nullptr,
                          memory_encoder_ != nullptr},
                         "SAM2 HOI pipeline received a null TensorRT module");
    validate_pipeline_modules(*image_front_, *detector_, *interaction_, *prompt_tracker_,
                              *recurrent_tracker_, *memory_encoder_);
    for (const char* name : {"detector_feature_0", "detector_feature_1", "detector_feature_2"})
        bind_output_to_input(*image_front_, name, *detector_, name);
    bind_tracker_inputs(*image_front_, *prompt_tracker_, *recurrent_tracker_, *memory_encoder_);
}

Sam2HoiPipeline::Sam2HoiPipeline(std::shared_ptr<void> image_stream_owner,
                                 std::unique_ptr<ITrtModule> image_front,
                                 std::unique_ptr<IPafpnComposite> pafpn,
                                 std::unique_ptr<ITrtModule> detector,
                                 std::unique_ptr<ITrtModule> interaction,
                                 std::unique_ptr<ITrtModule> prompt_tracker,
                                 std::unique_ptr<ITrtModule> recurrent_tracker,
                                 std::unique_ptr<ITrtModule> memory_encoder, std::string model_id)
    : image_stream_owner_(std::move(image_stream_owner)), image_front_(std::move(image_front)),
      pafpn_(std::move(pafpn)), detector_(std::move(detector)),
      interaction_(std::move(interaction)), prompt_tracker_(std::move(prompt_tracker)),
      recurrent_tracker_(std::move(recurrent_tracker)), memory_encoder_(std::move(memory_encoder)),
      model_id_(std::move(model_id)) {
    require_dependencies({image_stream_owner_ != nullptr, image_front_ != nullptr,
                          pafpn_ != nullptr, detector_ != nullptr, interaction_ != nullptr,
                          prompt_tracker_ != nullptr, recurrent_tracker_ != nullptr,
                          memory_encoder_ != nullptr},
                         "SAM2 HOI Phase-A pipeline received a null dependency");
    validate_pipeline_modules(*image_front_, *detector_, *interaction_, *prompt_tracker_,
                              *recurrent_tracker_, *memory_encoder_);
    validate_phase_a_streams(*image_front_, *pafpn_, *detector_);
    pafpn_->bind_external_input("fpn_input_0", *image_front_, "fpn_input_0");
    pafpn_->bind_external_input("fpn_input_1", *image_front_, "tracker_feature_2");
    pafpn_->bind_external_input("fpn_input_2", *image_front_, "fpn_input_2");
    for (const char* name : {"detector_feature_0", "detector_feature_1", "detector_feature_2"})
        pafpn_->bind_output_to(name, *detector_, name);
    bind_tracker_inputs(*image_front_, *prompt_tracker_, *recurrent_tracker_, *memory_encoder_);
}

Sam2HoiPipeline::~Sam2HoiPipeline() {
    // Destruction can follow an exceptional exit after asynchronous front or
    // composite work. Never release producer buffers while the shared stream
    // may still reference them, and never throw from a destructor.
    try {
        if (pafpn_ != nullptr && detector_ != nullptr && detector_->stream() != nullptr)
            detector_->sync();
        else if (image_front_ != nullptr && image_front_->stream() != nullptr)
            image_front_->sync();
    } catch (...) {
    }
}

Sam2HoiVideoFrame Sam2HoiPipeline::load_video_frame(const std::string& path) {
    if (is_jpeg_path(path))
        return decode_jpeg_pillow_rgb(path);

    auto image = io::read_image(path);
    return {std::move(image.pixels), image.height, image.width};
}

std::vector<Sam2HoiVideoFrame>
Sam2HoiPipeline::load_video_frames(const std::vector<std::string>& paths) {
    if (!std::all_of(paths.begin(), paths.end(), is_jpeg_path)) {
        throw std::invalid_argument("SAM2 HOI batch frame loading requires only JPEG paths");
    }
    return decode_jpeg_pillow_rgb_batch(paths);
}

std::size_t Sam2HoiPipeline::max_video_frame_load_concurrency() const noexcept {
    return kMaxConcurrentJpegDecodes;
}

int32_t Sam2HoiPipeline::track_video(const std::vector<Sam2HoiVideoFrameView>& frames,
                                     const std::string& output_json,
                                     const std::string& output_masks_dir) {
    const bool discard_outputs = validate_video_request(frames, output_json, output_masks_dir);

    std::vector<FrameResult> results;
    std::vector<MemoryRecord> memory_records;
    std::vector<std::vector<float>> pointer_records;
    results.reserve(frames.size());
    memory_records.reserve(frames.size());
    pointer_records.reserve(frames.size());
    std::vector<int32_t> object_ids;
    bool tracking_started = false;
    RollingAsyncPreprocessor preprocessor(
        frames.size(), kMaxConcurrentPreprocessTasks, [&](std::size_t index) {
            const auto& frame = frames[index];
            return preprocess_image(frame.pixels, frame.height, frame.width);
        });
    OrderedAsyncMaskPostprocessor mask_postprocessor;

    for (std::size_t input_frame_index = 0; input_frame_index < frames.size();
         ++input_frame_index) {
        const auto& frame = frames[input_frame_index];
        auto pixels = preprocessor.take_next();
        process_video_frame(input_frame_index, frames.size(), frame, pixels, *image_front_,
                            pafpn_.get(), *detector_, *interaction_, *prompt_tracker_,
                            *recurrent_tracker_, *memory_encoder_, tracking_started, object_ids,
                            memory_records, pointer_records, mask_postprocessor, results);
    }

    while (!mask_postprocessor.empty())
        collect_mask_result(mask_postprocessor, results);

    if (results.empty())
        throw std::runtime_error("SAM2 HOI head found no detections in the video");
    if (!discard_outputs)
        write_results(results, output_json, output_masks_dir);
    return static_cast<int32_t>(results.size());
}

} // namespace trtmc::sam2_hoi
