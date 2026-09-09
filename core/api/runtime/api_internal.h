/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "trtmc/audio.h"
#include "trtmc/internal/audio.h"
#include "trtmc/internal/matrix.h"
#include "trtmc/internal/model.h"
#include "trtmc/internal/perception.h"
#include "trtmc/internal/scores.h"
#include "trtmc/internal/text.h"
#include "trtmc/internal/video.h"
#include "trtmc/matrix.h"
#include "trtmc/scores.h"
#include "trtmc/trtmc.h"

#include <cstddef>
#include <memory>
#include <mutex>
#include <new>
#include <string_view>
#include <utility>
#include <vector>

namespace trtmc::internal {
struct ImageView;
struct VideoView;
} // namespace trtmc::internal

namespace trtmc::api {

struct ApiFailure {
    trtmc_status status;
    const char* message;
};

struct OwnedApiFailure {
    trtmc_status status;
    std::string message;
};

void require(bool condition, const char* message);
std::size_t checked_size(std::uint64_t count, std::size_t element_size);

template <class T>
Span<const T> checked_span(const T* data, std::uint64_t count) {
    const auto size = checked_size(count, sizeof(T));
    require(data != nullptr || size == 0, "nonempty array has a null pointer");
    return {data, size};
}

std::string_view string_view(trtmc_string_view value);
trtmc_string_view borrowed_string(std::string_view value) noexcept;
std::string path_string(trtmc_string_view value);
std::string default_runtime_root();
trtmc_status store_error(trtmc_error**, trtmc_status, std::string_view) noexcept;
internal::ImageView image_input(const trtmc_image_input_v1&);
internal::VideoView video_input(const trtmc_video_view_v1&, std::vector<internal::ImageView>&);
internal::FloatMatrixView matrix_input(const trtmc_f32_matrix_view_v1&, bool optional = false);
trtmc_f32_matrix_view_v1 matrix_result_view(const internal::FloatMatrix&);

template <class Function>
trtmc_status guarded(trtmc_error** error, Function&& function) noexcept {
    if (error == nullptr)
        return TRTMC_INVALID_ARGUMENT;
    *error = nullptr;
    try {
        function();
        return TRTMC_OK;
    } catch (const ApiFailure& failure) {
        return store_error(error, failure.status, failure.message);
    } catch (const OwnedApiFailure& failure) {
        return store_error(error, failure.status, failure.message);
    } catch (const std::bad_alloc&) {
        return store_error(error, TRTMC_OUT_OF_MEMORY, "out of memory");
    } catch (const internal::ConfigError& failure) {
        return store_error(error, TRTMC_INVALID_CONFIG, failure.what());
    } catch (const internal::UnsupportedTask& failure) {
        return store_error(error, TRTMC_UNSUPPORTED, failure.what());
    } catch (const std::invalid_argument& failure) {
        return store_error(error, TRTMC_INVALID_ARGUMENT, failure.what());
    } catch (const std::exception& failure) {
        return store_error(error, TRTMC_INTERNAL_ERROR, failure.what());
    } catch (...) {
        return store_error(error, TRTMC_INTERNAL_ERROR, "unrecognized internal exception");
    }
}

// Borrowed input conversion. The object owns only the temporary view arrays.
class ConvertedConfig {
  public:
    explicit ConvertedConfig(const trtmc_config_view_v1* config);
    ConvertedConfig(const ConvertedConfig&) = delete;
    ConvertedConfig& operator=(const ConvertedConfig&) = delete;
    ConvertedConfig(ConvertedConfig&&) noexcept = default;
    ConvertedConfig& operator=(ConvertedConfig&&) noexcept = default;
    internal::ConfigView view() const noexcept { return {entries_.data(), entries_.size()}; }

  private:
    internal::ConfigValue convert(const trtmc_config_value_v1& value);
    std::vector<internal::ConfigEntry> entries_;
    std::vector<std::vector<std::string_view>> string_lists_;
};

// An owned snapshot of resolved family/session configuration, not a parser.
trtmc_result* make_config_snapshot(internal::ConfigView);
trtmc_status TRTMC_CALL config_snapshot_view(const trtmc_result*, trtmc_config_view_v1*,
                                             trtmc_error**) noexcept;

struct TaskBinding {
    std::string_view id;
    std::uint32_t major;
    std::uint32_t minor;
    const trtmc_api_header* api;
    bool (*implemented)(ITask&) noexcept;
};

template <class Interface>
bool implements(ITask& family) noexcept {
    return dynamic_cast<Interface*>(&family) != nullptr;
}

// These spans refer to immutable per-Task tables, not a registration service.
Span<const TaskBinding> text_continuation_bindings() noexcept;
Span<const TaskBinding> text_task_bindings() noexcept;
Span<const TaskBinding> image_task_bindings() noexcept;
Span<const TaskBinding> stream_task_bindings() noexcept;
Span<const TaskBinding> features_task_bindings() noexcept;
Span<const TaskBinding> audio_task_bindings() noexcept;
Span<const TaskBinding> numeric_task_bindings() noexcept;
Span<const TaskBinding> video_task_bindings() noexcept;
Span<const TaskBinding> perception_task_bindings() noexcept;
Span<const TaskBinding> language_task_bindings() noexcept;
Span<const TaskBinding> tracking_task_bindings() noexcept;
Span<const TaskBinding> speech_task_bindings() noexcept;
Span<const TaskBinding> action_task_bindings() noexcept;
Span<const TaskBinding> recurrent_task_bindings() noexcept;

struct ModelState;
std::shared_ptr<ModelState> model_owner(const trtmc_model* model);
std::mutex& model_mutex(const trtmc_model* model);
std::mutex& model_mutex(const std::shared_ptr<ModelState>& state);
ITask& model_family(const trtmc_model* model);
ITask& require_family(const trtmc_model* model, std::string_view task, std::uint32_t major = 1,
                      std::uint32_t minor = 0);
ITask& require_family(const std::shared_ptr<ModelState>& state, std::string_view task,
                      std::uint32_t major = 1, std::uint32_t minor = 0);
// Call with model_mutex held. Metadata queries do not require an idle model.
void require_model_idle(const trtmc_model* model);
void require_model_idle(const std::shared_ptr<ModelState>& state);

// A logical execution owner, not a thread-bound mutex lock or worker queue.
// Construct under model_mutex; destroy only after the family session stops,
// without holding model_mutex. It may safely be destroyed on another thread.
class ModelSession {
  public:
    explicit ModelSession(const trtmc_model* model);
    ~ModelSession();
    ModelSession(const ModelSession&) = delete;
    ModelSession& operator=(const ModelSession&) = delete;

  private:
    std::shared_ptr<ModelState> state_;
};

template <class Interface>
Interface& require_interface(const trtmc_model* model, std::string_view task) {
    require_model_idle(model);
    auto* interface = dynamic_cast<Interface*>(&require_family(model, task));
    if (interface == nullptr)
        throw ApiFailure{TRTMC_UNSUPPORTED, "family has not implemented this Task interface"};
    return *interface;
}

struct ResultStorage {
    ResultStorage() = default;
    ResultStorage(const ResultStorage&) = delete;
    ResultStorage& operator=(const ResultStorage&) = delete;
    ResultStorage(ResultStorage&&) = default;
    ResultStorage& operator=(ResultStorage&&) = default;
    virtual ~ResultStorage() = default;
};

struct TextResultStorage final : ResultStorage {
    explicit TextResultStorage(internal::TextResult result);
    internal::TextResult text;
    std::vector<trtmc_transcription_segment_v1> segments;
};

struct BatchTextResultStorage final : ResultStorage {
    explicit BatchTextResultStorage(internal::BatchTextResult results);
    std::vector<std::unique_ptr<TextResultStorage>> items;
};

struct LabelScoresStorage final : ResultStorage {
    explicit LabelScoresStorage(internal::LabelScoresResult result);
    internal::LabelScoresResult result;
    std::vector<trtmc_string_view> labels;
};

struct AudioResultStorage final : ResultStorage {
    explicit AudioResultStorage(internal::AudioResult result);
    internal::AudioResult result;
};

// Shared owned action representation; the only schema/timeline conversion is
// implemented with the existing video result code.
struct ActionSequenceStorage final : ResultStorage {
    explicit ActionSequenceStorage(internal::ActionSequenceResult);
    internal::ActionSequenceResult result;
    std::vector<trtmc_string_view> names, units;
    std::vector<trtmc_action_frame_span_v1> spans;
    trtmc_action_sequence_view_v1 view{};
};

internal::AudioView audio_view(const trtmc_audio_view_v1&);
void fill_audio_result_view(const AudioResultStorage&, trtmc_audio_result_view_v1*) noexcept;
trtmc_status TRTMC_CALL audio_result_view(const trtmc_result*, trtmc_audio_result_view_v1*,
                                          trtmc_error**) noexcept;

internal::TextSource text_source(const trtmc_text_source_v1&);
void fill_text_result_view(const TextResultStorage&, trtmc_text_result_view_v1*) noexcept;
trtmc_status TRTMC_CALL text_result_view(const trtmc_result*, trtmc_text_result_view_v1*,
                                         trtmc_error**) noexcept;
trtmc_status TRTMC_CALL text_batch_result_count(const trtmc_result*, std::uint64_t*,
                                                trtmc_error**) noexcept;
trtmc_status TRTMC_CALL text_batch_result_item_view(const trtmc_result*, std::uint64_t,
                                                    trtmc_text_result_view_v1*,
                                                    trtmc_error**) noexcept;
void fill_label_scores_view(const LabelScoresStorage&, trtmc_label_scores_view_v1*) noexcept;
trtmc_status TRTMC_CALL label_scores_result_view(const trtmc_result*, trtmc_label_scores_view_v1*,
                                                 trtmc_error**) noexcept;

internal::PixelBox pixel_box_input(trtmc_pixel_box_v1);
std::vector<internal::PointPrompt> point_prompts_input(const trtmc_point_prompt_v1*, std::uint64_t,
                                                       bool required);
internal::PoseMatricesView pose_matrices_input(const trtmc_pose_matrices_v1&,
                                               bool optional = false);
internal::TriangleMeshView triangle_mesh_input(const trtmc_triangle_mesh_v1&);
internal::PoseCropBatch pose_crops_input(void*, trtmc_pose_crop_callback_v1,
                                         const internal::PoseCropRequest&);
internal::PoseHypothesesCropsToRefinedPosesRequest
pose_refinement_input(const trtmc_pose_refinement_request_v1&);
std::array<float, 9> pixel_intrinsics_input(const float*);
trtmc_result* make_masks_result(internal::MasksResult);
trtmc_result* make_refined_poses_result(internal::RefinedPosesResult);
trtmc_result* make_object_pose_result(internal::ObjectPoseResult);
trtmc_status TRTMC_CALL masks_result_view(const trtmc_result*, trtmc_masks_view_v1*,
                                          trtmc_error**) noexcept;
trtmc_status TRTMC_CALL refined_poses_result_view(const trtmc_result*, trtmc_refined_poses_view_v1*,
                                                  trtmc_error**) noexcept;
trtmc_status TRTMC_CALL object_pose_result_view(const trtmc_result*, trtmc_object_pose_view_v1*,
                                                trtmc_error**) noexcept;

trtmc_status TRTMC_CALL bundle_open(trtmc_string_view, trtmc_bundle**, trtmc_error**) noexcept;
void TRTMC_CALL bundle_release(trtmc_bundle*) noexcept;
trtmc_status TRTMC_CALL bundle_info(const trtmc_bundle*, trtmc_bundle_info_v1*,
                                    trtmc_error**) noexcept;
trtmc_status TRTMC_CALL bundle_read_section(const trtmc_bundle*, trtmc_string_view, trtmc_result**,
                                            trtmc_error**) noexcept;
trtmc_status TRTMC_CALL bytes_result_view(const trtmc_result*, trtmc_bytes_view*,
                                          trtmc_error**) noexcept;
trtmc_status TRTMC_CALL byok_load(const trtmc_byok_options_v1*, trtmc_error**) noexcept;
trtmc_status TRTMC_CALL model_get_lora_api(const trtmc_model*, std::uint32_t, std::uint32_t,
                                           const trtmc_lora_api_v1**, trtmc_error**) noexcept;

} // namespace trtmc::api

struct trtmc_result {
    std::unique_ptr<trtmc::api::ResultStorage> storage;
};

namespace trtmc::api {

template <class Storage, class... Arguments>
trtmc_result* make_result(Arguments&&... arguments) {
    return new trtmc_result{std::make_unique<Storage>(std::forward<Arguments>(arguments)...)};
}

template <class Storage>
const Storage& require_result(const trtmc_result* result) {
    require(result != nullptr, "result is null");
    const auto* storage = dynamic_cast<const Storage*>(result->storage.get());
    require(storage != nullptr, "result does not contain the requested Task result type");
    return *storage;
}

} // namespace trtmc::api
