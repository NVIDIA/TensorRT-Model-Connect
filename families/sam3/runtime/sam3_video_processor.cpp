/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/sam3/runtime/sam3_video_processor.h"

#include "families/sam3/runtime/sam3_video_kernels.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <future>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace trtmc {

namespace {

constexpr int32_t kChannels = 3;
constexpr unsigned int kPillowResizePrecision = 22;
constexpr std::size_t kUint8Values = 256;
constexpr int32_t kTrackerMemoryChannels = 64;
constexpr int32_t kObjectPointerChannels = 256;
constexpr std::size_t kTrackerStepBatch2Size = 2;
constexpr std::size_t kHostProcessingWorkers = 4;
constexpr float kNoObjectLogit = -10.0F;
constexpr float kInitialMaskLogit = 1024.0F;
constexpr float kMemoryQualityThreshold = 0.01F;
constexpr int32_t kMemoryMaskInterpolationScale = 4;
constexpr float kMemoryAreaRetentionThreshold = 0.3F;

thread_local bool g_in_persistent_host_leaf_worker = false;

class HostLeafTaskRecord {
  public:
    template <typename Function>
    void configure(Function& function, std::size_t begin, std::size_t end) noexcept {
        context_ = std::addressof(function);
        invoke_ = [](void* context, std::size_t range_begin, std::size_t range_end) {
            (*static_cast<std::remove_reference_t<Function>*>(context))(range_begin, range_end);
        };
        begin_ = begin;
        end_ = end;
    }

    void run() noexcept {
        std::exception_ptr failure;
        try {
            invoke_(context_, begin_, end_);
        } catch (...) {
            failure = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            failure_ = std::move(failure);
            done_ = true;
        }
        complete_.notify_one();
    }

    std::exception_ptr wait_failure() {
        std::unique_lock<std::mutex> lock(mutex_);
        complete_.wait(lock, [this] { return done_; });
        return failure_;
    }

  private:
    using Invoke = void (*)(void*, std::size_t, std::size_t);

    void* context_{nullptr};
    Invoke invoke_{nullptr};
    std::size_t begin_{0};
    std::size_t end_{0};
    std::mutex mutex_;
    std::condition_variable complete_;
    std::exception_ptr failure_;
    bool done_{false};
};

class PersistentHostLeafExecutor {
  public:
    static constexpr std::size_t kWorkers = kHostProcessingWorkers - 1U;
    static constexpr std::size_t kQueueCapacity = 256;

    PersistentHostLeafExecutor() {
        try {
            for (auto& worker : workers_)
                worker = std::thread([this] { worker_loop(); });
        } catch (...) {
            shutdown();
            throw;
        }
    }

    ~PersistentHostLeafExecutor() { shutdown(); }

    PersistentHostLeafExecutor(const PersistentHostLeafExecutor&) = delete;
    PersistentHostLeafExecutor& operator=(const PersistentHostLeafExecutor&) = delete;

    bool try_submit(HostLeafTaskRecord& task) noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_ || queue_size_ == kQueueCapacity)
                return false;
            queue_[queue_tail_] = std::addressof(task);
            queue_tail_ = (queue_tail_ + 1U) % kQueueCapacity;
            ++queue_size_;
        }
        ready_.notify_one();
        return true;
    }

  private:
    void shutdown() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable())
                worker.join();
        }
    }

    void worker_loop() noexcept {
        g_in_persistent_host_leaf_worker = true;
        while (true) {
            HostLeafTaskRecord* task = nullptr;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this] { return stopping_ || queue_size_ != 0; });
                if (queue_size_ == 0) {
                    if (stopping_)
                        return;
                    continue;
                }
                task = queue_[queue_head_];
                queue_head_ = (queue_head_ + 1U) % kQueueCapacity;
                --queue_size_;
            }
            task->run();
        }
    }

    std::array<std::thread, kWorkers> workers_;
    std::array<HostLeafTaskRecord*, kQueueCapacity> queue_{};
    std::size_t queue_head_{0};
    std::size_t queue_tail_{0};
    std::size_t queue_size_{0};
    std::mutex mutex_;
    std::condition_variable ready_;
    bool stopping_{false};
};

PersistentHostLeafExecutor& persistent_host_leaf_executor() {
    static PersistentHostLeafExecutor executor;
    return executor;
}

template <typename Function>
void persistent_parallel_for_host_ranges(std::size_t count, std::size_t workers, std::size_t chunk,
                                         Function& function) {
    auto& executor = persistent_host_leaf_executor();
    std::array<HostLeafTaskRecord, kHostProcessingWorkers - 1U> tasks;
    std::array<bool, kHostProcessingWorkers - 1U> submitted{};
    for (std::size_t worker = 1; worker < workers; ++worker) {
        const auto begin = worker * chunk;
        const auto end = std::min(count, begin + chunk);
        auto& task = tasks[worker - 1U];
        task.configure(function, begin, end);
        submitted[worker - 1U] = executor.try_submit(task);
    }

    std::exception_ptr failure;
    try {
        function(0, std::min(count, chunk));
    } catch (...) {
        failure = std::current_exception();
    }
    for (std::size_t worker = 1; worker < workers; ++worker) {
        auto& task = tasks[worker - 1U];
        if (!submitted[worker - 1U])
            task.run();
        const auto worker_failure = task.wait_failure();
        if (failure == nullptr && worker_failure != nullptr)
            failure = worker_failure;
    }
    if (failure != nullptr)
        std::rethrow_exception(failure);
}

template <typename Function>
void run_nested_host_ranges(std::size_t count, std::size_t workers, std::size_t chunk,
                            Function& function) {
    // No current call site nests. Preserve the exact range and
    // exception-precedence contract anyway while executing nested work
    // serially so future composition cannot exhaust the pool.
    std::exception_ptr failure;
    for (std::size_t worker = 0; worker < workers; ++worker) {
        const auto begin = worker * chunk;
        const auto end = std::min(count, begin + chunk);
        try {
            function(begin, end);
        } catch (...) {
            if (failure == nullptr)
                failure = std::current_exception();
        }
    }
    if (failure != nullptr)
        std::rethrow_exception(failure);
}

template <typename Function>
void parallel_for_host_ranges(std::size_t count, Function&& function) {
    const auto workers = std::min(kHostProcessingWorkers, count);
    if (workers <= 1) {
        function(0, count);
        return;
    }

    const auto chunk = (count + workers - 1U) / workers;
    if (g_in_persistent_host_leaf_worker) {
        run_nested_host_ranges(count, workers, chunk, function);
        return;
    }
    persistent_parallel_for_host_ranges(count, workers, chunk, function);
}

template <typename Function>
void parallel_for_host_lanes(std::size_t count, Function&& function) {
    if (count == 0)
        return;
    const auto lanes = std::min(kHostProcessingWorkers, count);
    const auto lane_size = count / lanes;
    const auto extra = count % lanes;
    parallel_for_host_ranges(lanes, [&](std::size_t lane_begin, std::size_t lane_end) {
        for (std::size_t lane = lane_begin; lane < lane_end; ++lane) {
            const auto begin = lane * lane_size + std::min(lane, extra);
            const auto end = begin + lane_size + (lane < extra ? 1U : 0U);
            function(lane, begin, end);
        }
    });
}

template <typename T>
class FutureArrayJoinGuard {
  public:
    explicit FutureArrayJoinGuard(std::array<std::future<T>, 2>& futures) : futures_(futures) {}
    ~FutureArrayJoinGuard() {
        for (auto& future : futures_) {
            if (future.valid())
                future.wait();
        }
    }

  private:
    std::array<std::future<T>, 2>& futures_;
};

template <typename T>
class FutureJoinGuard {
  public:
    explicit FutureJoinGuard(std::future<T>& future) : future_(future) {}

    FutureJoinGuard(const FutureJoinGuard&) = delete;
    FutureJoinGuard& operator=(const FutureJoinGuard&) = delete;

    ~FutureJoinGuard() {
        if (!future_.valid())
            return;
        try {
            future_.wait();
        } catch (...) {
            // Preserve the exception already leaving the caller-owned detector
            // path. This also keeps every tracker capture alive until a
            // pipeline-owned worker has completely stopped using it.
        }
    }

  private:
    std::future<T>& future_;
};

std::size_t static_shape_values(const std::vector<int64_t>& shape) {
    std::size_t values = 1;
    if (shape.empty())
        return 0;
    for (const auto dimension : shape) {
        if (dimension <= 0 || values > std::numeric_limits<std::size_t>::max() /
                                           static_cast<std::size_t>(dimension)) {
            return 0;
        }
        values *= static_cast<std::size_t>(dimension);
    }
    return values;
}

struct Sam3FrameFeatures {
    TensorMap vision_outputs;
    // Candidate masks are stored compactly; detector_mask_queries maps each row back to the
    // original detector query index. Device-backed runtimes avoid downloading rejected rows.
    std::vector<float> detector_masks;
    std::vector<int32_t> detector_mask_queries;
    int32_t detector_queries{0};
    int32_t mask_height{0};
    int32_t mask_width{0};
    std::vector<float> detector_scores;
};

} // namespace

namespace {

struct PreprocessedFramePixels {
    std::vector<float> owned;
    const std::vector<float>& view() const { return owned; }
};

struct Detection {
    int32_t query_idx{0};
    float score{0.0F};
    std::vector<float> mask;
    std::vector<std::uint64_t> binary_words;
};

struct DeviceEncodedMemory {
    DeviceEncodedMemory(std::size_t values, cudaStream_t stream, int32_t cuda_device)
        : features({static_cast<int64_t>(values)}, DType::kFloat32, stream),
          position({static_cast<int64_t>(values)}, DType::kFloat32, stream), values(values),
          cuda_device(cuda_device) {}

    DeviceTensor features;
    DeviceTensor position;
    std::size_t values{0};
    int32_t cuda_device{0};
    // An exception may unwind after asynchronous writes were queued. Such an
    // entry stays owned until workspace destruction but is never handed to a
    // later session. Successful producers publish it only after their terminal
    // stream synchronization.
    bool reusable{false};
    bool quarantined{false};
};

struct TrackerStepPositionTemplate {
    float* destination{nullptr};
    std::size_t values_per_record{0};
    std::size_t populated_values{0};

    void invalidate() noexcept {
        destination = nullptr;
        values_per_record = 0;
        populated_values = 0;
    }
};

struct TrackerNeuralOutput {
    std::vector<float> mask;
    float object_score_logit{0.0F};
    float selected_iou{0.0F};
    float effective_iou_score{0.0F};
    std::vector<float> object_pointer;
    std::vector<float> memory_features;
    std::vector<float> memory_position;
    std::shared_ptr<DeviceEncodedMemory> device_memory;
};

struct EncodedMemory {
    std::vector<float> features;
    std::vector<float> position;
    std::shared_ptr<DeviceEncodedMemory> device;
};

struct TrackerFrameRecord {
    int32_t frame_idx{-1};
    bool conditioning{false};
    // Meta stores one frame output per tracker inference state.  Keep the
    // row-local term as well as the cohort mean so removing a row can
    // recompute the surviving state's shared quality without replaying the
    // tracker head.
    float individual_effective_iou_score{0.0F};
    float effective_iou_score{0.0F};
    std::vector<float> object_pointer;
    std::vector<float> memory_features;
    std::vector<float> memory_position;
    std::shared_ptr<DeviceEncodedMemory> device_memory;
};

bool record_has_memory(const TrackerFrameRecord& record) {
    return record.device_memory != nullptr || !record.memory_features.empty();
}

std::size_t record_memory_values(const TrackerFrameRecord& record) {
    return record.device_memory != nullptr ? record.device_memory->values
                                           : record.memory_features.size();
}

struct TrackState {
    int32_t object_id{-1};
    // Objects detected together are added to one Meta tracker inference
    // state. Later detections, even on a nearby frame, belong to a different
    // state and must not contribute to this cohort's frame-quality mean.
    int32_t cohort_id{-1};
    int32_t first_frame_idx{-1};
    float detection_score{0.0F};
    float tracker_score{0.0F};
    int32_t keep_alive{0};
    int32_t last_occluded{-1};
    int32_t conditioning_records_seen{0};
    // Meta's prompt-time add_new_mask path stores these consolidated low-resolution logits.
    // The first propagate call emits frame zero again and replaces its hard point-prompt memory
    // with a soft memory encoded from these logits before frame one may consume the state.
    std::vector<float> pending_frame_zero_soft_mask;
    std::optional<float> pending_frame_zero_object_score_logit;
    std::vector<int32_t> unmatched_frame_indices;
    std::vector<TrackerFrameRecord> records;
};

struct TrackerStepRequest {
    int32_t object_id{-1};
    std::vector<const TrackerFrameRecord*> memory_records;
    std::vector<int32_t> memory_temporal_offsets;
    std::vector<const TrackerFrameRecord*> pointer_records;
    std::vector<int32_t> pointer_temporal_offsets;
};

struct TrackerStepGroup {
    std::size_t memory_count{0};
    std::size_t pointer_count{0};
    std::vector<TrackerStepRequest> requests;
};

struct AssociationPlan {
    std::vector<int32_t> new_detection_indices;
    std::vector<int32_t> unmatched_track_ids;
    std::vector<int32_t> empty_track_ids;
    std::map<int32_t, std::vector<int32_t>> detection_to_track_ids;
    std::map<int32_t, int32_t> track_to_recondition_detection;
};

struct PropagatedTrackBatch {
    std::vector<int32_t> object_ids;
    std::map<int32_t, TrackerNeuralOutput> outputs;
    std::vector<std::vector<std::uint64_t>> packed_masks;
    std::vector<bool> has_foreground;
};

// Existing-track propagation is launched once per recurrent frame so its
// tracker stream can overlap the caller-owned detector/core stream. A fresh
// std::async thread is a measurable fixed cost for that overlap. Keep exactly
// one worker at pipeline lifetime instead: Sam3Pipeline serializes all video
// callbacks that share this workspace, and each callback joins its task before
// returning, so one pending slot is sufficient. This executor is deliberately
// independent from PersistentHostLeafExecutor because propagated-mask cleanup
// may submit nested leaf work and wait for all of it to finish.
struct MaskCleanupWorkspace {
    std::vector<std::uint32_t> visited;
    std::vector<std::uint32_t> component;
    std::uint32_t visit_epoch{0U};

    void begin_pass(std::size_t area) {
        if (visited.size() != area) {
            visited.assign(area, 0U);
            visit_epoch = 0U;
        }
        ++visit_epoch;
        if (visit_epoch == 0U) {
            std::fill(visited.begin(), visited.end(), 0U);
            visit_epoch = 1U;
        }
        component.clear();
        component.reserve(area);
    }
};

struct DeferredResultObject {
    int32_t object_id{-1};
    float detection_score{0.0F};
    float tracker_score{0.0F};
    std::vector<float> low_res_mask;
};

struct DeferredFrameResult {
    int32_t frame_idx{-1};
    int32_t height{0};
    int32_t width{0};
    int32_t low_res_height{0};
    int32_t low_res_width{0};
    std::vector<int32_t> removed_object_ids;
    std::vector<int32_t> suppressed_object_ids;
    std::vector<DeferredResultObject> objects;
};

struct PreparedNewTrackMasks {
    const Detection* detection{nullptr};
    std::vector<float> prompt_mask;
    std::vector<float> tracker_mask;
};

std::size_t checked_image_elements(int32_t height, int32_t width) {
    if (height <= 0 || width <= 0)
        throw std::invalid_argument("SAM3 video frame dimensions must be positive");
    const auto h = static_cast<std::size_t>(height);
    const auto w = static_cast<std::size_t>(width);
    if (h > std::numeric_limits<std::size_t>::max() / w ||
        h * w > std::numeric_limits<std::size_t>::max() / kChannels) {
        throw std::overflow_error("SAM3 video frame dimensions overflow");
    }
    return h * w * kChannels;
}

const Tensor& require_output(const TensorMap& outputs, const std::string& name,
                             const char* producer) {
    const auto iter = outputs.find(name);
    if (iter == outputs.end() || iter->second.data == nullptr) {
        throw std::runtime_error(std::string("SAM3 video ") + producer + " missing output " + name);
    }
    if (iter->second.dtype != DType::kFloat32) {
        throw std::runtime_error(std::string("SAM3 video ") + producer + " output " + name +
                                 " must be float32");
    }
    return iter->second;
}

std::vector<float> copy_float_tensor(const Tensor& tensor) {
    if (tensor.data == nullptr)
        return {};
    const auto* data = static_cast<const float*>(tensor.data);
    return std::vector<float>(data, data + tensor.numel());
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("SAM3 video ") + operation +
                                 " failed: " + cudaGetErrorString(status));
    }
}

class Sam3CudaPreprocessWorkspace;

} // namespace

class PinnedFloatBuffer {
  public:
    PinnedFloatBuffer() = default;
    PinnedFloatBuffer(const PinnedFloatBuffer&) = delete;
    PinnedFloatBuffer& operator=(const PinnedFloatBuffer&) = delete;

    ~PinnedFloatBuffer() {
        if (data_ != nullptr)
            (void)cudaFreeHost(data_);
    }

    void ensure(std::size_t values) {
        if (values <= capacity_)
            return;
        float* next = nullptr;
        check_cuda(cudaMallocHost(reinterpret_cast<void**>(&next), values * sizeof(float)),
                   "tracker output staging allocation");
        if (data_ != nullptr)
            (void)cudaFreeHost(data_);
        data_ = next;
        capacity_ = values;
    }

    float* data() const { return data_; }
    std::size_t capacity() const { return capacity_; }

  private:
    float* data_{nullptr};
    std::size_t capacity_{0};
};

struct TrackerHeadOutputStaging {
    PinnedFloatBuffer masks;
    PinnedFloatBuffer pointers;
    PinnedFloatBuffer scores;
    PinnedFloatBuffer selected_ious;
};

struct Sam3VideoVisionWorkspace {
    TrackerHeadOutputStaging tracker_init_output_staging;
    TrackerHeadOutputStaging parallel_tracker_init_output_staging;
    TrackerHeadOutputStaging tracker_step_output_staging;
    TrackerHeadOutputStaging tracker_step_batch2_output_staging;
    std::array<MaskCleanupWorkspace, kHostProcessingWorkers> propagated_mask_cleanup_workspaces;
    std::shared_ptr<Sam3CudaPreprocessWorkspace> cuda_preprocess;
    std::vector<std::shared_ptr<DeviceEncodedMemory>> recurrent_device_memory_pool;
};

namespace {

using EngineEnqueueCallback = std::function<void(bool, cudaStream_t)>;

std::size_t positive_shape_values(const std::vector<int64_t>& shape) {
    std::size_t values = 1;
    if (shape.empty())
        return 0;
    for (const auto dimension : shape) {
        if (dimension <= 0 || values > std::numeric_limits<std::size_t>::max() /
                                           static_cast<std::size_t>(dimension)) {
            return 0;
        }
        values *= static_cast<std::size_t>(dimension);
    }
    return values;
}

// The generic synchronous module path downloads every output with cudaMemcpy,
// which uses the CUDA null stream and therefore introduces cross-stream
// barriers. SAM3 tracker plans expose the four head outputs directly; copy only
// those outputs on the engine's own stream into reusable pinned host storage.
void require_tracker_device_output(const ITrtModule& engine, const char* name,
                                   std::size_t required_values) {
    if (!engine.has_output(name) || engine.device_ptr(name) == nullptr ||
        engine.tensor_dtype(name) != DType::kFloat32 ||
        positive_shape_values(engine.tensor_shape(name)) < required_values) {
        throw std::runtime_error(std::string("SAM3 tracker device contract is missing output ") +
                                 name);
    }
}

void forward_tracker_head_sparse(ITrtModule& engine, const TensorMap& inputs,
                                 std::size_t batch_size, int32_t mask_height, int32_t mask_width,
                                 TrackerHeadOutputStaging& staging, TensorMap& outputs,
                                 const EngineEnqueueCallback& after_enqueue = {}) {
    if (batch_size == 0 || mask_height <= 0 || mask_width <= 0)
        throw std::runtime_error("SAM3 tracker head received invalid output geometry");
    const auto mask_area = static_cast<std::size_t>(mask_height) * mask_width;
    if (mask_area > std::numeric_limits<std::size_t>::max() / batch_size)
        throw std::overflow_error("SAM3 tracker head output geometry overflow");
    const auto mask_values = batch_size * mask_area;
    const auto pointer_values = batch_size * kObjectPointerChannels;
    require_tracker_device_output(engine, "pred_masks", mask_values);
    require_tracker_device_output(engine, "object_pointer", pointer_values);
    require_tracker_device_output(engine, "object_score_logits", batch_size);
    require_tracker_device_output(engine, "selected_iou", batch_size);

    staging.masks.ensure(mask_values);
    staging.pointers.ensure(pointer_values);
    staging.scores.ensure(batch_size);
    staging.selected_ious.ensure(batch_size);
    const auto stream = engine.stream();
    try {
        engine.forward_async(inputs);
        if (after_enqueue)
            after_enqueue(true, stream);
        check_cuda(cudaMemcpyAsync(staging.masks.data(), engine.device_ptr("pred_masks"),
                                   mask_values * sizeof(float), cudaMemcpyDeviceToHost, stream),
                   "tracker mask download");
        check_cuda(cudaMemcpyAsync(staging.pointers.data(), engine.device_ptr("object_pointer"),
                                   pointer_values * sizeof(float), cudaMemcpyDeviceToHost, stream),
                   "tracker pointer download");
        check_cuda(cudaMemcpyAsync(staging.scores.data(), engine.device_ptr("object_score_logits"),
                                   batch_size * sizeof(float), cudaMemcpyDeviceToHost, stream),
                   "tracker score download");
        check_cuda(cudaMemcpyAsync(staging.selected_ious.data(), engine.device_ptr("selected_iou"),
                                   batch_size * sizeof(float), cudaMemcpyDeviceToHost, stream),
                   "tracker selected IoU download");
        engine.sync();
    } catch (...) {
        engine.sync();
        throw;
    }

    outputs.clear();
    outputs["pred_masks"] = Tensor{staging.masks.data(),
                                   {static_cast<int64_t>(batch_size), 1, mask_height, mask_width},
                                   DType::kFloat32};
    outputs["object_pointer"] =
        Tensor{staging.pointers.data(),
               {static_cast<int64_t>(batch_size), 1, kObjectPointerChannels},
               DType::kFloat32};
    outputs["object_score_logits"] =
        Tensor{staging.scores.data(), {static_cast<int64_t>(batch_size), 1, 1}, DType::kFloat32};
    outputs["selected_iou"] = Tensor{
        staging.selected_ious.data(), {static_cast<int64_t>(batch_size), 1, 1}, DType::kFloat32};
}

class ModuleSyncGuard {
  public:
    explicit ModuleSyncGuard(ITrtModule& module) : module_(module) {}
    ModuleSyncGuard(const ModuleSyncGuard&) = delete;
    ModuleSyncGuard& operator=(const ModuleSyncGuard&) = delete;
    ~ModuleSyncGuard() noexcept {
        try {
            module_.sync();
        } catch (...) {
            // Destructors must not terminate the process if a failed enqueue
            // also makes the stream synchronization report an error. The
            // propagation path surfaces the primary failure explicitly.
        }
    }

  private:
    ITrtModule& module_;
};

float sigmoid(float value) {
    if (value >= 0.0F) {
        const float z = std::exp(-value);
        return 1.0F / (1.0F + z);
    }
    const float z = std::exp(value);
    return z / (1.0F + z);
}

std::array<float, 3> channel_values(const std::vector<float>& configured,
                                    std::array<float, 3> fallback) {
    for (std::size_t index = 0; index < std::min(configured.size(), fallback.size()); ++index)
        fallback[index] = configured[index];
    return fallback;
}

struct ResizeAxisEntry {
    int32_t first{0};
    std::vector<int32_t> weights;
};

struct ResizeAxisPlan {
    std::vector<ResizeAxisEntry> entries;
    unsigned int precision{0};
};

struct ResizeFloatAxisEntry {
    int32_t first{0};
    std::vector<double> weights;
};

ResizeFloatAxisEntry make_float_axis_entry(int32_t input_size, int32_t output_index, double scale,
                                           double support, int32_t max_interp_size,
                                           double& max_weight) {
    const double center = scale * (static_cast<double>(output_index) + 0.5);
    const double invscale = scale >= 1.0 ? 1.0 / scale : 1.0;
    const int64_t first = std::max<int64_t>(static_cast<int64_t>(center - support + 0.5), 0);
    int64_t count =
        std::min<int64_t>(static_cast<int64_t>(center + support + 0.5), input_size) - first;
    count = std::clamp<int64_t>(count, 0, max_interp_size);

    ResizeFloatAxisEntry entry;
    entry.first = static_cast<int32_t>(first);
    entry.weights.resize(static_cast<std::size_t>(count));
    double total_weight = 0.0;
    for (int64_t index = 0; index < count; ++index) {
        const double distance = (static_cast<double>(index + first) - center + 0.5) * invscale;
        const double weight = std::max(0.0, 1.0 - std::abs(distance));
        entry.weights[static_cast<std::size_t>(index)] = weight;
        total_weight += weight;
    }
    if (total_weight != 0.0) {
        for (auto& weight : entry.weights) {
            weight /= total_weight;
            max_weight = std::max(max_weight, weight);
        }
    }
    return entry;
}

ResizeAxisEntry quantize_axis_entry(const ResizeFloatAxisEntry& source, unsigned int precision) {
    ResizeAxisEntry entry;
    entry.first = source.first;
    entry.weights.reserve(source.weights.size());
    const double multiplier = static_cast<double>(1U << precision);
    for (const double weight : source.weights) {
        const double scaled = weight * multiplier;
        const int rounded = static_cast<int>(scaled < 0.0 ? scaled - 0.5 : scaled + 0.5);
        entry.weights.push_back(static_cast<int32_t>(rounded));
    }
    return entry;
}

ResizeAxisPlan make_uint8_antialias_axis(int32_t input_size, int32_t output_size) {
    // Meta's image-folder loader resizes a PIL RGB image through
    // torchvision.transforms.functional.resize. Pillow's bilinear uint8
    // implementation computes coefficients in double, quantizes every axis
    // to a fixed 22-bit signed int32 representation, and rounds each
    // separable pass back to uint8.
    const double scale = static_cast<double>(input_size) / static_cast<double>(output_size);
    const double support = scale >= 1.0 ? scale : 1.0;
    const int32_t max_interp_size = static_cast<int32_t>(std::ceil(support)) * 2 + 1;

    std::vector<ResizeFloatAxisEntry> float_entries;
    float_entries.reserve(static_cast<std::size_t>(output_size));
    double max_weight = 0.0;
    for (int32_t output_index = 0; output_index < output_size; ++output_index)
        float_entries.push_back(make_float_axis_entry(input_size, output_index, scale, support,
                                                      max_interp_size, max_weight));

    ResizeAxisPlan plan;
    (void)max_weight;
    plan.precision = kPillowResizePrecision;
    plan.entries.reserve(float_entries.size());
    for (const auto& float_entry : float_entries)
        plan.entries.push_back(quantize_axis_entry(float_entry, plan.precision));
    return plan;
}

uint8_t apply_uint8_resize_weights(const uint8_t* values, std::size_t stride,
                                   const ResizeAxisEntry& entry, unsigned int precision) {
    int32_t accumulated = 1 << (precision - 1U);
    for (std::size_t index = 0; index < entry.weights.size(); ++index)
        accumulated += static_cast<int32_t>(values[index * stride]) * entry.weights[index];
    return static_cast<uint8_t>(std::clamp(accumulated >> precision, 0, 255));
}

void resize_uint8_hwc_antialiased_into(const std::vector<uint8_t>& input, int32_t input_height,
                                       int32_t input_width, int32_t output_height,
                                       int32_t output_width, const ResizeAxisPlan& horizontal_plan,
                                       const ResizeAxisPlan& vertical_plan,
                                       std::vector<uint8_t>& horizontal,
                                       std::vector<uint8_t>& output) {
    const std::vector<uint8_t>* vertical_input = &input;
    int32_t vertical_input_width = input_width;
    if (input_width != output_width) {
        horizontal.resize(static_cast<std::size_t>(input_height) * output_width * kChannels);
        parallel_for_host_ranges(static_cast<std::size_t>(input_height), [&](std::size_t begin,
                                                                             std::size_t end) {
            for (std::size_t y = begin; y < end; ++y) {
                for (int32_t x = 0; x < output_width; ++x) {
                    const auto& entry = horizontal_plan.entries[static_cast<std::size_t>(x)];
                    for (int32_t channel = 0; channel < kChannels; ++channel) {
                        const auto input_offset =
                            (y * static_cast<std::size_t>(input_width) + entry.first) * kChannels +
                            channel;
                        const auto output_offset =
                            (y * static_cast<std::size_t>(output_width) + x) * kChannels + channel;
                        horizontal[output_offset] =
                            apply_uint8_resize_weights(input.data() + input_offset, kChannels,
                                                       entry, horizontal_plan.precision);
                    }
                }
            }
        });
        vertical_input = &horizontal;
        vertical_input_width = output_width;
    }

    if (input_height == output_height) {
        output.assign(vertical_input->begin(), vertical_input->end());
        return;
    }

    output.resize(static_cast<std::size_t>(output_height) * vertical_input_width * kChannels);
    const std::size_t row_stride = static_cast<std::size_t>(vertical_input_width) * kChannels;
    parallel_for_host_ranges(static_cast<std::size_t>(output_height), [&](std::size_t begin,
                                                                          std::size_t end) {
        for (std::size_t y = begin; y < end; ++y) {
            const auto& entry = vertical_plan.entries[y];
            for (int32_t x = 0; x < vertical_input_width; ++x) {
                for (int32_t channel = 0; channel < kChannels; ++channel) {
                    const auto input_offset = static_cast<std::size_t>(entry.first) * row_stride +
                                              static_cast<std::size_t>(x) * kChannels + channel;
                    const auto output_offset =
                        (y * static_cast<std::size_t>(vertical_input_width) + x) * kChannels +
                        channel;
                    output[output_offset] =
                        apply_uint8_resize_weights(vertical_input->data() + input_offset,
                                                   row_stride, entry, vertical_plan.precision);
                }
            }
        }
    });
}

struct Sam3ImageNormalization {
    std::array<float, static_cast<std::size_t>(kChannels) * kUint8Values> lookup{};
};

struct Sam3ImagePreprocessWorkspace {
    int32_t input_height{0};
    int32_t input_width{0};
    int32_t output_size{0};
    ResizeAxisPlan horizontal_plan;
    ResizeAxisPlan vertical_plan;
    std::vector<uint8_t> input;
    std::vector<uint8_t> horizontal;
    std::vector<uint8_t> resized;
    std::vector<float> normalized;
    Sam3ImageNormalization normalization;
    std::array<float, kChannels> normalization_mean{};
    std::array<float, kChannels> normalization_stddev{};
    bool normalization_is_valid{false};
};

void refresh_image_preprocess_plan(int32_t height, int32_t width, int32_t output_size,
                                   Sam3ImagePreprocessWorkspace& workspace) {
    if (workspace.input_height == height && workspace.input_width == width &&
        workspace.output_size == output_size) {
        return;
    }
    workspace.input_height = height;
    workspace.input_width = width;
    workspace.output_size = output_size;
    workspace.horizontal_plan =
        width != output_size ? make_uint8_antialias_axis(width, output_size) : ResizeAxisPlan{};
    workspace.vertical_plan =
        height != output_size ? make_uint8_antialias_axis(height, output_size) : ResizeAxisPlan{};
}

void quantize_sam3_image(const float* hwc_pixels, std::size_t input_elements,
                         std::vector<uint8_t>& quantized) {
    quantized.resize(input_elements);
    parallel_for_host_ranges(quantized.size(), [&](std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
            const float value = hwc_pixels[index];
            if (!std::isfinite(value))
                throw std::invalid_argument("SAM3 image pixels must be finite");
            quantized[index] = static_cast<uint8_t>(std::clamp(
                static_cast<int>(std::floor(std::fma(std::clamp(value, 0.0F, 1.0F), 255.0F, 0.5F))),
                0, 255));
        }
    });
}

std::uint16_t float_special_to_fp16_bits(std::uint16_t sign, std::uint32_t mantissa) {
    if (mantissa == 0U)
        return static_cast<std::uint16_t>(sign | 0x7C00U);
    const auto payload = static_cast<std::uint16_t>(std::max(1U, mantissa >> 13U));
    return static_cast<std::uint16_t>(sign | 0x7C00U | payload | 0x0200U);
}

std::uint16_t float_subnormal_to_fp16_bits(std::uint16_t sign, std::uint32_t mantissa,
                                           int32_t half_exponent) {
    if (half_exponent < -10)
        return sign;
    const std::uint32_t significand = mantissa | 0x00800000U;
    const auto shift = static_cast<unsigned int>(14 - half_exponent);
    std::uint32_t rounded = significand >> shift;
    const std::uint32_t remainder = significand & ((1U << shift) - 1U);
    const std::uint32_t midpoint = 1U << (shift - 1U);
    if (remainder > midpoint || (remainder == midpoint && (rounded & 1U) != 0U))
        ++rounded;
    return static_cast<std::uint16_t>(sign | rounded);
}

std::uint16_t float_normal_to_fp16_bits(std::uint16_t sign, std::uint32_t mantissa,
                                        int32_t half_exponent) {
    std::uint32_t rounded_mantissa = mantissa >> 13U;
    const std::uint32_t remainder = mantissa & 0x1FFFU;
    if (remainder > 0x1000U || (remainder == 0x1000U && (rounded_mantissa & 1U) != 0U))
        ++rounded_mantissa;
    auto rounded_exponent = static_cast<std::uint32_t>(half_exponent);
    if (rounded_mantissa == 0x400U) {
        rounded_mantissa = 0U;
        ++rounded_exponent;
        if (rounded_exponent >= 31U)
            return static_cast<std::uint16_t>(sign | 0x7C00U);
    }
    return static_cast<std::uint16_t>(sign | (rounded_exponent << 10U) | rounded_mantissa);
}

std::uint16_t float_to_fp16_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint16_t sign = static_cast<std::uint16_t>((bits >> 16U) & 0x8000U);
    const std::uint32_t exponent = (bits >> 23U) & 0xFFU;
    const std::uint32_t mantissa = bits & 0x007FFFFFU;

    if (exponent == 0xFFU)
        return float_special_to_fp16_bits(sign, mantissa);
    const int32_t half_exponent = static_cast<int32_t>(exponent) - 127 + 15;
    if (half_exponent >= 31)
        return static_cast<std::uint16_t>(sign | 0x7C00U);
    if (half_exponent <= 0)
        return float_subnormal_to_fp16_bits(sign, mantissa, half_exponent);
    return float_normal_to_fp16_bits(sign, mantissa, half_exponent);
}

float fp16_bits_to_float(std::uint16_t value) {
    const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000U) << 16U;
    int32_t exponent = static_cast<int32_t>((value >> 10U) & 0x1FU);
    std::uint32_t mantissa = value & 0x03FFU;
    std::uint32_t bits = sign;
    if (exponent == 0) {
        if (mantissa != 0U) {
            exponent = 1;
            while ((mantissa & 0x0400U) == 0U) {
                mantissa <<= 1U;
                --exponent;
            }
            mantissa &= 0x03FFU;
            bits |= static_cast<std::uint32_t>(exponent + (127 - 15)) << 23U;
            bits |= mantissa << 13U;
        }
    } else if (exponent == 0x1F) {
        bits |= 0x7F800000U | (mantissa << 13U);
    } else {
        bits |= static_cast<std::uint32_t>(exponent + (127 - 15)) << 23U;
        bits |= mantissa << 13U;
    }
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float round_to_fp16(float value) {
    return fp16_bits_to_float(float_to_fp16_bits(value));
}

Sam3ImageNormalization image_normalization(const Sam3Config& config) {
    const auto mean = channel_values(config.image_mean, {0.5F, 0.5F, 0.5F});
    const auto stddev = channel_values(config.image_std, {0.5F, 0.5F, 0.5F});
    Sam3ImageNormalization result;
    for (int32_t channel = 0; channel < kChannels; ++channel) {
        const auto channel_index = static_cast<std::size_t>(channel);
        const float fp16_mean = round_to_fp16(mean[channel_index]);
        const float fp16_stddev = round_to_fp16(stddev[channel_index]);
        if (!std::isfinite(fp16_mean) || !std::isfinite(fp16_stddev) || fp16_stddev == 0.0F)
            throw std::invalid_argument("SAM3 image normalization parameters are invalid in FP16");
        for (std::size_t value = 0; value < kUint8Values; ++value) {
            // Match Meta's image-folder loader operation order: ToTensor
            // divides uint8 by 255 in FP32, then the tensor, mean, and std are
            // converted to FP16 before in-place subtraction and division.
            const float pixel = round_to_fp16(static_cast<float>(value) / 255.0F);
            const float centered = round_to_fp16(pixel - fp16_mean);
            result.lookup[channel_index * kUint8Values + value] =
                round_to_fp16(centered / fp16_stddev);
        }
    }
    return result;
}

const Sam3ImageNormalization& cached_image_normalization(const Sam3Config& config,
                                                         Sam3ImagePreprocessWorkspace& workspace) {
    const auto mean = channel_values(config.image_mean, {0.5F, 0.5F, 0.5F});
    const auto stddev = channel_values(config.image_std, {0.5F, 0.5F, 0.5F});
    if (!workspace.normalization_is_valid || workspace.normalization_mean != mean ||
        workspace.normalization_stddev != stddev) {
        workspace.normalization = image_normalization(config);
        workspace.normalization_mean = mean;
        workspace.normalization_stddev = stddev;
        workspace.normalization_is_valid = true;
    }
    return workspace.normalization;
}

void normalize_sam3_image(const std::vector<uint8_t>& resized, int32_t output_size,
                          const Sam3ImageNormalization& normalization, std::vector<float>& output) {
    const std::size_t plane = static_cast<std::size_t>(output_size) * output_size;
    output.resize(static_cast<std::size_t>(kChannels) * plane);
    parallel_for_host_ranges(static_cast<std::size_t>(output_size), [&](std::size_t begin,
                                                                        std::size_t end) {
        for (std::size_t y = begin; y < end; ++y) {
            for (int32_t x = 0; x < output_size; ++x) {
                for (int32_t channel = 0; channel < kChannels; ++channel) {
                    const auto input_offset =
                        (y * static_cast<std::size_t>(output_size) + x) * kChannels + channel;
                    const auto channel_index = static_cast<std::size_t>(channel);
                    const auto output_offset = channel_index * plane + y * output_size + x;
                    output[output_offset] =
                        normalization.lookup[channel_index * kUint8Values + resized[input_offset]];
                }
            }
        }
    });
}

std::vector<float>& preprocess_sam3_image_into(const float* hwc_pixels, int32_t height,
                                               int32_t width, const Sam3Config& config,
                                               Sam3ImagePreprocessWorkspace& workspace) {
    const auto input_elements = checked_image_elements(height, width);
    if (hwc_pixels == nullptr)
        throw std::invalid_argument("SAM3 image pixels must not be null");
    if (config.image_size <= 0)
        throw std::invalid_argument("SAM3 image size must be positive");

    const int32_t output_size = config.image_size;
    refresh_image_preprocess_plan(height, width, output_size, workspace);
    quantize_sam3_image(hwc_pixels, input_elements, workspace.input);
    resize_uint8_hwc_antialiased_into(workspace.input, height, width, output_size, output_size,
                                      workspace.horizontal_plan, workspace.vertical_plan,
                                      workspace.horizontal, workspace.resized);
    normalize_sam3_image(workspace.resized, output_size,
                         cached_image_normalization(config, workspace), workspace.normalized);
    return workspace.normalized;
}

class ReusableCudaAllocation {
  public:
    ReusableCudaAllocation() = default;
    ReusableCudaAllocation(const ReusableCudaAllocation&) = delete;
    ReusableCudaAllocation& operator=(const ReusableCudaAllocation&) = delete;

    ~ReusableCudaAllocation() {
        if (data_ != nullptr)
            (void)cudaFree(data_);
    }

    bool ensure(std::size_t bytes) noexcept {
        if (bytes <= capacity_)
            return true;
        void* next = nullptr;
        if (cudaMalloc(&next, bytes) != cudaSuccess) {
            (void)cudaGetLastError();
            return false;
        }
        if (data_ != nullptr)
            (void)cudaFree(data_);
        data_ = next;
        capacity_ = bytes;
        return true;
    }

    template <typename T>
    T* as() const noexcept {
        return static_cast<T*>(data_);
    }

  private:
    void* data_{nullptr};
    std::size_t capacity_{0};
};

class Sam3CudaPreprocessWorkspace {
    struct PreflightCapacity {
        std::size_t input_values{0};
        std::size_t horizontal_values{0};
        std::size_t horizontal_entries{0};
        std::size_t horizontal_weights{0};
        std::size_t vertical_entries{0};
        std::size_t vertical_weights{0};
    };

  public:
    struct FlatAxisPlan {
        std::vector<Sam3CudaResizeAxisEntry> entries;
        std::vector<std::int32_t> weights;
        unsigned int precision{0};
    };

    struct CachedPlan {
        int32_t input_height{0};
        int32_t input_width{0};
        int32_t output_size{0};
        FlatAxisPlan horizontal;
        FlatAxisPlan vertical;
    };

    struct DevicePlanView {
        const Sam3CudaResizeAxisEntry* horizontal_entries{nullptr};
        int32_t horizontal_entry_count{0};
        const std::int32_t* horizontal_weights{nullptr};
        int32_t horizontal_weight_count{0};
        unsigned int horizontal_precision{0};
        const Sam3CudaResizeAxisEntry* vertical_entries{nullptr};
        int32_t vertical_entry_count{0};
        const std::int32_t* vertical_weights{nullptr};
        int32_t vertical_weight_count{0};
        unsigned int vertical_precision{0};
    };

    Sam3CudaPreprocessWorkspace() = default;
    Sam3CudaPreprocessWorkspace(const Sam3CudaPreprocessWorkspace&) = delete;
    Sam3CudaPreprocessWorkspace& operator=(const Sam3CudaPreprocessWorkspace&) = delete;

    ~Sam3CudaPreprocessWorkspace() {
        drain_noexcept();
        if (host_nonfinite_status_ != nullptr)
            (void)cudaFreeHost(host_nonfinite_status_);
        if (stream_ != nullptr)
            (void)cudaStreamDestroy(stream_);
    }

    bool preflight(const Sam3VideoFrame* frames, std::size_t frame_count, int32_t output_size,
                   const Sam3Config& config) noexcept {
        try {
            if (frames == nullptr || frame_count != 1)
                return false;
            if (!output_extent_is_representable(output_size))
                return false;
            PreflightCapacity capacity;
            if (!measure_preflight_frame(frames[0], output_size, capacity))
                return false;
            if (!preflight_byte_capacities_fit(capacity))
                return false;
            if (!initialize_normalization(config))
                return false;
            return prepare_preflight_resources(capacity);
        } catch (...) {
            return false;
        }
    }

    const CachedPlan* find_plan(int32_t input_height, int32_t input_width,
                                int32_t output_size) const noexcept {
        for (const auto& plan : plans_) {
            if (plan.input_height == input_height && plan.input_width == input_width &&
                plan.output_size == output_size) {
                return &plan;
            }
        }
        return nullptr;
    }

    DevicePlanView upload_plan(const CachedPlan& plan) {
        DevicePlanView view;
        upload_axis(plan.horizontal, horizontal_entries_, horizontal_weights_, stream_,
                    view.horizontal_entries, view.horizontal_entry_count, view.horizontal_weights,
                    view.horizontal_weight_count, view.horizontal_precision,
                    "CUDA preprocessing horizontal plan upload");
        upload_axis(plan.vertical, vertical_entries_, vertical_weights_, stream_,
                    view.vertical_entries, view.vertical_entry_count, view.vertical_weights,
                    view.vertical_weight_count, view.vertical_precision,
                    "CUDA preprocessing vertical plan upload");
        return view;
    }

    void drain_noexcept() noexcept {
        if (stream_ != nullptr)
            (void)cudaStreamSynchronize(stream_);
    }

    cudaStream_t stream() const noexcept { return stream_; }
    float* raw_input() const noexcept { return raw_input_.as<float>(); }
    std::uint8_t* quantized() const noexcept { return quantized_.as<std::uint8_t>(); }
    std::uint8_t* horizontal() const noexcept { return horizontal_.as<std::uint8_t>(); }
    int* device_nonfinite_status() const noexcept { return nonfinite_status_.as<int>(); }
    int* host_nonfinite_status() const noexcept { return host_nonfinite_status_; }
    const float* device_normalization_lut() const noexcept {
        return device_normalization_lut_.as<float>();
    }

  private:
    static bool output_extent_is_representable(int32_t output_size) {
        if (output_size <= 0)
            return false;
        const auto output = static_cast<std::size_t>(output_size);
        return output <= std::numeric_limits<std::size_t>::max() / output &&
               output * output <= std::numeric_limits<std::size_t>::max() / kChannels;
    }

    static bool valid_preflight_frame(const Sam3VideoFrame& frame, std::size_t& input_values) {
        if (frame.height <= 0 || frame.width <= 0 || frame.pixel_data() == nullptr)
            return false;
        input_values = checked_image_elements(frame.height, frame.width);
        return frame.pixel_count() == input_values;
    }

    static bool horizontal_value_capacity(const Sam3VideoFrame& frame, std::size_t output,
                                          std::size_t& values) {
        values = 0;
        if (static_cast<std::size_t>(frame.width) == output)
            return true;
        const auto height = static_cast<std::size_t>(frame.height);
        if (height > std::numeric_limits<std::size_t>::max() / output)
            return false;
        const auto pixels = height * output;
        if (pixels > std::numeric_limits<std::size_t>::max() / kChannels)
            return false;
        values = pixels * kChannels;
        return true;
    }

    bool measure_preflight_frame(const Sam3VideoFrame& frame, int32_t output_size,
                                 PreflightCapacity& capacity) {
        if (!valid_preflight_frame(frame, capacity.input_values))
            return false;
        const CachedPlan* plan = find_or_build_plan(frame.height, frame.width, output_size);
        if (plan == nullptr)
            return false;
        capacity.horizontal_entries = plan->horizontal.entries.size();
        capacity.horizontal_weights = plan->horizontal.weights.size();
        capacity.vertical_entries = plan->vertical.entries.size();
        capacity.vertical_weights = plan->vertical.weights.size();
        return horizontal_value_capacity(frame, static_cast<std::size_t>(output_size),
                                         capacity.horizontal_values);
    }

    static bool preflight_byte_capacities_fit(const PreflightCapacity& capacity) {
        const auto limit = std::numeric_limits<std::size_t>::max();
        return capacity.input_values <= limit / sizeof(float) &&
               capacity.horizontal_entries <= limit / sizeof(Sam3CudaResizeAxisEntry) &&
               capacity.horizontal_weights <= limit / sizeof(std::int32_t) &&
               capacity.vertical_entries <= limit / sizeof(Sam3CudaResizeAxisEntry) &&
               capacity.vertical_weights <= limit / sizeof(std::int32_t);
    }

    bool initialize_normalization(const Sam3Config& config) {
        auto next = image_normalization(config).lookup;
        if (!normalization_lut_is_valid_ || next != normalization_lut_) {
            normalization_lut_ = std::move(next);
            normalization_lut_is_valid_ = true;
            normalization_lut_is_uploaded_ = false;
        }
        return true;
    }

    bool ensure_preflight_stream() {
        if (stream_ != nullptr)
            return true;
        if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) == cudaSuccess)
            return true;
        stream_ = nullptr;
        (void)cudaGetLastError();
        return false;
    }

    bool ensure_preflight_allocations(const PreflightCapacity& capacity) {
        return raw_input_.ensure(capacity.input_values * sizeof(float)) &&
               quantized_.ensure(capacity.input_values) &&
               horizontal_.ensure(capacity.horizontal_values) &&
               horizontal_entries_.ensure(capacity.horizontal_entries *
                                          sizeof(Sam3CudaResizeAxisEntry)) &&
               horizontal_weights_.ensure(capacity.horizontal_weights * sizeof(std::int32_t)) &&
               vertical_entries_.ensure(capacity.vertical_entries *
                                        sizeof(Sam3CudaResizeAxisEntry)) &&
               vertical_weights_.ensure(capacity.vertical_weights * sizeof(std::int32_t)) &&
               device_normalization_lut_.ensure(normalization_lut_.size() * sizeof(float)) &&
               nonfinite_status_.ensure(sizeof(int));
    }

    bool ensure_host_nonfinite_status() {
        if (host_nonfinite_status_ != nullptr)
            return true;
        if (cudaMallocHost(reinterpret_cast<void**>(&host_nonfinite_status_), sizeof(int)) ==
            cudaSuccess) {
            return true;
        }
        host_nonfinite_status_ = nullptr;
        (void)cudaGetLastError();
        return false;
    }

    bool prepare_preflight_resources(const PreflightCapacity& capacity) {
        if (!ensure_preflight_stream() || !ensure_preflight_allocations(capacity) ||
            !ensure_host_nonfinite_status()) {
            return false;
        }
        if (normalization_lut_is_uploaded_)
            return true;
        if (cudaMemcpyAsync(device_normalization_lut_.as<float>(), normalization_lut_.data(),
                            normalization_lut_.size() * sizeof(float), cudaMemcpyHostToDevice,
                            stream_) == cudaSuccess) {
            normalization_lut_is_uploaded_ = true;
            return true;
        }
        (void)cudaGetLastError();
        return false;
    }

    static bool valid_flat_axis_entry(const ResizeAxisEntry& entry, int32_t input_size,
                                      std::size_t accumulated_weights) {
        const auto max_int = static_cast<std::size_t>(std::numeric_limits<int32_t>::max());
        return entry.first >= 0 && !entry.weights.empty() &&
               entry.weights.size() <= static_cast<std::size_t>(input_size) &&
               static_cast<std::size_t>(entry.first) <=
                   static_cast<std::size_t>(input_size) - entry.weights.size() &&
               accumulated_weights <= max_int &&
               entry.weights.size() <= max_int - accumulated_weights;
    }

    static bool flatten_axis(const ResizeAxisPlan& source, int32_t input_size, int32_t output_size,
                             FlatAxisPlan& target) {
        target = {};
        if (input_size == output_size)
            return source.entries.empty();
        if (source.entries.size() != static_cast<std::size_t>(output_size) ||
            source.precision != kPillowResizePrecision) {
            return false;
        }
        target.precision = source.precision;
        target.entries.reserve(source.entries.size());
        for (const auto& entry : source.entries) {
            if (!valid_flat_axis_entry(entry, input_size, target.weights.size()))
                return false;
            Sam3CudaResizeAxisEntry flat;
            flat.first = entry.first;
            flat.weight_offset = static_cast<int32_t>(target.weights.size());
            flat.weight_count = static_cast<int32_t>(entry.weights.size());
            target.entries.push_back(flat);
            target.weights.insert(target.weights.end(), entry.weights.begin(), entry.weights.end());
        }
        return !target.weights.empty();
    }

    static bool axis_plan_is_representable(int32_t input_size, int32_t output_size) {
        if (input_size <= 0 || output_size <= 0)
            return false;
        if (input_size == output_size)
            return true;
        const double support = std::max(static_cast<double>(input_size) / output_size, 1.0);
        return std::ceil(support) <=
               static_cast<double>((std::numeric_limits<int32_t>::max() - 1) / 2);
    }

    static ResizeAxisPlan make_optional_axis_plan(int32_t input_size, int32_t output_size) {
        return input_size == output_size ? ResizeAxisPlan{}
                                         : make_uint8_antialias_axis(input_size, output_size);
    }

    static bool build_cached_plan(int32_t input_height, int32_t input_width, int32_t output_size,
                                  CachedPlan& plan) {
        if (!axis_plan_is_representable(input_height, output_size) ||
            !axis_plan_is_representable(input_width, output_size)) {
            return false;
        }
        plan.input_height = input_height;
        plan.input_width = input_width;
        plan.output_size = output_size;
        const auto horizontal = make_optional_axis_plan(input_width, output_size);
        const auto vertical = make_optional_axis_plan(input_height, output_size);
        return flatten_axis(horizontal, input_width, output_size, plan.horizontal) &&
               flatten_axis(vertical, input_height, output_size, plan.vertical);
    }

    const CachedPlan* find_or_build_plan(int32_t input_height, int32_t input_width,
                                         int32_t output_size) {
        if (const auto* existing = find_plan(input_height, input_width, output_size))
            return existing;
        CachedPlan plan;
        if (!build_cached_plan(input_height, input_width, output_size, plan))
            return nullptr;
        plans_.push_back(std::move(plan));
        return &plans_.back();
    }

    static void upload_axis(const FlatAxisPlan& plan, ReusableCudaAllocation& device_entries,
                            ReusableCudaAllocation& device_weights, cudaStream_t stream,
                            const Sam3CudaResizeAxisEntry*& entries, int32_t& entry_count,
                            const std::int32_t*& weights, int32_t& weight_count,
                            unsigned int& precision, const char* operation) {
        entries = nullptr;
        weights = nullptr;
        entry_count = 0;
        weight_count = 0;
        precision = 0;
        if (plan.entries.empty())
            return;
        check_cuda(cudaMemcpyAsync(device_entries.as<Sam3CudaResizeAxisEntry>(),
                                   plan.entries.data(),
                                   plan.entries.size() * sizeof(Sam3CudaResizeAxisEntry),
                                   cudaMemcpyHostToDevice, stream),
                   operation);
        check_cuda(cudaMemcpyAsync(device_weights.as<std::int32_t>(), plan.weights.data(),
                                   plan.weights.size() * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, stream),
                   operation);
        entries = device_entries.as<Sam3CudaResizeAxisEntry>();
        weights = device_weights.as<std::int32_t>();
        entry_count = static_cast<int32_t>(plan.entries.size());
        weight_count = static_cast<int32_t>(plan.weights.size());
        precision = plan.precision;
    }

    std::vector<CachedPlan> plans_;
    cudaStream_t stream_{nullptr};
    ReusableCudaAllocation raw_input_;
    ReusableCudaAllocation quantized_;
    ReusableCudaAllocation horizontal_;
    ReusableCudaAllocation horizontal_entries_;
    ReusableCudaAllocation horizontal_weights_;
    ReusableCudaAllocation vertical_entries_;
    ReusableCudaAllocation vertical_weights_;
    ReusableCudaAllocation device_normalization_lut_;
    ReusableCudaAllocation nonfinite_status_;
    int* host_nonfinite_status_{nullptr};
    std::array<float, static_cast<std::size_t>(kChannels) * kUint8Values> normalization_lut_{};
    bool normalization_lut_is_valid_{false};
    bool normalization_lut_is_uploaded_{false};
};

std::vector<int64_t> batched_text_shape(const std::vector<int64_t>& shape) {
    if (shape.size() == 2)
        return {1, shape[0], shape[1]};
    return shape;
}

int32_t query_count(const std::vector<int64_t>& shape) {
    if (shape.size() == 2)
        return static_cast<int32_t>(shape[1]);
    if (shape.size() == 1)
        return static_cast<int32_t>(shape[0]);
    return 0;
}

bool mask_geometry(const std::vector<int64_t>& shape, int32_t& queries, int32_t& height,
                   int32_t& width) {
    if (shape.size() == 4) {
        queries = static_cast<int32_t>(shape[1]);
        height = static_cast<int32_t>(shape[2]);
        width = static_cast<int32_t>(shape[3]);
    } else if (shape.size() == 3) {
        queries = static_cast<int32_t>(shape[0]);
        height = static_cast<int32_t>(shape[1]);
        width = static_cast<int32_t>(shape[2]);
    } else {
        return false;
    }
    return queries > 0 && height > 0 && width > 0;
}

std::vector<std::uint64_t> pack_binary_mask(const float* mask, std::size_t size) {
    std::vector<std::uint64_t> words((size + 63U) / 64U, 0U);
    for (std::size_t index = 0; index < size; ++index) {
        if (mask[index] > 0.0F)
            words[index / 64U] |= std::uint64_t{1} << (index % 64U);
    }
    return words;
}

std::size_t popcount64(std::uint64_t value) {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<std::size_t>(__builtin_popcountll(value));
#else
    std::size_t count = 0;
    while (value != 0U) {
        value &= value - 1U;
        ++count;
    }
    return count;
#endif
}

float packed_mask_iou(const std::vector<std::uint64_t>& lhs,
                      const std::vector<std::uint64_t>& rhs) {
    if (lhs.size() != rhs.size())
        throw std::runtime_error("SAM3 video packed mask IoU received different mask sizes");
    std::size_t intersection = 0;
    std::size_t union_count = 0;
    for (std::size_t index = 0; index < lhs.size(); ++index) {
        intersection += popcount64(lhs[index] & rhs[index]);
        union_count += popcount64(lhs[index] | rhs[index]);
    }
    return union_count == 0 ? 0.0F
                            : static_cast<float>(intersection) / static_cast<float>(union_count);
}

bool packed_mask_has_foreground(const std::vector<std::uint64_t>& mask) {
    return std::any_of(mask.begin(), mask.end(), [](std::uint64_t word) { return word != 0U; });
}

std::vector<int32_t> sorted_detection_candidates(const Sam3FrameFeatures& frame, float threshold) {
    std::vector<int32_t> candidates;
    for (int32_t query = 0; query < frame.detector_queries; ++query) {
        if (frame.detector_scores[static_cast<std::size_t>(query)] > threshold)
            candidates.push_back(query);
    }
    std::stable_sort(candidates.begin(), candidates.end(), [&](int32_t lhs, int32_t rhs) {
        return frame.detector_scores[static_cast<std::size_t>(lhs)] >
               frame.detector_scores[static_cast<std::size_t>(rhs)];
    });
    return candidates;
}

std::vector<std::vector<std::uint64_t>> pack_candidate_masks(const Sam3FrameFeatures& frame,
                                                             const std::vector<int32_t>& candidates,
                                                             std::size_t mask_area) {
    std::vector<std::vector<std::uint64_t>> packed_masks(
        static_cast<std::size_t>(frame.detector_queries));
    for (const int32_t candidate : candidates) {
        const auto query = std::lower_bound(frame.detector_mask_queries.begin(),
                                            frame.detector_mask_queries.end(), candidate);
        if (query == frame.detector_mask_queries.end() || *query != candidate)
            throw std::runtime_error("SAM3 video detector candidate mask is missing");
        const auto offset =
            static_cast<std::size_t>(std::distance(frame.detector_mask_queries.begin(), query)) *
            mask_area;
        packed_masks[static_cast<std::size_t>(candidate)] =
            pack_binary_mask(frame.detector_masks.data() + offset, mask_area);
    }
    return packed_masks;
}

bool detection_survives_nms(int32_t candidate, const std::vector<int32_t>& selected,
                            const std::vector<std::vector<std::uint64_t>>& packed_masks,
                            float threshold) {
    if (threshold <= 0.0F)
        return true;
    for (const int32_t prior : selected) {
        if (packed_mask_iou(packed_masks[static_cast<std::size_t>(candidate)],
                            packed_masks[static_cast<std::size_t>(prior)]) > threshold) {
            return false;
        }
    }
    return true;
}

std::vector<bool>
detection_nms_keep_flags(int32_t query_count, const std::vector<int32_t>& candidates,
                         const std::vector<std::vector<std::uint64_t>>& packed_masks,
                         float threshold) {
    std::vector<bool> keep(static_cast<std::size_t>(query_count), false);
    std::vector<int32_t> selected;
    for (const int32_t candidate : candidates) {
        if (!detection_survives_nms(candidate, selected, packed_masks, threshold))
            continue;
        keep[static_cast<std::size_t>(candidate)] = true;
        selected.push_back(candidate);
    }
    return keep;
}

Detection make_detection(const Sam3FrameFeatures& frame, int32_t query, std::size_t mask_area,
                         std::vector<std::uint64_t> binary_words) {
    const auto compact_query = std::lower_bound(frame.detector_mask_queries.begin(),
                                                frame.detector_mask_queries.end(), query);
    if (compact_query == frame.detector_mask_queries.end() || *compact_query != query)
        throw std::runtime_error("SAM3 video selected detector mask is missing");
    const auto offset = static_cast<std::size_t>(
                            std::distance(frame.detector_mask_queries.begin(), compact_query)) *
                        mask_area;
    Detection detection;
    detection.query_idx = query;
    detection.score = frame.detector_scores[static_cast<std::size_t>(query)];
    detection.mask.assign(frame.detector_masks.begin() + static_cast<std::ptrdiff_t>(offset),
                          frame.detector_masks.begin() +
                              static_cast<std::ptrdiff_t>(offset + mask_area));
    detection.binary_words = std::move(binary_words);
    return detection;
}

std::vector<Detection> select_detections(const Sam3FrameFeatures& frame, const Sam3Config& config) {
    const auto mask_area = static_cast<std::size_t>(frame.mask_height) * frame.mask_width;
    const auto candidates = sorted_detection_candidates(frame, config.detection_threshold);
    auto packed_masks = pack_candidate_masks(frame, candidates, mask_area);
    const auto keep = detection_nms_keep_flags(frame.detector_queries, candidates, packed_masks,
                                               config.detection_nms_threshold);

    std::vector<Detection> detections;
    for (int32_t query = 0; query < frame.detector_queries; ++query) {
        if (!keep[static_cast<std::size_t>(query)])
            continue;
        detections.push_back(make_detection(
            frame, query, mask_area, std::move(packed_masks[static_cast<std::size_t>(query)])));
    }
    return detections;
}

void validate_neural_output(TrackerNeuralOutput& output, int32_t mask_height, int32_t mask_width,
                            const char* producer) {
    const auto mask_area = static_cast<std::size_t>(mask_height) * mask_width;
    if (output.mask.size() != mask_area)
        throw std::runtime_error(std::string("SAM3 video ") + producer +
                                 " returned an unexpected mask shape");
    if (output.object_pointer.size() != kObjectPointerChannels)
        throw std::runtime_error(std::string("SAM3 video ") + producer +
                                 " returned an unexpected object pointer shape");
}

std::vector<TrackerNeuralOutput> parse_tracker_head_outputs(const TensorMap& outputs,
                                                            std::size_t batch_size,
                                                            int32_t mask_height, int32_t mask_width,
                                                            const char* producer) {
    if (batch_size == 0)
        throw std::invalid_argument("SAM3 video tracker output batch must not be empty");
    const auto masks = copy_float_tensor(require_output(outputs, "pred_masks", producer));
    const auto pointers = copy_float_tensor(require_output(outputs, "object_pointer", producer));
    const auto scores = copy_float_tensor(require_output(outputs, "object_score_logits", producer));
    const auto selected_ious = copy_float_tensor(require_output(outputs, "selected_iou", producer));
    const auto mask_area = static_cast<std::size_t>(mask_height) * mask_width;
    if (masks.size() != batch_size * mask_area ||
        pointers.size() != batch_size * kObjectPointerChannels || scores.size() != batch_size ||
        selected_ious.size() != batch_size) {
        throw std::runtime_error(std::string("SAM3 video ") + producer +
                                 " returned an unexpected object-major batch shape");
    }

    std::vector<TrackerNeuralOutput> results(batch_size);
    for (std::size_t batch = 0; batch < batch_size; ++batch) {
        auto& result = results[batch];
        result.mask.assign(masks.begin() + static_cast<std::ptrdiff_t>(batch * mask_area),
                           masks.begin() + static_cast<std::ptrdiff_t>((batch + 1) * mask_area));
        result.object_pointer.assign(
            pointers.begin() + static_cast<std::ptrdiff_t>(batch * kObjectPointerChannels),
            pointers.begin() + static_cast<std::ptrdiff_t>((batch + 1) * kObjectPointerChannels));
        result.object_score_logit = scores[batch];
        result.selected_iou = selected_ious[batch];
        const float object_score_norm = result.object_score_logit > 0.0F
                                            ? sigmoid(result.object_score_logit) * 2.0F - 1.0F
                                            : 0.0F;
        result.effective_iou_score = object_score_norm * result.selected_iou;
        validate_neural_output(result, mask_height, mask_width, producer);
    }
    return results;
}

TrackerNeuralOutput parse_tracker_init_output(const TensorMap& outputs) {
    TrackerNeuralOutput result;
    result.object_pointer =
        copy_float_tensor(require_output(outputs, "object_pointer", "tracker init engine"));
    const auto scores =
        copy_float_tensor(require_output(outputs, "object_score_logits", "tracker init engine"));
    if (result.object_pointer.size() != kObjectPointerChannels || scores.size() != 1) {
        throw std::runtime_error(
            "SAM3 video tracker init engine returned an unexpected pointer shape");
    }
    result.object_score_logit = scores.front();
    return result;
}

bool component_pixel_selected(const std::vector<float>& mask, std::size_t index, bool foreground) {
    return foreground ? mask[index] > 0.0F : mask[index] <= 0.0F;
}

struct MaskCleanupStats {
    std::size_t unselected_pixels{0};
    std::size_t replaced_pixels{0};
};

void append_component_neighbors(const std::vector<float>& mask, int32_t height, int32_t width,
                                bool foreground, std::size_t current,
                                MaskCleanupWorkspace& workspace) {
    const int32_t current_y = static_cast<int32_t>(current / width);
    const int32_t current_x = static_cast<int32_t>(current % width);
    const int32_t min_y = std::max(current_y - 1, 0);
    const int32_t max_y = std::min(current_y + 1, height - 1);
    const int32_t min_x = std::max(current_x - 1, 0);
    const int32_t max_x = std::min(current_x + 1, width - 1);
    for (int32_t neighbor_y = min_y; neighbor_y <= max_y; ++neighbor_y) {
        for (int32_t neighbor_x = min_x; neighbor_x <= max_x; ++neighbor_x) {
            const auto neighbor = static_cast<std::size_t>(neighbor_y) * width + neighbor_x;
            if (workspace.visited[neighbor] == workspace.visit_epoch ||
                !component_pixel_selected(mask, neighbor, foreground)) {
                continue;
            }
            workspace.visited[neighbor] = workspace.visit_epoch;
            workspace.component.push_back(static_cast<std::uint32_t>(neighbor));
        }
    }
}

void collect_mask_component(const std::vector<float>& mask, int32_t height, int32_t width,
                            bool foreground, std::size_t first, MaskCleanupWorkspace& workspace) {
    workspace.component.clear();
    workspace.component.push_back(static_cast<std::uint32_t>(first));
    workspace.visited[first] = workspace.visit_epoch;
    for (std::size_t head = 0; head < workspace.component.size(); ++head) {
        append_component_neighbors(mask, height, width, foreground,
                                   static_cast<std::size_t>(workspace.component[head]), workspace);
    }
}

std::size_t replace_small_mask_component(std::vector<float>& mask, int32_t threshold,
                                         float replacement, const MaskCleanupWorkspace& workspace) {
    if (workspace.component.size() > static_cast<std::size_t>(threshold))
        return 0;
    for (const auto index : workspace.component)
        mask[index] = replacement;
    return workspace.component.size();
}

MaskCleanupStats clean_mask_components(std::vector<float>& mask, int32_t height, int32_t width,
                                       bool foreground, int32_t threshold, float replacement,
                                       MaskCleanupWorkspace& workspace) {
    MaskCleanupStats stats;
    workspace.begin_pass(mask.size());
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            const auto pixel = static_cast<std::size_t>(y) * width + x;
            if (!component_pixel_selected(mask, pixel, foreground)) {
                ++stats.unselected_pixels;
                continue;
            }
            if (workspace.visited[pixel] == workspace.visit_epoch)
                continue;
            collect_mask_component(mask, height, width, foreground, pixel, workspace);
            stats.replaced_pixels +=
                replace_small_mask_component(mask, threshold, replacement, workspace);
        }
    }
    return stats;
}

void fill_small_components(std::vector<float>& mask, int32_t height, int32_t width,
                           int32_t max_area, MaskCleanupWorkspace& workspace) {
    if (max_area <= 0 || height <= 0 || width <= 0)
        return;
    const auto area = static_cast<std::size_t>(height) * width;
    if (mask.size() != area)
        throw std::runtime_error("SAM3 video component cleanup mask shape mismatch");
    if (height > 0xFFFF || width > 0xFFFF || area > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("SAM3 video component cleanup mask dimensions are too large");
    }

    const auto background =
        clean_mask_components(mask, height, width, false, max_area, 0.1F, workspace);
    const auto foreground_area =
        static_cast<int32_t>(background.unselected_pixels + background.replaced_pixels);
    clean_mask_components(mask, height, width, true, std::min(max_area, foreground_area / 2), -0.1F,
                          workspace);
}

struct ResizedMaskSummary {
    bool has_foreground{false};
    std::array<float, 4> box{0.0F, 0.0F, 0.0F, 0.0F};
};

struct MaskResizeAxisEntry {
    int32_t lower{0};
    int32_t upper{0};
    float weight{0.0F};
};

struct MaskResizePlan {
    int32_t source_height{0};
    int32_t source_width{0};
    int32_t output_height{0};
    int32_t output_width{0};
    std::vector<MaskResizeAxisEntry> y;
    std::vector<MaskResizeAxisEntry> x;
};

float resized_mask_logit(const std::vector<float>& source, const MaskResizePlan& plan,
                         const MaskResizeAxisEntry& y_entry, const MaskResizeAxisEntry& x_entry);

struct ResizedMaskBounds {
    int32_t min_x{0};
    int32_t min_y{0};
    int32_t max_x{-1};
    int32_t max_y{-1};

    explicit ResizedMaskBounds(const MaskResizePlan& plan)
        : min_x(plan.output_width), min_y(plan.output_height) {}

    bool has_foreground() const noexcept { return max_x >= 0; }

    void include(int32_t x, int32_t y) noexcept {
        min_x = std::min(min_x, x);
        min_y = std::min(min_y, y);
        max_x = std::max(max_x, x);
        max_y = std::max(max_y, y);
    }

    void merge(const ResizedMaskBounds& other) noexcept {
        min_x = std::min(min_x, other.min_x);
        min_y = std::min(min_y, other.min_y);
        max_x = std::max(max_x, other.max_x);
        max_y = std::max(max_y, other.max_y);
    }
};

std::vector<MaskResizeAxisEntry> make_mask_resize_axis_plan(int32_t source_size,
                                                            int32_t output_size) {
    std::vector<MaskResizeAxisEntry> plan;
    plan.reserve(output_size);
    for (int32_t output_index = 0; output_index < output_size; ++output_index) {
        const float source =
            (static_cast<float>(output_index) + 0.5F) * source_size / output_size - 0.5F;
        const float clamped = std::clamp(source, 0.0F, static_cast<float>(source_size - 1));
        const int32_t lower = static_cast<int32_t>(std::floor(clamped));
        const int32_t upper = std::min(lower + 1, source_size - 1);
        plan.push_back({lower, upper, clamped - lower});
    }
    return plan;
}

MaskResizePlan make_mask_resize_plan(int32_t source_height, int32_t source_width,
                                     int32_t output_height, int32_t output_width) {
    return {source_height,
            source_width,
            output_height,
            output_width,
            make_mask_resize_axis_plan(source_height, output_height),
            make_mask_resize_axis_plan(source_width, output_width)};
}

std::vector<float> resize_float_mask_bilinear(const std::vector<float>& source,
                                              int32_t source_height, int32_t source_width,
                                              int32_t output_height, int32_t output_width) {
    const auto source_area = static_cast<std::size_t>(source_height) * source_width;
    if (source_height <= 0 || source_width <= 0 || output_height <= 0 || output_width <= 0 ||
        source.size() != source_area) {
        throw std::invalid_argument("SAM3 mask bilinear resize received invalid geometry");
    }
    if (source_height == output_height && source_width == output_width)
        return source;
    const auto plan =
        make_mask_resize_plan(source_height, source_width, output_height, output_width);
    std::vector<float> output(static_cast<std::size_t>(output_height) * output_width);
    parallel_for_host_ranges(static_cast<std::size_t>(output_height), [&](std::size_t begin,
                                                                          std::size_t end) {
        for (std::size_t y = begin; y < end; ++y) {
            const auto& y_entry = plan.y[y];
            for (int32_t x = 0; x < output_width; ++x) {
                output[y * output_width + x] =
                    resized_mask_logit(source, plan, y_entry, plan.x[static_cast<std::size_t>(x)]);
            }
        }
    });
    return output;
}

struct GlobalHardMaskOwnership {
    int32_t height{0};
    int32_t width{0};
    std::size_t object_count{0};
    std::vector<float> winning_logits;
    std::vector<std::uint32_t> winning_indices;
};

void update_global_hard_mask_owners(std::size_t object_index, const std::vector<float>& resized,
                                    GlobalHardMaskOwnership& ownership) {
    if (object_index >= ownership.object_count ||
        resized.size() != ownership.winning_indices.size()) {
        throw std::invalid_argument("SAM3 resized hard mask has invalid geometry");
    }
    if (object_index == 0) {
        ownership.winning_logits = resized;
        return;
    }
    if (ownership.winning_logits.size() != resized.size())
        throw std::invalid_argument("SAM3 hard-memory ownership is incomplete");
    parallel_for_host_ranges(resized.size(), [&](std::size_t begin, std::size_t end) {
        for (std::size_t pixel = begin; pixel < end; ++pixel) {
            // Meta uses torch.argmax over the object axis. A strict comparison
            // preserves the first row for equal logits.
            if (resized[pixel] > ownership.winning_logits[pixel]) {
                ownership.winning_logits[pixel] = resized[pixel];
                ownership.winning_indices[pixel] = static_cast<std::uint32_t>(object_index);
            }
        }
    });
}

GlobalHardMaskOwnership make_global_hard_mask_ownership(std::size_t object_count,
                                                        int32_t tracker_image_size) {
    GlobalHardMaskOwnership ownership;
    ownership.height = tracker_image_size;
    ownership.width = tracker_image_size;
    ownership.object_count = object_count;
    if (object_count == 0)
        return ownership;
    if (tracker_image_size <= 0 ||
        object_count > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
        throw std::invalid_argument("SAM3 hard-memory ownership received invalid geometry");
    }
    const auto area = static_cast<std::size_t>(tracker_image_size) * tracker_image_size;
    ownership.winning_indices.assign(area, 0U);
    return ownership;
}

std::vector<float> materialize_owned_hard_mask(const GlobalHardMaskOwnership& ownership,
                                               std::size_t object_index) {
    if (object_index >= ownership.object_count || ownership.winning_logits.empty() ||
        ownership.winning_logits.size() != ownership.winning_indices.size()) {
        throw std::invalid_argument("SAM3 hard-memory ownership is incomplete");
    }
    std::vector<float> mask(ownership.winning_logits.size(), 0.0F);
    parallel_for_host_ranges(mask.size(), [&](std::size_t begin, std::size_t end) {
        for (std::size_t pixel = begin; pixel < end; ++pixel) {
            mask[pixel] = ownership.winning_indices[pixel] == object_index &&
                                  ownership.winning_logits[pixel] > 0.0F
                              ? 1.0F
                              : 0.0F;
        }
    });
    return mask;
}

struct FloatMaskResizeAxisEntry {
    int32_t first{0};
    std::vector<float> weights;
};

std::vector<FloatMaskResizeAxisEntry> make_float_antialias_axis_plan(int32_t input_size,
                                                                     int32_t output_size) {
    if (input_size <= 0 || output_size <= 0)
        throw std::invalid_argument("SAM3 antialias resize received invalid axis geometry");
    // Match ATen's CUDA antialiased bilinear implementation for float input:
    // all coordinate, coefficient, normalization, and accumulation arithmetic
    // is float, with a float materialization between the horizontal and
    // vertical passes.
    const float scale = static_cast<float>(input_size) / static_cast<float>(output_size);
    const float support = scale >= 1.0F ? scale : 1.0F;
    const float invscale = scale >= 1.0F ? 1.0F / scale : 1.0F;
    std::vector<FloatMaskResizeAxisEntry> plan;
    plan.reserve(static_cast<std::size_t>(output_size));
    for (int32_t output_index = 0; output_index < output_size; ++output_index) {
        const float center = scale * (static_cast<float>(output_index) + 0.5F);
        const int32_t first = std::max(static_cast<int32_t>(center - support + 0.5F), 0);
        const int32_t count =
            std::min(static_cast<int32_t>(center + support + 0.5F), input_size) - first;
        FloatMaskResizeAxisEntry entry;
        entry.first = first;
        entry.weights.resize(static_cast<std::size_t>(std::max(count, 0)));
        float total_weight = 0.0F;
        for (int32_t tap = 0; tap < count; ++tap) {
            const float distance = (static_cast<float>(tap + first) - center + 0.5F) * invscale;
            const float weight = std::max(0.0F, 1.0F - std::abs(distance));
            entry.weights[static_cast<std::size_t>(tap)] = weight;
            total_weight += weight;
        }
        if (total_weight != 0.0F) {
            for (auto& weight : entry.weights)
                weight /= total_weight;
        }
        plan.push_back(std::move(entry));
    }
    return plan;
}

std::vector<float>
resize_float_mask_horizontal(const std::vector<float>& source, int32_t source_height,
                             int32_t source_width, int32_t output_width,
                             const std::vector<FloatMaskResizeAxisEntry>& x_plan) {
    std::vector<float> horizontal(static_cast<std::size_t>(source_height) * output_width);
    parallel_for_host_ranges(
        static_cast<std::size_t>(source_height), [&](std::size_t begin, std::size_t end) {
            for (std::size_t y = begin; y < end; ++y) {
                for (int32_t x = 0; x < output_width; ++x) {
                    const auto& entry = x_plan[static_cast<std::size_t>(x)];
                    float value = 0.0F;
                    for (std::size_t tap = 0; tap < entry.weights.size(); ++tap) {
                        value += source[y * source_width + entry.first + tap] * entry.weights[tap];
                    }
                    horizontal[y * output_width + x] = value;
                }
            }
        });
    return horizontal;
}

std::vector<float> resize_float_mask_vertical(const std::vector<float>& horizontal,
                                              int32_t output_height, int32_t output_width,
                                              const std::vector<FloatMaskResizeAxisEntry>& y_plan) {
    std::vector<float> output(static_cast<std::size_t>(output_height) * output_width);
    parallel_for_host_ranges(
        static_cast<std::size_t>(output_height), [&](std::size_t begin, std::size_t end) {
            for (std::size_t y = begin; y < end; ++y) {
                const auto& entry = y_plan[y];
                for (int32_t x = 0; x < output_width; ++x) {
                    float value = 0.0F;
                    for (std::size_t tap = 0; tap < entry.weights.size(); ++tap) {
                        value +=
                            horizontal[(entry.first + tap) * output_width + x] * entry.weights[tap];
                    }
                    output[y * output_width + x] = value;
                }
            }
        });
    return output;
}

std::vector<float> resize_float_mask_antialiased(const std::vector<float>& source,
                                                 int32_t source_height, int32_t source_width,
                                                 int32_t output_height, int32_t output_width) {
    const auto source_area = static_cast<std::size_t>(source_height) * source_width;
    if (source_height <= 0 || source_width <= 0 || output_height <= 0 || output_width <= 0 ||
        source.size() != source_area) {
        throw std::invalid_argument("SAM3 mask antialias resize received invalid geometry");
    }
    if (source_height == output_height && source_width == output_width)
        return source;

    const auto x_plan = make_float_antialias_axis_plan(source_width, output_width);
    const auto y_plan = make_float_antialias_axis_plan(source_height, output_height);
    const auto horizontal =
        resize_float_mask_horizontal(source, source_height, source_width, output_width, x_plan);
    return resize_float_mask_vertical(horizontal, output_height, output_width, y_plan);
}

void validate_initial_tracker_mask_geometry(int32_t low_res_height, int32_t low_res_width,
                                            int32_t video_height, int32_t video_width) {
    if (low_res_height <= 0 || low_res_width <= 0 || video_height <= 0 || video_width <= 0)
        throw std::invalid_argument("SAM3 tracker initialization received invalid mask geometry");
    if (low_res_height > std::numeric_limits<int32_t>::max() / 4 ||
        low_res_width > std::numeric_limits<int32_t>::max() / 4) {
        throw std::overflow_error("SAM3 tracker mask-input geometry overflow");
    }
}

std::vector<std::vector<std::uint8_t>>
make_initial_tracker_foregrounds(const std::vector<const Detection*>& detections,
                                 int32_t low_res_height, int32_t low_res_width,
                                 int32_t video_height, int32_t video_width) {
    const int32_t mask_input_height = low_res_height * 4;
    const int32_t mask_input_width = low_res_width * 4;
    const auto low_res_area = static_cast<std::size_t>(low_res_height) * low_res_width;
    const auto video_area = static_cast<std::size_t>(video_height) * video_width;

    std::vector<std::vector<std::uint8_t>> video_foregrounds;
    video_foregrounds.reserve(detections.size());
    for (const auto* detection : detections) {
        if (detection == nullptr || detection->mask.size() != low_res_area) {
            throw std::runtime_error(
                "SAM3 tracker initialization received an invalid detector mask");
        }
        auto binary_input = resize_float_mask_bilinear(
            detection->mask, low_res_height, low_res_width, mask_input_height, mask_input_width);
        std::transform(binary_input.begin(), binary_input.end(), binary_input.begin(),
                       [](float value) { return value > 0.0F ? 1.0F : 0.0F; });
        const auto resized_video = resize_float_mask_antialiased(
            binary_input, mask_input_height, mask_input_width, video_height, video_width);
        std::vector<std::uint8_t> foreground(video_area);
        std::transform(resized_video.begin(), resized_video.end(), foreground.begin(),
                       [](float value) { return static_cast<std::uint8_t>(value > 0.5F); });
        video_foregrounds.push_back(std::move(foreground));
    }
    return video_foregrounds;
}

std::vector<std::vector<float>>
resolve_initial_tracker_overlaps(const std::vector<std::vector<std::uint8_t>>& video_foregrounds,
                                 int32_t video_height, int32_t video_width, int32_t low_res_height,
                                 int32_t low_res_width) {
    const auto video_area = static_cast<std::size_t>(video_height) * video_width;
    // Meta adds new objects in association order. Every later object carves
    // its foreground from every earlier object, independent of detector score.
    std::vector<std::uint8_t> claimed(video_area, 0U);
    std::vector<std::vector<float>> low_res_masks(video_foregrounds.size());
    for (std::size_t reverse = video_foregrounds.size(); reverse > 0; --reverse) {
        const std::size_t index = reverse - 1;
        std::vector<float> video_logits(video_area, -kInitialMaskLogit);
        const auto& foreground = video_foregrounds[index];
        for (std::size_t pixel = 0; pixel < video_area; ++pixel) {
            if (foreground[pixel] != 0U && claimed[pixel] == 0U)
                video_logits[pixel] = kInitialMaskLogit;
            if (foreground[pixel] != 0U)
                claimed[pixel] = 1U;
        }
        low_res_masks[index] = resize_float_mask_antialiased(
            video_logits, video_height, video_width, low_res_height, low_res_width);
    }
    return low_res_masks;
}

std::vector<std::vector<float>>
prepare_initial_tracker_masks(const std::vector<const Detection*>& detections,
                              int32_t low_res_height, int32_t low_res_width, int32_t video_height,
                              int32_t video_width) {
    validate_initial_tracker_mask_geometry(low_res_height, low_res_width, video_height,
                                           video_width);
    const auto foregrounds = make_initial_tracker_foregrounds(
        detections, low_res_height, low_res_width, video_height, video_width);
    return resolve_initial_tracker_overlaps(foregrounds, video_height, video_width, low_res_height,
                                            low_res_width);
}

void validate_result_overlap_state(const std::vector<std::uint32_t>* pixel_owners,
                                   const std::vector<float>* tracker_scores,
                                   std::size_t output_area) {
    if ((pixel_owners == nullptr) != (tracker_scores == nullptr)) {
        throw std::invalid_argument(
            "SAM3 fused result overlap requires both owner and score state");
    }
    if (pixel_owners != nullptr && pixel_owners->size() != output_area)
        throw std::invalid_argument("SAM3 fused result overlap owner geometry is invalid");
}

float resized_mask_logit(const std::vector<float>& source, const MaskResizePlan& plan,
                         const MaskResizeAxisEntry& y_entry, const MaskResizeAxisEntry& x_entry) {
    const auto at = [&](int32_t y, int32_t x) {
        return source[static_cast<std::size_t>(y) * plan.source_width + x];
    };
    const float top = at(y_entry.lower, x_entry.lower) * (1.0F - x_entry.weight) +
                      at(y_entry.lower, x_entry.upper) * x_entry.weight;
    const float bottom = at(y_entry.upper, x_entry.lower) * (1.0F - x_entry.weight) +
                         at(y_entry.upper, x_entry.upper) * x_entry.weight;
    return top * (1.0F - y_entry.weight) + bottom * y_entry.weight;
}

bool resolve_resized_mask_visibility(bool foreground, std::size_t pixel, std::size_t output_area,
                                     std::vector<std::uint32_t>* pixel_owners,
                                     const std::vector<float>* tracker_scores,
                                     float current_tracker_score, std::uint32_t current_object,
                                     std::vector<float>& output) {
    if (!foreground || pixel_owners == nullptr)
        return foreground;
    constexpr auto kNoOwner = std::numeric_limits<std::uint32_t>::max();
    auto& owner = (*pixel_owners)[pixel];
    if (owner == kNoOwner) {
        owner = current_object;
        return true;
    }
    if (owner >= tracker_scores->size())
        throw std::logic_error("SAM3 fused result overlap owner is invalid");
    if (current_tracker_score > (*tracker_scores)[owner]) {
        output[static_cast<std::size_t>(owner) * output_area + pixel] = 0.0F;
        owner = current_object;
        return true;
    }
    return false;
}

ResizedMaskBounds
append_resized_mask_rows(const std::vector<float>& source, const MaskResizePlan& plan,
                         std::size_t begin, std::size_t end, std::size_t output_offset,
                         std::size_t output_area, std::vector<std::uint32_t>* pixel_owners,
                         const std::vector<float>* tracker_scores, float current_tracker_score,
                         std::uint32_t current_object, std::vector<float>& output) {
    ResizedMaskBounds bounds(plan);
    for (std::size_t row = begin; row < end; ++row) {
        const auto y = static_cast<int32_t>(row);
        const auto& y_entry = plan.y[row];
        for (int32_t x = 0; x < plan.output_width; ++x) {
            const auto& x_entry = plan.x[static_cast<std::size_t>(x)];
            const bool foreground = resized_mask_logit(source, plan, y_entry, x_entry) > 0.0F;
            const auto pixel = row * plan.output_width + static_cast<std::size_t>(x);
            const bool visible = resolve_resized_mask_visibility(
                foreground, pixel, output_area, pixel_owners, tracker_scores, current_tracker_score,
                current_object, output);
            output[output_offset + pixel] = visible ? 1.0F : 0.0F;
            if (foreground)
                bounds.include(x, y);
        }
    }
    return bounds;
}

void merge_resized_mask_bounds(const ResizedMaskBounds& local, ResizedMaskBounds& combined,
                               std::mutex& mutex) {
    if (!local.has_foreground())
        return;
    std::lock_guard<std::mutex> lock(mutex);
    combined.merge(local);
}

ResizedMaskSummary append_resized_mask_to_frame(const std::vector<float>& source,
                                                const MaskResizePlan& plan,
                                                std::vector<float>& output,
                                                std::vector<std::uint32_t>* pixel_owners = nullptr,
                                                const std::vector<float>* tracker_scores = nullptr,
                                                float current_tracker_score = 0.0F,
                                                std::uint32_t current_object = 0) {
    const auto output_offset = output.size();
    const auto output_area = static_cast<std::size_t>(plan.output_height) * plan.output_width;
    validate_result_overlap_state(pixel_owners, tracker_scores, output_area);
    output.resize(output_offset + output_area);
    ResizedMaskBounds bounds(plan);
    std::mutex bounds_mutex;
    parallel_for_host_ranges(static_cast<std::size_t>(plan.output_height), [&](std::size_t begin,
                                                                               std::size_t end) {
        const auto local = append_resized_mask_rows(source, plan, begin, end, output_offset,
                                                    output_area, pixel_owners, tracker_scores,
                                                    current_tracker_score, current_object, output);
        merge_resized_mask_bounds(local, bounds, bounds_mutex);
    });
    if (!bounds.has_foreground()) {
        output.resize(output_offset);
        return {};
    }
    return {true,
            {static_cast<float>(bounds.min_x), static_cast<float>(bounds.min_y),
             static_cast<float>(bounds.max_x), static_cast<float>(bounds.max_y)}};
}

void filter_result_objects(Sam3VideoFrameResult& result,
                           const std::unordered_set<int32_t>& hidden) {
    if (hidden.empty() || result.object_ids.empty())
        return;
    const auto mask_area = static_cast<std::size_t>(result.height) * result.width;
    Sam3VideoFrameResult filtered;
    filtered.frame_idx = result.frame_idx;
    filtered.height = result.height;
    filtered.width = result.width;
    filtered.removed_object_ids = result.removed_object_ids;
    filtered.suppressed_object_ids = result.suppressed_object_ids;
    for (std::size_t index = 0; index < result.object_ids.size(); ++index) {
        if (hidden.count(result.object_ids[index]) != 0)
            continue;
        filtered.object_ids.push_back(result.object_ids[index]);
        filtered.detection_scores.push_back(result.detection_scores[index]);
        filtered.tracker_scores.push_back(result.tracker_scores[index]);
        filtered.masks.insert(filtered.masks.end(), result.masks.begin() + index * mask_area,
                              result.masks.begin() + (index + 1) * mask_area);
        filtered.boxes.insert(filtered.boxes.end(), result.boxes.begin() + index * 4,
                              result.boxes.begin() + (index + 1) * 4);
    }
    result = std::move(filtered);
}

void validate_video_text_input(const Sam3VideoTextInput& input) {
    if (input.features.empty() || input.features_shape.empty() || input.attention_mask.empty())
        throw std::invalid_argument("SAM3 video processor requires cached text features");
}

void validate_reviewed_memory_policy(const Sam3Config& config) {
    if (config.num_mask_memory_frames != 7 || config.max_conditioning_frames != 4 ||
        config.max_object_pointers != 16 || config.recondition_every_nth_frame != 16 ||
        config.max_video_frames != 1024 || config.max_conditioning_pointers != 4 ||
        config.max_pointer_inputs != 19) {
        throw std::runtime_error(
            "SAM3 video processor currently supports the reviewed SAM3.0 memory policy "
            "(7 mask memories, 4 conditioning memories, 16-frame reconditioning/pointer "
            "divisor, 1024 frames, 4 conditioning pointers, 19 total pointer inputs)");
    }
}

void validate_tracking_policy_bounds(const Sam3Config& config) {
    if (config.low_res_mask_size <= 0 || config.max_tracked_objects <= 0 ||
        config.hotstart_delay < 0 || config.hotstart_unmatch_threshold <= 0 ||
        config.hotstart_duplicate_threshold <= 0) {
        throw std::runtime_error("SAM3 video processor received invalid tracking policy bounds");
    }
}

std::vector<const TrackerFrameRecord*> conditioning_records(const TrackState& track,
                                                            int32_t frame_idx) {
    std::vector<const TrackerFrameRecord*> records;
    for (const auto& record : track.records) {
        if (record.conditioning && record.frame_idx < frame_idx && !record.object_pointer.empty())
            records.push_back(&record);
    }
    return records;
}

void order_and_limit_conditioning_memory(std::vector<const TrackerFrameRecord*>& records,
                                         int32_t frame_idx, int32_t records_seen, int32_t limit) {
    if (records_seen > limit || records.size() > static_cast<std::size_t>(limit)) {
        std::stable_sort(
            records.begin(), records.end(), [frame_idx](const auto* lhs, const auto* rhs) {
                return std::abs(frame_idx - lhs->frame_idx) < std::abs(frame_idx - rhs->frame_idx);
            });
    }
    if (records.size() > static_cast<std::size_t>(limit))
        records.resize(static_cast<std::size_t>(limit));
}

const TrackerFrameRecord* find_nonconditioning_record(const TrackState& track, int32_t frame_idx) {
    const auto iter =
        std::find_if(track.records.begin(), track.records.end(), [frame_idx](const auto& record) {
            return !record.conditioning && record.frame_idx == frame_idx;
        });
    return iter == track.records.end() ? nullptr : &*iter;
}

const TrackerFrameRecord* find_conditioning_record(const TrackState& track, int32_t frame_idx) {
    const auto iter =
        std::find_if(track.records.begin(), track.records.end(), [frame_idx](const auto& record) {
            return record.conditioning && record.frame_idx == frame_idx;
        });
    return iter == track.records.end() ? nullptr : &*iter;
}

std::vector<int32_t> quality_selected_frame_indices(const TrackState& track, int32_t frame_idx,
                                                    int32_t max_nonconditioning_frames) {
    std::vector<int32_t> recent_to_old;
    for (int32_t wanted = frame_idx - 1;
         wanted > 0 && recent_to_old.size() < static_cast<std::size_t>(max_nonconditioning_frames);
         --wanted) {
        const auto* record = find_nonconditioning_record(track, wanted);
        if (record != nullptr && record->effective_iou_score > kMemoryQualityThreshold)
            recent_to_old.push_back(wanted);
    }

    // Meta's frame filter always retains t-1, even if it is below the quality
    // threshold or is a conditioning frame.  It still consumes the newest
    // ordinal temporal slot when the selected conditioning bank already owns
    // that frame.
    const int32_t must_include = frame_idx - 1;
    if (std::find(recent_to_old.begin(), recent_to_old.end(), must_include) ==
        recent_to_old.end()) {
        recent_to_old.insert(recent_to_old.begin(), must_include);
    }
    std::reverse(recent_to_old.begin(), recent_to_old.end());
    return recent_to_old;
}

void validate_tracker_step_contract(const ITrtModule& engine) {
    if (!engine.has_input("memory_temporal_offsets") ||
        !engine.has_input("object_pointer_temporal_offsets") ||
        !engine.has_input("max_object_pointers_to_use") || !engine.has_output("selected_iou")) {
        throw std::runtime_error(
            "SAM3 tracker step plan uses an obsolete recurrent-state contract; rebuild the "
            "bundle so temporal encoding and memory-quality selection remain exact");
    }
}

void validate_tracker_history(const std::vector<const TrackerFrameRecord*>& memory_records,
                              const std::vector<const TrackerFrameRecord*>& pointer_records) {
    if (memory_records.empty() || pointer_records.empty())
        throw std::runtime_error("SAM3 tracker state has no conditioning memory");
}

AssociationPlan all_detections_are_new(std::size_t count) {
    AssociationPlan plan;
    plan.new_detection_indices.resize(count);
    std::iota(plan.new_detection_indices.begin(), plan.new_detection_indices.end(), 0);
    return plan;
}

AssociationPlan all_tracks_are_unmatched(const PropagatedTrackBatch& propagated) {
    AssociationPlan plan;
    for (std::size_t index = 0; index < propagated.object_ids.size(); ++index) {
        if (propagated.has_foreground[index])
            plan.unmatched_track_ids.push_back(propagated.object_ids[index]);
        else
            plan.empty_track_ids.push_back(propagated.object_ids[index]);
    }
    return plan;
}

std::vector<std::vector<float>> association_iou_matrix(const std::vector<Detection>& detections,
                                                       const PropagatedTrackBatch& propagated) {
    std::vector<std::vector<float>> ious(detections.size(),
                                         std::vector<float>(propagated.object_ids.size(), 0.0F));
    for (std::size_t detection = 0; detection < detections.size(); ++detection) {
        for (std::size_t track = 0; track < propagated.object_ids.size(); ++track) {
            ious[detection][track] =
                packed_mask_iou(detections[detection].binary_words, propagated.packed_masks[track]);
        }
    }
    return ious;
}

bool track_matches_detection(const std::vector<std::vector<float>>& ious, std::size_t track,
                             float threshold) {
    return std::any_of(ious.begin(), ious.end(),
                       [track, threshold](const auto& row) { return row[track] >= threshold; });
}

void classify_track_associations(AssociationPlan& plan, const PropagatedTrackBatch& propagated,
                                 const std::vector<std::vector<float>>& ious, float threshold) {
    for (std::size_t track = 0; track < propagated.object_ids.size(); ++track) {
        const int32_t object_id = propagated.object_ids[track];
        if (!propagated.has_foreground[track]) {
            plan.empty_track_ids.push_back(object_id);
        } else if (!track_matches_detection(ious, track, threshold)) {
            plan.unmatched_track_ids.push_back(object_id);
        }
    }
}

void classify_detection_association(AssociationPlan& plan, const std::vector<Detection>& detections,
                                    const std::vector<int32_t>& track_ids,
                                    const std::vector<std::vector<float>>& ious,
                                    std::size_t detection, const Sam3Config& config) {
    bool matches_any = false;
    float maximum_iou = -1.0F;
    std::size_t maximum_track = 0;
    for (std::size_t track = 0; track < track_ids.size(); ++track) {
        const float iou = ious[detection][track];
        if (iou >= config.association_iou_threshold) {
            matches_any = true;
            plan.detection_to_track_ids[static_cast<int32_t>(detection)].push_back(
                track_ids[track]);
        }
        if (iou > maximum_iou) {
            maximum_iou = iou;
            maximum_track = track;
        }
    }
    const bool is_new =
        detections[detection].score >= config.new_detection_threshold && !matches_any;
    if (is_new)
        plan.new_detection_indices.push_back(static_cast<int32_t>(detection));
    if (!is_new && detections[detection].score >= config.high_confidence_threshold &&
        maximum_iou >= config.high_iou_threshold) {
        plan.track_to_recondition_detection[track_ids[maximum_track]] =
            static_cast<int32_t>(detection);
    }
}

struct VisionFeatureBinding {
    ITrtModule* consumer{nullptr};
    std::string input_name;
    std::string output_name;
};

bool compatible_tensor_shapes(const std::vector<int64_t>& producer,
                              const std::vector<int64_t>& consumer) {
    if (producer.empty() || consumer.empty() || producer.size() != consumer.size())
        return false;
    for (std::size_t index = 0; index < producer.size(); ++index) {
        if (producer[index] > 0 && consumer[index] > 0 && producer[index] != consumer[index])
            return false;
    }
    return true;
}

void bind_sam3_external_safely(ITrtModule& module, const std::string& name, void* address) {
    if (address == nullptr)
        throw std::invalid_argument("SAM3 cannot bind a null tensor address: " + name);
    if (!module.has_input(name) && !module.has_output(name))
        throw std::invalid_argument("SAM3 cannot bind an unknown tensor: " + name);

    void* const original = module.device_ptr(name);
    // The generic backend historically treats bind_external(current_address)
    // as an ownership transfer. Avoid that call here so a module-owned buffer
    // cannot be relabeled external and leaked by a SAM3 no-op rebind.
    if (original == address)
        return;

    module.bind_external(name, address);
    if (module.device_ptr(name) == address)
        return;

    throw std::runtime_error("SAM3 failed to bind tensor address: " + name);
}

std::vector<VisionFeatureBinding> vision_feature_bindings(
    ITrtModule& core_engine, ITrtModule& tracker_init_engine, ITrtModule& tracker_step_engine,
    ITrtModule& tracker_memory_engine, ITrtModule& tracker_hard_memory_engine,
    ITrtModule& tracker_hard_memory_batch2_engine, ITrtModule& tracker_step_batch2_engine,
    ITrtModule& tracker_memory_batch2_engine, ITrtModule& parallel_tracker_init_engine) {
    std::vector<VisionFeatureBinding> bindings;
    bindings.reserve(24);
    for (int32_t level = 0; level < 3; ++level) {
        const auto suffix = std::to_string(level);
        bindings.push_back(
            {&core_engine, "sam3_fpn_hidden_" + suffix, "sam3_fpn_hidden_" + suffix});
        bindings.push_back(
            {&core_engine, "sam3_fpn_position_" + suffix, "sam3_fpn_position_" + suffix});
        bindings.push_back(
            {&tracker_init_engine, "tracker_feature_" + suffix, "sam3_tracker_feature_" + suffix});
        bindings.push_back({&parallel_tracker_init_engine, "tracker_feature_" + suffix,
                            "sam3_tracker_feature_" + suffix});
        bindings.push_back(
            {&tracker_step_engine, "tracker_feature_" + suffix, "sam3_tracker_feature_" + suffix});
        bindings.push_back({&tracker_step_batch2_engine, "tracker_feature_" + suffix,
                            "sam3_tracker_feature_" + suffix});
    }
    bindings.push_back({&tracker_step_engine, "tracker_position_2", "sam3_tracker_position_2"});
    bindings.push_back(
        {&tracker_step_batch2_engine, "tracker_position_2", "sam3_tracker_position_2"});
    bindings.push_back({&tracker_memory_engine, "tracker_feature_2", "sam3_tracker_feature_2"});
    bindings.push_back(
        {&tracker_hard_memory_engine, "tracker_feature_2", "sam3_tracker_feature_2"});
    bindings.push_back(
        {&tracker_hard_memory_batch2_engine, "tracker_feature_2", "sam3_tracker_feature_2"});
    bindings.push_back(
        {&tracker_memory_batch2_engine, "tracker_feature_2", "sam3_tracker_feature_2"});
    return bindings;
}

bool bind_device_resident_vision_features_impl(
    ITrtModule& vision_encoder, ITrtModule& core_engine, ITrtModule& tracker_init_engine,
    ITrtModule& tracker_step_engine, ITrtModule& tracker_memory_engine,
    ITrtModule& tracker_hard_memory_engine, ITrtModule& tracker_hard_memory_batch2_engine,
    ITrtModule& tracker_step_batch2_engine, ITrtModule& tracker_memory_batch2_engine,
    ITrtModule& parallel_tracker_init_engine) {
    auto bindings = vision_feature_bindings(
        core_engine, tracker_init_engine, tracker_step_engine, tracker_memory_engine,
        tracker_hard_memory_engine, tracker_hard_memory_batch2_engine, tracker_step_batch2_engine,
        tracker_memory_batch2_engine, parallel_tracker_init_engine);
    for (const auto& binding : bindings) {
        if (!vision_encoder.has_output(binding.output_name) ||
            !binding.consumer->has_input(binding.input_name)) {
            return false;
        }
        if (vision_encoder.device_ptr(binding.output_name) == nullptr ||
            vision_encoder.tensor_dtype(binding.output_name) !=
                binding.consumer->tensor_dtype(binding.input_name) ||
            !compatible_tensor_shapes(vision_encoder.tensor_shape(binding.output_name),
                                      binding.consumer->tensor_shape(binding.input_name))) {
            return false;
        }
    }
    for (const auto& binding : bindings) {
        void* output = vision_encoder.device_ptr(binding.output_name);
        bind_sam3_external_safely(*binding.consumer, binding.input_name, output);
    }
    return true;
}

std::shared_ptr<Sam3VideoVisionWorkspace> make_vision_workspace_impl(
    ITrtModule& vision_encoder, ITrtModule& core_engine, ITrtModule& tracker_init_engine,
    ITrtModule& tracker_step_engine, ITrtModule& tracker_memory_engine,
    ITrtModule& tracker_hard_memory_engine, ITrtModule& tracker_hard_memory_batch2_engine,
    ITrtModule& tracker_step_batch2_engine, ITrtModule& tracker_memory_batch2_engine,
    ITrtModule& parallel_tracker_init_engine) {
    if (!bind_device_resident_vision_features_impl(
            vision_encoder, core_engine, tracker_init_engine, tracker_step_engine,
            tracker_memory_engine, tracker_hard_memory_engine, tracker_hard_memory_batch2_engine,
            tracker_step_batch2_engine, tracker_memory_batch2_engine,
            parallel_tracker_init_engine)) {
        return nullptr;
    }
    return std::make_shared<Sam3VideoVisionWorkspace>();
}

void await_vision_outputs(ITrtModule& vision_encoder) {
    const auto stream = vision_encoder.stream();
    try {
        check_cuda(cudaStreamSynchronize(stream), "vision output completion");
    } catch (...) {
        (void)cudaStreamSynchronize(stream);
        throw;
    }
}

class Sam3VideoProcessorState {
    class ProcessingFailureGuard {
      public:
        explicit ProcessingFailureGuard(Sam3VideoProcessorState& state) noexcept
            : state_(state), uncaught_exceptions_(std::uncaught_exceptions()) {}
        ProcessingFailureGuard(const ProcessingFailureGuard&) = delete;
        ProcessingFailureGuard& operator=(const ProcessingFailureGuard&) = delete;
        ~ProcessingFailureGuard() noexcept {
            if (std::uncaught_exceptions() > uncaught_exceptions_)
                state_.quarantine_and_quiesce_after_failure();
        }

      private:
        Sam3VideoProcessorState& state_;
        int uncaught_exceptions_{0};
    };

  public:
    Sam3VideoProcessorState(
        ITrtModule& vision_encoder, ITrtModule& core_engine, ITrtModule& tracker_init_engine,
        ITrtModule& tracker_step_engine, ITrtModule& tracker_memory_engine,
        ITrtModule& tracker_hard_memory_engine, ITrtModule& tracker_hard_memory_batch2_engine,
        Sam3Config config, Sam3VideoTextInput text_input,
        std::shared_ptr<Sam3VideoVisionWorkspace> vision_workspace,
        ITrtModule& tracker_step_batch2_engine, ITrtModule& tracker_memory_batch2_engine,
        ITrtModule& parallel_tracker_init_engine, ITrtModule& hard_mask_resize_engine,
        ITrtModule& hard_mask_resize_batch2_engine)
        : vision_encoder_(vision_encoder), core_engine_(core_engine),
          tracker_init_engine_(tracker_init_engine),
          parallel_tracker_init_engine_(parallel_tracker_init_engine),
          tracker_step_engine_(tracker_step_engine),
          tracker_step_batch2_engine_(tracker_step_batch2_engine),
          tracker_memory_engine_(tracker_memory_engine),
          tracker_memory_batch2_engine_(tracker_memory_batch2_engine),
          tracker_hard_memory_engine_(tracker_hard_memory_engine),
          tracker_hard_memory_batch2_engine_(tracker_hard_memory_batch2_engine),
          hard_mask_resize_engine_(hard_mask_resize_engine),
          hard_mask_resize_batch2_engine_(hard_mask_resize_batch2_engine),
          config_(std::move(config)), text_input_(std::move(text_input)),
          vision_workspace_(std::move(vision_workspace)) {
        if (vision_workspace_ == nullptr)
            throw std::invalid_argument("SAM3 video requires its B1 vision workspace");
        check_cuda(cudaGetDevice(&cuda_device_), "SAM3 video CUDA device query");
        validate_video_text_input(text_input_);
        validate_reviewed_memory_policy(config_);
        validate_tracking_policy_bounds(config_);
        parallel_tracker_init_enabled_ =
            &parallel_tracker_init_engine_ != &tracker_init_engine_ &&
            tracker_init_engine_.stream() != nullptr &&
            parallel_tracker_init_engine_.stream() != nullptr &&
            tracker_init_engine_.stream() != parallel_tracker_init_engine_.stream();
        validate_tracker_device_memory_contract();
    }

    ~Sam3VideoProcessorState() noexcept {
        release_frame_zero_soft_refresh_inputs_on_owner_noexcept();
    }

    Sam3VideoFrameResult accept_prompt(const Sam3VideoFrame& frame) {
        std::lock_guard<std::mutex> lock(mutex_);
        check_cuda(cudaSetDevice(cuda_device_), "SAM3 video CUDA device selection");
        if (processed_frames_ != 0 || frame.frame_idx != 0)
            throw std::invalid_argument("SAM3 prompt processing requires frame zero");
        ProcessingFailureGuard failure_guard(*this);
        if (try_cuda_preprocess_batch1(frame))
            return process_frame(frame, true, -1, nullptr, true);
        auto pixels = preprocess_frame_pixels(frame, preprocess_workspace_);
        return process_frame(frame, true, -1, &pixels.view());
    }

    std::vector<Sam3VideoFrameResult>
    continue_borrowed(Sam3VideoFrameResult prompt_result,
                      const std::vector<Sam3VideoFrame>& remaining_frames, int32_t total_frames) {
        std::lock_guard<std::mutex> lock(mutex_);
        check_cuda(cudaSetDevice(cuda_device_), "SAM3 video CUDA device selection");
        if (processed_frames_ != 1 || prompt_result.frame_idx != 0 || total_frames < 1 ||
            static_cast<std::size_t>(total_frames) != remaining_frames.size() + 1U) {
            throw std::invalid_argument(
                "SAM3 stateful offline continuation requires prompt frame zero plus a complete "
                "remaining sequence");
        }
        for (std::size_t index = 0; index < remaining_frames.size(); ++index) {
            if (remaining_frames[index].frame_idx != static_cast<int32_t>(index + 1U)) {
                throw std::invalid_argument(
                    "SAM3 stateful offline continuation frames must be contiguous after zero");
            }
        }

        ProcessingFailureGuard failure_guard(*this);
        // Meta's propagate_in_video includes the already-consolidated prompt frame. Dense update
        // planning re-encodes that frame with soft-mask semantics and overwrites its conditioning
        // memory before frame one is allowed to consume it.
        auto propagated_frame_zero = prepare_frame_zero_propagation_result(prompt_result);
        refresh_frame_zero_conditioning_memories();
        std::vector<Sam3VideoFrameResult> results;
        results.reserve(static_cast<std::size_t>(total_frames));
        results.push_back(std::move(propagated_frame_zero));
        for (const auto& frame : remaining_frames) {
            if (try_cuda_preprocess_batch1(frame)) {
                results.push_back(process_frame(frame, false, total_frames, nullptr, true));
            } else {
                auto pixels = preprocess_frame_pixels(frame, preprocess_workspace_);
                results.push_back(process_frame(frame, false, total_frames, &pixels.view()));
            }
        }
        for (auto& result : results)
            filter_result_objects(result, removed_object_ids_);
        return results;
    }

  private:
    static bool has_device_float_output(const ITrtModule& module, const std::string& name) {
        return module.has_output(name) && module.device_ptr(name) != nullptr &&
               module.tensor_dtype(name) == DType::kFloat32;
    }

    static bool has_device_memory_pair(const ITrtModule& module, const std::string& feature_name,
                                       const std::string& position_name) {
        return has_device_float_output(module, feature_name) &&
               has_device_float_output(module, position_name);
    }

    static bool has_device_memory_inputs(const ITrtModule& module) {
        return module.device_ptr("memory_features") != nullptr &&
               module.device_ptr("memory_position") != nullptr;
    }

    void validate_tracker_device_memory_contract() const {
        if (!has_device_memory_pair(tracker_memory_engine_, "new_memory_features",
                                    "new_memory_position") ||
            !has_device_memory_pair(tracker_memory_batch2_engine_, "new_memory_features",
                                    "new_memory_position") ||
            !has_device_memory_pair(tracker_hard_memory_engine_, "new_memory_features",
                                    "new_memory_position") ||
            !has_device_memory_pair(tracker_hard_memory_batch2_engine_, "new_memory_features",
                                    "new_memory_position") ||
            !has_device_memory_inputs(tracker_step_engine_) ||
            !has_device_memory_inputs(tracker_step_batch2_engine_)) {
            throw std::runtime_error(
                "SAM3 tracker plans do not satisfy the device memory contract");
        }
    }

    void quarantine_and_quiesce_after_failure() noexcept {
        for (auto* memory : acquired_device_memories_) {
            if (memory != nullptr) {
                memory->reusable = false;
                memory->quarantined = true;
            }
        }
        const auto sync_noexcept = [](ITrtModule* module) noexcept {
            if (module == nullptr)
                return;
            try {
                module->sync();
            } catch (...) {
            }
        };
        if (vision_workspace_ != nullptr && vision_workspace_->cuda_preprocess != nullptr)
            vision_workspace_->cuda_preprocess->drain_noexcept();
        sync_noexcept(&vision_encoder_);
        sync_noexcept(&core_engine_);
        sync_noexcept(&tracker_init_engine_);
        sync_noexcept(&parallel_tracker_init_engine_);
        sync_noexcept(&tracker_step_engine_);
        sync_noexcept(&tracker_step_batch2_engine_);
        sync_noexcept(&tracker_memory_engine_);
        sync_noexcept(&tracker_memory_batch2_engine_);
        sync_noexcept(&tracker_hard_memory_engine_);
        sync_noexcept(&tracker_hard_memory_batch2_engine_);
        sync_noexcept(&hard_mask_resize_engine_);
        sync_noexcept(&hard_mask_resize_batch2_engine_);
    }

    bool cuda_preprocess_image_size_is_valid() const {
        if (config_.image_size <= 0)
            return false;
        const auto image_size = static_cast<std::size_t>(config_.image_size);
        if (image_size > std::numeric_limits<std::size_t>::max() / image_size)
            return false;
        return image_size * image_size <= std::numeric_limits<std::size_t>::max() / kChannels;
    }

    void* cuda_preprocess_output_if_compatible(const std::vector<int64_t>& expected_shape) {
        try {
            if (vision_encoder_.stream() == nullptr || !vision_encoder_.has_input("pixel_values") ||
                vision_encoder_.tensor_dtype("pixel_values") != DType::kFloat32 ||
                vision_encoder_.tensor_shape("pixel_values") != expected_shape) {
                return nullptr;
            }
            return vision_encoder_.device_ptr("pixel_values");
        } catch (...) {
            return nullptr;
        }
    }

    bool cuda_preprocess_binding_is_stable(const std::vector<int64_t>& expected_shape,
                                           const void* output) {
        try {
            return vision_encoder_.stream() != nullptr &&
                   vision_encoder_.tensor_dtype("pixel_values") == DType::kFloat32 &&
                   vision_encoder_.tensor_shape("pixel_values") == expected_shape &&
                   vision_encoder_.device_ptr("pixel_values") == output;
        } catch (...) {
            return false;
        }
    }

    std::shared_ptr<Sam3CudaPreprocessWorkspace> cuda_preprocess_workspace() {
        auto workspace = vision_workspace_->cuda_preprocess;
        if (workspace != nullptr)
            return workspace;
        try {
            workspace = std::make_shared<Sam3CudaPreprocessWorkspace>();
            vision_workspace_->cuda_preprocess = workspace;
            return workspace;
        } catch (...) {
            return nullptr;
        }
    }

    void enqueue_cuda_preprocess_batch1(const Sam3VideoFrame& frame,
                                        Sam3CudaPreprocessWorkspace& workspace, float* output) {
        bool queued = false;
        try {
            check_cuda(cudaMemsetAsync(workspace.device_nonfinite_status(), 0, sizeof(int),
                                       workspace.stream()),
                       "CUDA batch-1 preprocessing status reset");
            queued = true;
            const auto* plan = workspace.find_plan(frame.height, frame.width, config_.image_size);
            if (plan == nullptr)
                throw std::runtime_error("SAM3 CUDA batch-1 preprocessing plan disappeared");
            const auto device_plan = workspace.upload_plan(*plan);
            check_cuda(cudaMemcpyAsync(workspace.raw_input(), frame.pixel_data(),
                                       frame.pixel_count() * sizeof(float), cudaMemcpyHostToDevice,
                                       workspace.stream()),
                       "CUDA batch-1 preprocessing source upload");
            auto* horizontal = frame.width == config_.image_size ? nullptr : workspace.horizontal();
            const bool launched = sam3_cuda_preprocess_image(
                workspace.raw_input(), frame.height, frame.width, workspace.quantized(), horizontal,
                device_plan.horizontal_entries, device_plan.horizontal_entry_count,
                device_plan.horizontal_weights, device_plan.horizontal_weight_count,
                device_plan.horizontal_precision, device_plan.vertical_entries,
                device_plan.vertical_entry_count, device_plan.vertical_weights,
                device_plan.vertical_weight_count, device_plan.vertical_precision, output, 0,
                config_.image_size, config_.image_size, workspace.device_normalization_lut(),
                workspace.device_nonfinite_status(), workspace.stream());
            if (!launched) {
                throw std::runtime_error(
                    "SAM3 CUDA batch-1 preprocessing kernel rejected a preflighted frame");
            }
            check_cuda(cudaPeekAtLastError(), "CUDA batch-1 preprocessing kernel launch");
            check_cuda(cudaMemcpyAsync(workspace.host_nonfinite_status(),
                                       workspace.device_nonfinite_status(), sizeof(int),
                                       cudaMemcpyDeviceToHost, workspace.stream()),
                       "CUDA batch-1 preprocessing status download");
            check_cuda(cudaStreamSynchronize(workspace.stream()),
                       "CUDA batch-1 preprocessing synchronization");
        } catch (...) {
            if (queued)
                workspace.drain_noexcept();
            throw;
        }
    }

    static void validate_cuda_preprocess_status(const Sam3CudaPreprocessWorkspace& workspace) {
        const int status = *workspace.host_nonfinite_status();
        if ((status & 1) != 0)
            throw std::invalid_argument("SAM3 image pixels must be finite");
        if (status != 0) {
            throw std::runtime_error(
                "SAM3 CUDA batch-1 preprocessing device resize plan validation failed");
        }
    }

    bool try_cuda_preprocess_batch1(const Sam3VideoFrame& frame) {
        if (!cuda_preprocess_image_size_is_valid())
            return false;
        const std::vector<int64_t> expected_shape{1, kChannels, config_.image_size,
                                                  config_.image_size};
        void* output = cuda_preprocess_output_if_compatible(expected_shape);
        if (output == nullptr)
            return false;

        auto workspace = cuda_preprocess_workspace();
        if (workspace == nullptr)
            return false;
        // Passing a pointer/count keeps the initial owned frame in place. A
        // vector initializer here would copy a production frame's full pixel
        // payload merely to inspect one lane.
        if (!workspace->preflight(&frame, 1, config_.image_size, config_))
            return false;

        // Workspace allocation must not silently change the stable TensorRT
        // input binding selected above. Every rejection through this point is
        // prequeue and may use the canonical CPU/cache path.
        if (!cuda_preprocess_binding_is_stable(expected_shape, output))
            return false;

        enqueue_cuda_preprocess_batch1(frame, *workspace, static_cast<float*>(output));
        validate_cuda_preprocess_status(*workspace);
        return true;
    }

    PreprocessedFramePixels preprocess_frame_pixels(const Sam3VideoFrame& frame,
                                                    Sam3ImagePreprocessWorkspace& workspace) {
        if (frame.pixel_count() != checked_image_elements(frame.height, frame.width)) {
            throw std::invalid_argument(
                "SAM3 video frame pixel count does not match its dimensions");
        }
        auto& pixels = preprocess_sam3_image_into(frame.pixel_data(), frame.height, frame.width,
                                                  config_, workspace);
        return {std::move(pixels)};
    }

    void launch_device_frame_backbone(const Sam3VideoFrame& frame,
                                      const std::vector<float>& pixels) {
        if (frame.pixel_count() != checked_image_elements(frame.height, frame.width)) {
            throw std::invalid_argument(
                "SAM3 video frame pixel count does not match its dimensions");
        }
        const std::size_t expected_pixels =
            static_cast<std::size_t>(kChannels) * config_.image_size * config_.image_size;
        if (pixels.size() != expected_pixels)
            throw std::runtime_error("SAM3 preprocessed frame has an unexpected tensor size");
        Tensor pixel_tensor;
        pixel_tensor.data = const_cast<float*>(pixels.data());
        pixel_tensor.shape = {1, kChannels, config_.image_size, config_.image_size};
        pixel_tensor.dtype = DType::kFloat32;
        vision_encoder_.forward_async({{"pixel_values", pixel_tensor}});
    }

    Sam3FrameFeatures finish_device_frame_backbone() {
        // The vision outputs are shared directly with all downstream engines.  Fence the producer
        // once before any consumer stream uses those externally-bound addresses.
        await_vision_outputs(vision_encoder_);
        Sam3FrameFeatures result;
        result.mask_height = config_.low_res_mask_size;
        result.mask_width = config_.low_res_mask_size;
        return result;
    }

    Sam3FrameFeatures run_frame_backbone(const Sam3VideoFrame& frame,
                                         const std::vector<float>* preprocessed_pixels = nullptr,
                                         bool cuda_preprocessed_input_ready = false) {
        if (frame.pixel_count() != checked_image_elements(frame.height, frame.width)) {
            throw std::invalid_argument(
                "SAM3 video frame pixel count does not match its dimensions");
        }
        if (cuda_preprocessed_input_ready) {
            if (preprocessed_pixels != nullptr) {
                throw std::runtime_error(
                    "SAM3 CUDA-preprocessed frame reached an incompatible vision path");
            }
            vision_encoder_.forward_async({});
            return finish_device_frame_backbone();
        }
        auto& pixels =
            preprocessed_pixels != nullptr
                ? *preprocessed_pixels
                : preprocess_sam3_image_into(frame.pixel_data(), frame.height, frame.width, config_,
                                             preprocess_workspace_);
        launch_device_frame_backbone(frame, pixels);
        return finish_device_frame_backbone();
    }

    void retain_device_frame_zero_tracker_feature_2() {
        constexpr const char* output_name = "sam3_tracker_feature_2";
        if (!vision_encoder_.has_output(output_name) ||
            vision_encoder_.tensor_dtype(output_name) != DType::kFloat32) {
            throw std::runtime_error(
                "SAM3 frame-zero tracker feature has an invalid device contract");
        }
        const auto shape = vision_encoder_.tensor_shape(output_name);
        const auto values = static_shape_values(shape);
        const auto* source = vision_encoder_.device_ptr(output_name);
        if (values == 0 || source == nullptr) {
            throw std::runtime_error("SAM3 frame-zero tracker feature has no device storage");
        }
        DeviceTensor retained(shape, DType::kFloat32, vision_encoder_.stream());
        if (!retained.ok() || retained.nbytes() != values * sizeof(float)) {
            throw std::runtime_error("SAM3 frame-zero tracker feature allocation failed");
        }
        check_cuda(cudaMemcpyAsync(retained.data(), source, retained.nbytes(),
                                   cudaMemcpyDeviceToDevice, vision_encoder_.stream()),
                   "frame-zero tracker feature retention");
        check_cuda(cudaStreamSynchronize(vision_encoder_.stream()),
                   "frame-zero tracker feature retention completion");
        frame_zero_tracker_feature_2_ = std::move(retained);
        frame_zero_tracker_feature_2_shape_ = shape;
    }

    void retain_frame_zero_tracker_feature_2() {
        if (frame_zero_soft_memory_refreshed_)
            throw std::logic_error("SAM3 frame-zero memory was already refreshed");
        retain_device_frame_zero_tracker_feature_2();
    }

    TensorMap detector_inputs() {
        Tensor text_features;
        text_features.data = text_input_.features.data();
        text_features.shape = batched_text_shape(text_input_.features_shape);
        text_features.dtype = DType::kFloat32;
        Tensor text_mask;
        text_mask.data = text_input_.attention_mask.data();
        text_mask.shape = {1, static_cast<int64_t>(text_input_.attention_mask.size())};
        text_mask.dtype = DType::kInt32;
        TensorMap core_inputs;
        core_inputs["sam3_text_features"] = text_features;
        core_inputs["sam3_text_attention_mask"] = text_mask;
        return core_inputs;
    }

    static bool supports_sparse_detector_download(ITrtModule& engine) {
        return engine.has_output("pred_masks") && engine.has_output("pred_logits") &&
               engine.device_ptr("pred_masks") != nullptr &&
               engine.device_ptr("pred_logits") != nullptr &&
               engine.tensor_dtype("pred_masks") == DType::kFloat32 &&
               engine.tensor_dtype("pred_logits") == DType::kFloat32;
    }

    static void validate_detector_output_geometry(Sam3FrameFeatures& result,
                                                  const std::vector<int64_t>& masks_shape,
                                                  const std::vector<int64_t>& logits_shape) {
        int32_t mask_queries = 0;
        int32_t mask_height = 0;
        int32_t mask_width = 0;
        if (!mask_geometry(masks_shape, mask_queries, mask_height, mask_width))
            throw std::runtime_error("SAM3 video core engine returned invalid mask geometry");
        if (mask_height != result.mask_height || mask_width != result.mask_width)
            throw std::runtime_error("SAM3 video core engine returned unexpected mask geometry");
        result.detector_queries = query_count(logits_shape);
        if (result.detector_queries <= 0 || mask_queries != result.detector_queries)
            throw std::runtime_error("SAM3 video detector query dimensions do not align");
    }

    static float detector_presence_from_outputs(const TensorMap& outputs) {
        const auto presence_iter = outputs.find("presence_logits");
        if (presence_iter == outputs.end() || presence_iter->second.data == nullptr)
            return 1.0F;
        const auto presence_logits = copy_float_tensor(presence_iter->second);
        return presence_logits.empty() ? 1.0F : sigmoid(presence_logits.front());
    }

    static void populate_detector_host_outputs(Sam3FrameFeatures& result,
                                               const TensorMap& outputs) {
        const Tensor masks = require_output(outputs, "pred_masks", "core engine");
        const Tensor logits = require_output(outputs, "pred_logits", "core engine");
        validate_detector_output_geometry(result, masks.shape, logits.shape);
        result.detector_masks = copy_float_tensor(masks);
        result.detector_mask_queries.resize(static_cast<std::size_t>(result.detector_queries));
        std::iota(result.detector_mask_queries.begin(), result.detector_mask_queries.end(), 0);
        const auto raw_logits = copy_float_tensor(logits);
        if (raw_logits.size() < static_cast<std::size_t>(result.detector_queries))
            throw std::runtime_error("SAM3 video core engine returned too few detection logits");
        const float presence = detector_presence_from_outputs(outputs);
        result.detector_scores.reserve(static_cast<std::size_t>(result.detector_queries));
        for (int32_t query = 0; query < result.detector_queries; ++query) {
            result.detector_scores.push_back(sigmoid(raw_logits[static_cast<std::size_t>(query)]) *
                                             presence);
        }
    }

    static bool supports_device_presence_download(ITrtModule& engine) {
        return engine.has_output("presence_logits") &&
               engine.device_ptr("presence_logits") != nullptr &&
               engine.tensor_dtype("presence_logits") == DType::kFloat32;
    }

    void populate_detector_device_scores(Sam3FrameFeatures& result, ITrtModule& engine,
                                         std::vector<float>& raw_logits) {
        const auto stream = engine.stream();
        check_cuda(cudaMemcpyAsync(raw_logits.data(), engine.device_ptr("pred_logits"),
                                   raw_logits.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                   stream),
                   "detector logits download");
        float presence_logit = 0.0F;
        const bool has_presence = supports_device_presence_download(engine);
        if (has_presence) {
            check_cuda(cudaMemcpyAsync(&presence_logit, engine.device_ptr("presence_logits"),
                                       sizeof(float), cudaMemcpyDeviceToHost, stream),
                       "detector presence download");
        }
        engine.sync();
        const float presence = has_presence ? sigmoid(presence_logit) : 1.0F;
        result.detector_scores.reserve(static_cast<std::size_t>(result.detector_queries));
        for (int32_t query = 0; query < result.detector_queries; ++query) {
            result.detector_scores.push_back(sigmoid(raw_logits[static_cast<std::size_t>(query)]) *
                                             presence);
            if (result.detector_scores.back() > config_.detection_threshold)
                result.detector_mask_queries.push_back(query);
        }
    }

    static void download_detector_candidate_masks(Sam3FrameFeatures& result, ITrtModule& engine,
                                                  int32_t mask_height, int32_t mask_width) {
        const auto mask_area = static_cast<std::size_t>(mask_height) * mask_width;
        result.detector_masks.resize(result.detector_mask_queries.size() * mask_area);
        const auto* device_masks = static_cast<const float*>(engine.device_ptr("pred_masks"));
        const auto mask_bytes = mask_area * sizeof(float);
        const auto stream = engine.stream();
        for (std::size_t compact = 0; compact < result.detector_mask_queries.size(); ++compact) {
            const auto query = static_cast<std::size_t>(result.detector_mask_queries[compact]);
            check_cuda(cudaMemcpyAsync(result.detector_masks.data() + compact * mask_area,
                                       device_masks + query * mask_area, mask_bytes,
                                       cudaMemcpyDeviceToHost, stream),
                       "detector candidate mask download");
        }
        if (!result.detector_mask_queries.empty())
            engine.sync();
    }

    void run_frame_detector_with_engine(Sam3FrameFeatures& result, ITrtModule& core_engine,
                                        const EngineEnqueueCallback& after_enqueue = {}) {
        auto core_inputs = detector_inputs();
        if (!supports_sparse_detector_download(core_engine)) {
            const auto core_outputs = core_engine.forward(core_inputs);
            if (after_enqueue)
                after_enqueue(false, nullptr);
            populate_detector_host_outputs(result, core_outputs);
            return;
        }

        core_engine.forward_async(core_inputs);
        ModuleSyncGuard core_sync_guard(core_engine);
        if (after_enqueue)
            after_enqueue(true, core_engine.stream());
        const auto masks_shape = core_engine.tensor_shape("pred_masks");
        const auto logits_shape = core_engine.tensor_shape("pred_logits");
        int32_t mask_queries = 0;
        int32_t mask_height = 0;
        int32_t mask_width = 0;
        if (!mask_geometry(masks_shape, mask_queries, mask_height, mask_width))
            throw std::runtime_error("SAM3 video core engine returned invalid mask geometry");
        validate_detector_output_geometry(result, masks_shape, logits_shape);

        std::vector<float> raw_logits(static_cast<std::size_t>(result.detector_queries));
        populate_detector_device_scores(result, core_engine, raw_logits);
        download_detector_candidate_masks(result, core_engine, mask_height, mask_width);
    }

    void run_frame_detector(Sam3FrameFeatures& result,
                            const EngineEnqueueCallback& after_enqueue = {}) {
        run_frame_detector_with_engine(result, core_engine_, after_enqueue);
    }

    std::shared_ptr<DeviceEncodedMemory> acquire_device_memory(std::size_t values,
                                                               cudaStream_t stream) {
        std::lock_guard<std::mutex> lock(device_memory_pool_mutex_);
        auto& pool = vision_workspace_->recurrent_device_memory_pool;
        const auto available =
            std::find_if(pool.begin(), pool.end(), [this, values](const auto& memory) {
                return memory.use_count() == 1 && memory->reusable && !memory->quarantined &&
                       memory->values == values && memory->cuda_device == cuda_device_;
            });
        if (available != pool.end()) {
            (*available)->reusable = false;
            acquired_device_memories_.insert(available->get());
            return *available;
        }
        auto memory = std::make_shared<DeviceEncodedMemory>(values, stream, cuda_device_);
        if (!memory->features.ok() || !memory->position.ok())
            throw std::runtime_error("SAM3 video failed to allocate device recurrent memory");
        pool.push_back(memory);
        acquired_device_memories_.insert(memory.get());
        return memory;
    }

    TrackerNeuralOutput initialize_track_with_engine(const Sam3FrameFeatures& features,
                                                     const std::vector<float>& detector_mask,
                                                     ITrtModule& tracker_init_engine) {
        Tensor mask;
        // The direct TensorRT init graph owns Meta's mask-input transform:
        // resize signed 288x288 detector logits to 1152x1152, then threshold
        // at zero.  Keep the raw logits here so the operation order is exact.
        mask.data = const_cast<float*>(detector_mask.data());
        mask.shape = {1, 1, features.mask_height, features.mask_width};
        mask.dtype = DType::kFloat32;
        TensorMap inputs;
        inputs["detector_mask"] = mask;
        auto& staging = &tracker_init_engine == &parallel_tracker_init_engine_
                            ? vision_workspace_->parallel_tracker_init_output_staging
                            : vision_workspace_->tracker_init_output_staging;
        TensorMap outputs;
        forward_tracker_head_sparse(tracker_init_engine, inputs, 1, features.mask_height,
                                    features.mask_width, staging, outputs);
        return parse_tracker_init_output(outputs);
    }

    TrackerNeuralOutput initialize_track(const Sam3FrameFeatures& features,
                                         const std::vector<float>& detector_mask) {
        return initialize_track_with_engine(features, detector_mask, tracker_init_engine_);
    }

    std::vector<const TrackerFrameRecord*> selected_conditioning_records(const TrackState& track,
                                                                         int32_t frame_idx) const {
        auto selected = conditioning_records(track, frame_idx);
        order_and_limit_conditioning_memory(selected, frame_idx, track.conditioning_records_seen,
                                            config_.max_conditioning_frames);
        return selected;
    }

    void select_memory_records(const TrackState& track, int32_t frame_idx,
                               TrackerStepRequest& request) const {
        const auto selected_conditioning = selected_conditioning_records(track, frame_idx);
        std::unordered_set<int32_t> selected_conditioning_frames;
        for (const auto* record : selected_conditioning) {
            selected_conditioning_frames.insert(record->frame_idx);
            if (!record_has_memory(*record))
                continue;
            request.memory_records.push_back(record);
            request.memory_temporal_offsets.push_back(0);
        }

        const auto valid_indices =
            quality_selected_frame_indices(track, frame_idx, config_.max_object_pointers - 1);
        const auto memory_slots = static_cast<std::size_t>(config_.num_mask_memory_frames - 1);
        const auto first =
            valid_indices.size() > memory_slots ? valid_indices.size() - memory_slots : 0;
        for (std::size_t index = first; index < valid_indices.size(); ++index) {
            const int32_t selected_frame = valid_indices[index];
            const auto* record = find_nonconditioning_record(track, selected_frame);
            if (record == nullptr && selected_conditioning_frames.count(selected_frame) == 0)
                record = find_conditioning_record(track, selected_frame);
            if (record == nullptr || !record_has_memory(*record))
                continue;
            request.memory_records.push_back(record);
            request.memory_temporal_offsets.push_back(
                static_cast<int32_t>(valid_indices.size() - index));
        }
    }

    void select_pointer_records(const TrackState& track, int32_t frame_idx,
                                TrackerStepRequest& request) const {
        const auto selected_conditioning = selected_conditioning_records(track, frame_idx);
        std::unordered_set<int32_t> selected_conditioning_frames;
        for (const auto* record : selected_conditioning) {
            selected_conditioning_frames.insert(record->frame_idx);
            request.pointer_records.push_back(record);
            request.pointer_temporal_offsets.push_back(frame_idx - record->frame_idx);
        }

        const auto valid_indices =
            quality_selected_frame_indices(track, frame_idx, config_.max_object_pointers - 1);
        const auto pointer_slots = static_cast<std::size_t>(config_.max_object_pointers - 1);
        // Preserve Meta's reviewed frame_filter/pointer-loop boundary exactly.  Its
        // t_diff loop stops when t_diff reaches len(valid_indices), so the final
        // selected index is intentionally reserved for spatial memory and is not
        // also appended as a non-conditioning pointer.  This yields pointer counts
        // 1, 1, 2, 3 for recurrent frames 1..4 in a five-frame session.
        const auto nonconditioning_pointer_count =
            valid_indices.empty() ? 0U : valid_indices.size() - 1U;
        for (std::size_t ordinal = 1;
             ordinal <= pointer_slots && ordinal <= nonconditioning_pointer_count; ++ordinal) {
            const int32_t selected_frame = valid_indices[valid_indices.size() - ordinal];
            const auto* record = find_nonconditioning_record(track, selected_frame);
            if (record == nullptr && selected_conditioning_frames.count(selected_frame) == 0)
                record = find_conditioning_record(track, selected_frame);
            if (record == nullptr || record->object_pointer.empty())
                continue;
            request.pointer_records.push_back(record);
            request.pointer_temporal_offsets.push_back(static_cast<int32_t>(ordinal));
        }

        if (request.pointer_records.size() > static_cast<std::size_t>(config_.max_pointer_inputs)) {
            throw std::runtime_error("SAM3 pointer history exceeds the reviewed P19 profile");
        }
    }

    TrackerStepRequest make_tracker_step_request(int32_t object_id, const TrackState& track,
                                                 int32_t frame_idx) const {
        TrackerStepRequest request;
        request.object_id = object_id;
        select_memory_records(track, frame_idx, request);
        select_pointer_records(track, frame_idx, request);
        validate_tracker_history(request.memory_records, request.pointer_records);
        return request;
    }

    static void validate_tracker_batch_slice(const std::vector<TrackerStepRequest>& requests,
                                             std::size_t begin, std::size_t count) {
        if (count == 0 || count > kTrackerStepBatch2Size || begin + count > requests.size())
            throw std::invalid_argument("SAM3 tracker step received an invalid batch slice");
    }

    static bool tracker_record_device_storage_is_valid(const TrackerFrameRecord& record) {
        return record.device_memory != nullptr && record.device_memory->features.ok() &&
               record.device_memory->position.ok();
    }

    static void validate_tracker_record_geometry(const TrackerFrameRecord& record,
                                                 std::size_t expected_values,
                                                 const char* error_message) {
        const auto values = record_memory_values(record);
        if (values != expected_values || !tracker_record_device_storage_is_valid(record))
            throw std::runtime_error(error_message);
    }

    static std::size_t validate_first_tracker_memory(const TrackerFrameRecord& record) {
        const auto values = record_memory_values(record);
        if (values == 0 || values % kTrackerMemoryChannels != 0)
            throw std::runtime_error("SAM3 tracker step received invalid recurrent memory");
        validate_tracker_record_geometry(record, values,
                                         "SAM3 tracker step received invalid recurrent memory");
        return values / kTrackerMemoryChannels;
    }

    static void validate_tracker_request_shape(const TrackerStepRequest& request,
                                               std::size_t memory_count,
                                               std::size_t pointer_count) {
        if (request.memory_records.size() != memory_count ||
            request.memory_temporal_offsets.size() != memory_count ||
            request.pointer_records.size() != pointer_count ||
            request.pointer_temporal_offsets.size() != pointer_count) {
            throw std::runtime_error("SAM3 tracker step batch mixed incompatible history shapes");
        }
    }

    static void validate_tracker_request_memories(const TrackerStepRequest& request,
                                                  std::size_t expected_values) {
        for (const auto* record : request.memory_records) {
            validate_tracker_record_geometry(
                *record, expected_values,
                "SAM3 tracker step batch mixed incompatible memory geometry");
        }
    }

    static void validate_tracker_request_pointers(const TrackerStepRequest& request) {
        for (const auto* record : request.pointer_records) {
            if (record->object_pointer.size() != kObjectPointerChannels)
                throw std::runtime_error("SAM3 tracker step received an invalid object pointer");
        }
    }

    std::size_t validate_tracker_step_batch(const std::vector<TrackerStepRequest>& requests,
                                            std::size_t begin, std::size_t count) const {
        validate_tracker_batch_slice(requests, begin, count);
        const auto memory_count = requests[begin].memory_records.size();
        const auto pointer_count = requests[begin].pointer_records.size();
        const std::size_t spatial_tokens =
            validate_first_tracker_memory(*requests[begin].memory_records.front());
        for (std::size_t batch = begin; batch < begin + count; ++batch) {
            const auto& request = requests[batch];
            validate_tracker_request_shape(request, memory_count, pointer_count);
            validate_tracker_request_memories(request, spatial_tokens * kTrackerMemoryChannels);
            validate_tracker_request_pointers(request);
        }
        return spatial_tokens;
    }

    void populate_tracker_step_position_template(ITrtModule& tracker_step_engine,
                                                 TrackerStepPositionTemplate& position_template,
                                                 const DeviceEncodedMemory& canonical_memory,
                                                 float* destination_position,
                                                 std::size_t required_values) {
        const auto values_per_record = canonical_memory.values;
        if (destination_position == nullptr || !canonical_memory.position.ok() ||
            values_per_record == 0 || required_values % values_per_record != 0) {
            throw std::runtime_error("SAM3 tracker position template has an invalid layout");
        }

        // TensorRT owns a persistent input buffer for each step engine. Populate its repeated
        // position prefix once, then extend only when a later frame selects more history. The
        // position tensor is invariant across memory records, so one canonical device record
        // preserves the exact bytes while avoiding a copy for every selected record and frame.
        if (position_template.destination != destination_position ||
            position_template.values_per_record != values_per_record) {
            position_template.destination = destination_position;
            position_template.values_per_record = values_per_record;
            position_template.populated_values = 0;
        }
        if (position_template.populated_values >= required_values)
            return;

        const auto bytes_per_record = values_per_record * sizeof(float);
        const auto stream = tracker_step_engine.stream();
        for (std::size_t offset = position_template.populated_values; offset < required_values;
             offset += values_per_record) {
            check_cuda(cudaMemcpyAsync(destination_position + offset,
                                       canonical_memory.position.data(), bytes_per_record,
                                       cudaMemcpyDeviceToDevice, stream),
                       "tracker position template copy");
        }
        position_template.populated_values = required_values;
    }

    static void validate_tracker_step_engine_batch(bool batch2_layout, std::size_t count) {
        if (batch2_layout && count != kTrackerStepBatch2Size)
            throw std::invalid_argument("SAM3 tracker step received the wrong engine batch size");
        if (!batch2_layout && count != 1)
            throw std::invalid_argument("SAM3 tracker step received the wrong engine batch size");
    }

    void prepare_tracker_step_scratch(const std::vector<TrackerStepRequest>& requests,
                                      std::size_t begin, std::size_t count,
                                      std::size_t memory_count, std::size_t pointer_count) {
        memory_offsets_scratch_.clear();
        memory_offsets_scratch_.reserve(count * memory_count);
        pointers_scratch_.clear();
        pointer_offsets_scratch_.clear();
        pointers_scratch_.reserve(count * pointer_count * kObjectPointerChannels);
        pointer_offsets_scratch_.reserve(count * pointer_count);
        for (std::size_t batch = begin; batch < begin + count; ++batch) {
            const auto& request = requests[batch];
            for (std::size_t memory_index = 0; memory_index < request.memory_records.size();
                 ++memory_index) {
                memory_offsets_scratch_.push_back(request.memory_temporal_offsets[memory_index]);
            }
            for (std::size_t pointer_index = 0; pointer_index < request.pointer_records.size();
                 ++pointer_index) {
                const auto* record = request.pointer_records[pointer_index];
                pointers_scratch_.insert(pointers_scratch_.end(), record->object_pointer.begin(),
                                         record->object_pointer.end());
                pointer_offsets_scratch_.push_back(request.pointer_temporal_offsets[pointer_index]);
            }
        }
    }

    static Tensor tracker_memory_tensor(std::size_t count, std::size_t memory_count,
                                        std::size_t spatial_tokens) {
        Tensor tensor;
        tensor.data = nullptr;
        tensor.shape = {static_cast<int64_t>(count), static_cast<int64_t>(memory_count),
                        static_cast<int64_t>(spatial_tokens), kTrackerMemoryChannels};
        tensor.dtype = DType::kFloat32;
        return tensor;
    }

    static Tensor tracker_memory_position_tensor(const Tensor& memory) {
        Tensor tensor;
        tensor.data = nullptr;
        tensor.shape = memory.shape;
        tensor.dtype = DType::kFloat32;
        return tensor;
    }

    Tensor tracker_memory_offset_tensor(bool /*batch2_layout*/, std::size_t count,
                                        std::size_t memory_count) {
        Tensor tensor;
        tensor.data = memory_offsets_scratch_.data();
        tensor.shape = {static_cast<int64_t>(count), static_cast<int64_t>(memory_count)};
        tensor.dtype = DType::kInt32;
        return tensor;
    }

    Tensor tracker_pointer_tensor(bool /*batch2_layout*/, std::size_t count,
                                  std::size_t pointer_count) {
        Tensor tensor;
        tensor.data = pointers_scratch_.data();
        tensor.shape = {static_cast<int64_t>(count), static_cast<int64_t>(pointer_count),
                        kObjectPointerChannels};
        tensor.dtype = DType::kFloat32;
        return tensor;
    }

    Tensor tracker_pointer_offset_tensor(bool /*batch2_layout*/, std::size_t count,
                                         std::size_t pointer_count) {
        Tensor tensor;
        tensor.data = pointer_offsets_scratch_.data();
        tensor.shape = {static_cast<int64_t>(count), static_cast<int64_t>(pointer_count)};
        tensor.dtype = DType::kInt32;
        return tensor;
    }

    static Tensor tracker_max_pointer_tensor(int32_t& max_pointers_to_use) {
        Tensor tensor;
        tensor.data = &max_pointers_to_use;
        tensor.shape = {1};
        tensor.dtype = DType::kInt32;
        return tensor;
    }

    void copy_tracker_device_memories(const std::vector<TrackerStepRequest>& requests,
                                      std::size_t begin, std::size_t count,
                                      std::size_t memory_values, ITrtModule& engine,
                                      bool batch2_layout) {
        auto* destination_features = static_cast<float*>(engine.device_ptr("memory_features"));
        auto* destination_position = static_cast<float*>(engine.device_ptr("memory_position"));
        const auto stream = engine.stream();
        std::size_t destination_offset = 0;
        for (std::size_t batch = begin; batch < begin + count; ++batch) {
            for (const auto* record : requests[batch].memory_records) {
                const auto values = record->device_memory->values;
                check_cuda(cudaMemcpyAsync(destination_features + destination_offset,
                                           record->device_memory->features.data(),
                                           values * sizeof(float), cudaMemcpyDeviceToDevice,
                                           stream),
                           "tracker feature memory copy");
                destination_offset += values;
            }
        }
        if (destination_offset != memory_values)
            throw std::runtime_error("SAM3 tracker device memory layout mismatch");
        auto& position_template = batch2_layout ? tracker_step_batch2_position_template_
                                                : tracker_step_position_template_;
        populate_tracker_step_position_template(
            engine, position_template, *requests[begin].memory_records.front()->device_memory,
            destination_position, memory_values);
    }

    static TensorMap tracker_step_inputs(const Tensor& memory, const Tensor& memory_position,
                                         const Tensor& memory_offsets, const Tensor& pointers,
                                         const Tensor& pointer_offsets,
                                         const Tensor& max_pointers) {
        TensorMap inputs;
        inputs["memory_features"] = memory;
        inputs["memory_position"] = memory_position;
        inputs["memory_temporal_offsets"] = memory_offsets;
        inputs["object_pointers"] = pointers;
        inputs["object_pointer_temporal_offsets"] = pointer_offsets;
        inputs["max_object_pointers_to_use"] = max_pointers;
        return inputs;
    }

    std::vector<TrackerNeuralOutput>
    execute_tracker_step(const Sam3FrameFeatures& features, ITrtModule& engine,
                         const TensorMap& inputs, std::size_t count, bool batch2_layout,
                         const EngineEnqueueCallback& after_enqueue) {
        auto& staging =
            batch2_layout ? tracker_step_batch2_output_staging() : tracker_step_output_staging();
        TensorMap outputs;
        forward_tracker_head_sparse(engine, inputs, count, features.mask_height,
                                    features.mask_width, staging, outputs, after_enqueue);
        return parse_tracker_head_outputs(outputs, count, features.mask_height, features.mask_width,
                                          batch2_layout ? "tracker batch2 step engine"
                                                        : "tracker step engine");
    }

    std::vector<TrackerNeuralOutput>
    propagate_track_requests(const Sam3FrameFeatures& features,
                             const std::vector<TrackerStepRequest>& requests, std::size_t begin,
                             std::size_t count, int32_t /*frame_idx*/, bool streaming,
                             int32_t total_frames, ITrtModule& tracker_step_engine,
                             bool batch2_layout, const EngineEnqueueCallback& after_enqueue = {}) {
        validate_tracker_step_engine_batch(batch2_layout, count);
        validate_tracker_step_contract(tracker_step_engine);
        const std::size_t spatial_tokens = validate_tracker_step_batch(requests, begin, count);
        const auto memory_count = requests[begin].memory_records.size();
        const auto pointer_count = requests[begin].pointer_records.size();
        const std::size_t memory_values =
            count * memory_count * spatial_tokens * kTrackerMemoryChannels;
        prepare_tracker_step_scratch(requests, begin, count, memory_count, pointer_count);
        int32_t max_pointers_to_use =
            streaming ? config_.max_object_pointers
                      : std::min(std::max(total_frames, 1), config_.max_object_pointers);

        const Tensor memory = tracker_memory_tensor(count, memory_count, spatial_tokens);
        const Tensor memory_pos = tracker_memory_position_tensor(memory);
        const Tensor memory_offset_tensor =
            tracker_memory_offset_tensor(batch2_layout, count, memory_count);
        const Tensor pointer_tensor = tracker_pointer_tensor(batch2_layout, count, pointer_count);
        const Tensor pointer_offset_tensor =
            tracker_pointer_offset_tensor(batch2_layout, count, pointer_count);
        const Tensor max_pointer_tensor = tracker_max_pointer_tensor(max_pointers_to_use);

        copy_tracker_device_memories(requests, begin, count, memory_values, tracker_step_engine,
                                     batch2_layout);

        const auto inputs =
            tracker_step_inputs(memory, memory_pos, memory_offset_tensor, pointer_tensor,
                                pointer_offset_tensor, max_pointer_tensor);
        return execute_tracker_step(features, tracker_step_engine, inputs, count, batch2_layout,
                                    after_enqueue);
    }

    float* pack_final_memory_masks_batch2(
        const std::array<const std::vector<float>*, kTrackerStepBatch2Size>& final_masks,
        std::size_t mask_values) {
        if (mask_values > std::numeric_limits<std::size_t>::max() / kTrackerStepBatch2Size)
            throw std::overflow_error("SAM3 batch2 memory input size overflow");
        memory_batch2_input_scratch_.resize(kTrackerStepBatch2Size * mask_values);
        for (std::size_t batch = 0; batch < kTrackerStepBatch2Size; ++batch) {
            std::copy(final_masks[batch]->begin(), final_masks[batch]->end(),
                      memory_batch2_input_scratch_.data() + batch * mask_values);
        }
        return memory_batch2_input_scratch_.data();
    }

    static std::size_t memory_mask_values(int32_t height, int32_t width) {
        if (height <= 0 || width <= 0)
            throw std::invalid_argument("SAM3 memory encoder received invalid mask geometry");
        const auto h = static_cast<std::size_t>(height);
        const auto w = static_cast<std::size_t>(width);
        if (h > std::numeric_limits<std::size_t>::max() / w)
            throw std::overflow_error("SAM3 memory encoder mask geometry overflow");
        return h * w;
    }

    EncodedMemory encode_memory(const std::vector<float>& mask_values, float object_score_logit,
                                int32_t mask_height, int32_t mask_width, ITrtModule& memory_engine,
                                const char* mask_input_name, const char* producer,
                                const int32_t* suppress_area_shrinkage = nullptr) {
        if (mask_values.size() != memory_mask_values(mask_height, mask_width))
            throw std::runtime_error("SAM3 memory encoder received an unexpected mask size");
        Tensor mask;
        mask.data = const_cast<float*>(mask_values.data());
        mask.shape = {1, 1, mask_height, mask_width};
        mask.dtype = DType::kFloat32;
        Tensor score;
        score.data = &object_score_logit;
        score.shape = {1, 1};
        score.dtype = DType::kFloat32;
        TensorMap inputs{{mask_input_name, mask}, {"object_score_logits", score}};
        Tensor suppression;
        if (suppress_area_shrinkage != nullptr) {
            suppression.data = const_cast<int32_t*>(suppress_area_shrinkage);
            suppression.shape = {1, 1};
            suppression.dtype = DType::kInt32;
            inputs["suppress_area_shrinkage"] = suppression;
        }
        const auto memory_shape = memory_engine.tensor_shape("new_memory_features");
        const auto position_shape = memory_engine.tensor_shape("new_memory_position");
        const auto memory_values = static_shape_values(memory_shape);
        if (memory_values == 0 || static_shape_values(position_shape) != memory_values)
            throw std::runtime_error(std::string("SAM3 video ") + producer +
                                     " returned invalid device shapes");
        EncodedMemory encoded;
        const auto stream = memory_engine.stream();
        encoded.device = acquire_device_memory(memory_values, stream);
        ModuleSyncGuard memory_sync_guard(memory_engine);
        memory_engine.forward_async(inputs);
        sam3_round_bfloat16_copy(
            static_cast<const float*>(memory_engine.device_ptr("new_memory_features")),
            static_cast<float*>(encoded.device->features.data()), memory_values, stream);
        sam3_round_bfloat16_copy(
            static_cast<const float*>(memory_engine.device_ptr("new_memory_position")),
            static_cast<float*>(encoded.device->position.data()), memory_values, stream);
        check_cuda(cudaGetLastError(), "recurrent memory BF16 rounding");
        check_cuda(cudaStreamSynchronize(stream), "recurrent memory update");
        encoded.device->reusable = true;
        return encoded;
    }

    EncodedMemory encode_final_memory(const Sam3FrameFeatures& features,
                                      const std::vector<float>& final_mask,
                                      float object_score_logit, int32_t suppress_area_shrinkage) {
        return encode_memory(final_mask, object_score_logit, features.mask_height,
                             features.mask_width, tracker_memory_engine_, "final_mask",
                             "tracker memory engine", &suppress_area_shrinkage);
    }

    EncodedMemory encode_hard_memory(const std::vector<float>& owned_tracker_mask,
                                     float object_score_logit) {
        return encode_memory(owned_tracker_mask, object_score_logit, config_.image_size,
                             config_.image_size, tracker_hard_memory_engine_, "owned_tracker_mask",
                             "tracker hard memory engine");
    }

    TensorMap batch2_memory_inputs(
        const std::array<const std::vector<float>*, kTrackerStepBatch2Size>& final_masks,
        const std::array<float, kTrackerStepBatch2Size>& object_score_logits,
        std::size_t mask_values, int32_t mask_height, int32_t mask_width,
        const char* mask_input_name,
        const std::array<int32_t, kTrackerStepBatch2Size>* suppress_area_shrinkage = nullptr) {
        Tensor mask;
        mask.data = pack_final_memory_masks_batch2(final_masks, mask_values);
        mask.shape = {static_cast<int64_t>(kTrackerStepBatch2Size), 1, mask_height, mask_width};
        mask.dtype = DType::kFloat32;
        Tensor score;
        score.data = const_cast<float*>(object_score_logits.data());
        score.shape = {static_cast<int64_t>(kTrackerStepBatch2Size), 1};
        score.dtype = DType::kFloat32;
        TensorMap inputs{{mask_input_name, mask}, {"object_score_logits", score}};
        Tensor suppression;
        if (suppress_area_shrinkage != nullptr) {
            suppression.data = const_cast<int32_t*>(suppress_area_shrinkage->data());
            suppression.shape = {static_cast<int64_t>(kTrackerStepBatch2Size), 1};
            suppression.dtype = DType::kInt32;
            inputs["suppress_area_shrinkage"] = suppression;
        }
        return inputs;
    }

    static std::size_t validate_batch2_memory_device_shapes(const ITrtModule& engine) {
        const auto memory_shape = engine.tensor_shape("new_memory_features");
        const auto position_shape = engine.tensor_shape("new_memory_position");
        if (memory_shape.size() != 3 ||
            memory_shape.front() != static_cast<int64_t>(kTrackerStepBatch2Size) ||
            memory_shape[1] <= 0 || memory_shape.back() != kTrackerMemoryChannels ||
            position_shape != memory_shape) {
            throw std::runtime_error("SAM3 batch2 memory engine returned invalid device shapes");
        }
        const auto memory_values = static_shape_values(memory_shape);
        if (memory_values == 0 || memory_values % kTrackerStepBatch2Size != 0)
            throw std::runtime_error("SAM3 batch2 memory engine returned invalid size");
        return memory_values / kTrackerStepBatch2Size;
    }

    std::array<EncodedMemory, kTrackerStepBatch2Size>
    encode_batch2_device_memories(ITrtModule& memory_engine, const TensorMap& inputs,
                                  const char* producer) {
        const auto item_values = validate_batch2_memory_device_shapes(memory_engine);
        std::array<EncodedMemory, kTrackerStepBatch2Size> encoded;
        const auto stream = memory_engine.stream();
        for (auto& item : encoded)
            item.device = acquire_device_memory(item_values, stream);
        ModuleSyncGuard memory_sync_guard(memory_engine);
        memory_engine.forward_async(inputs);
        const auto* source_features =
            static_cast<const float*>(memory_engine.device_ptr("new_memory_features"));
        const auto* source_positions =
            static_cast<const float*>(memory_engine.device_ptr("new_memory_position"));
        for (std::size_t batch = 0; batch < kTrackerStepBatch2Size; ++batch) {
            sam3_round_bfloat16_copy(source_features + batch * item_values,
                                     static_cast<float*>(encoded[batch].device->features.data()),
                                     item_values, stream);
            sam3_round_bfloat16_copy(source_positions + batch * item_values,
                                     static_cast<float*>(encoded[batch].device->position.data()),
                                     item_values, stream);
            check_cuda(cudaGetLastError(), (std::string(producer) + " BF16 rounding").c_str());
        }
        check_cuda(cudaStreamSynchronize(stream), (std::string(producer) + " update").c_str());
        for (auto& item : encoded)
            item.device->reusable = true;
        return encoded;
    }

    std::array<EncodedMemory, kTrackerStepBatch2Size> encode_memories_batch2(
        const std::array<const std::vector<float>*, kTrackerStepBatch2Size>& final_masks,
        const std::array<float, kTrackerStepBatch2Size>& object_score_logits,
        ITrtModule& memory_engine, const char* mask_input_name, const char* producer,
        int32_t mask_height, int32_t mask_width,
        const std::array<int32_t, kTrackerStepBatch2Size>* suppress_area_shrinkage = nullptr) {
        const auto mask_values = final_masks.front()->size();
        if (mask_values != memory_mask_values(mask_height, mask_width) ||
            final_masks.back()->size() != mask_values) {
            throw std::runtime_error("SAM3 batch2 memory encoder received invalid masks");
        }
        const auto inputs =
            batch2_memory_inputs(final_masks, object_score_logits, mask_values, mask_height,
                                 mask_width, mask_input_name, suppress_area_shrinkage);

        // The pipeline-owned pinned input may be overwritten by the next
        // serialized fresh session only after this call returns. The device
        // path synchronizes the memory-engine stream below. The processing-
        // failure guard drains the same engine before exceptional return, so
        // no queued inference can outlive packed_masks.
        return encode_batch2_device_memories(memory_engine, inputs, producer);
    }

    std::array<EncodedMemory, kTrackerStepBatch2Size> encode_final_memories_batch2(
        const Sam3FrameFeatures& features,
        const std::array<const std::vector<float>*, kTrackerStepBatch2Size>& final_masks,
        const std::array<float, kTrackerStepBatch2Size>& object_score_logits,
        const std::array<int32_t, kTrackerStepBatch2Size>& suppress_area_shrinkage) {
        return encode_memories_batch2(final_masks, object_score_logits,
                                      tracker_memory_batch2_engine_, "final_mask",
                                      "batch2 tracker memory engine", features.mask_height,
                                      features.mask_width, &suppress_area_shrinkage);
    }

    std::array<EncodedMemory, kTrackerStepBatch2Size> encode_hard_memories_batch2(
        const std::array<const std::vector<float>*, kTrackerStepBatch2Size>& owned_masks,
        const std::array<float, kTrackerStepBatch2Size>& object_score_logits) {
        return encode_memories_batch2(owned_masks, object_score_logits,
                                      tracker_hard_memory_batch2_engine_, "owned_tracker_mask",
                                      "batch2 tracker hard memory engine", config_.image_size,
                                      config_.image_size);
    }

    Sam3FrameFeatures restore_frame_zero_tracker_feature_2() {
        constexpr const char* output_name = "sam3_tracker_feature_2";
        Sam3FrameFeatures features;
        features.mask_height = config_.low_res_mask_size;
        features.mask_width = config_.low_res_mask_size;
        const auto expected_shape = vision_encoder_.tensor_shape(output_name);
        auto* destination = vision_encoder_.device_ptr(output_name);
        if (!frame_zero_tracker_feature_2_.ok() || destination == nullptr ||
            frame_zero_tracker_feature_2_shape_ != expected_shape ||
            frame_zero_tracker_feature_2_.shape() != expected_shape ||
            vision_encoder_.tensor_dtype(output_name) != DType::kFloat32) {
            throw std::runtime_error("SAM3 frame-zero tracker feature retention is unavailable");
        }
        check_cuda(cudaMemcpyAsync(destination, frame_zero_tracker_feature_2_.data(),
                                   frame_zero_tracker_feature_2_.nbytes(), cudaMemcpyDeviceToDevice,
                                   vision_encoder_.stream()),
                   "frame-zero tracker feature restoration");
        check_cuda(cudaStreamSynchronize(vision_encoder_.stream()),
                   "frame-zero tracker feature restoration completion");
        return features;
    }

    static TrackerFrameRecord& frame_zero_conditioning_record(TrackState& track) {
        const auto record =
            std::find_if(track.records.begin(), track.records.end(),
                         [](const auto& item) { return item.conditioning && item.frame_idx == 0; });
        if (record == track.records.end()) {
            throw std::runtime_error("SAM3 frame-zero tracker state has no conditioning record");
        }
        return *record;
    }

    void release_frame_zero_soft_refresh_inputs() {
        frame_zero_tracker_feature_2_ = DeviceTensor{};
        frame_zero_tracker_feature_2_shape_.clear();
    }

    void release_frame_zero_soft_refresh_inputs_on_owner_noexcept() noexcept {
        if (!frame_zero_tracker_feature_2_.ok())
            return;
        int32_t previous_device = -1;
        const bool restore_device =
            cudaGetDevice(&previous_device) == cudaSuccess && previous_device != cuda_device_;
        if (previous_device != cuda_device_ && cudaSetDevice(cuda_device_) != cudaSuccess)
            return;
        release_frame_zero_soft_refresh_inputs();
        if (restore_device)
            (void)cudaSetDevice(previous_device);
    }

    Sam3VideoFrameResult
    prepare_frame_zero_propagation_result(const Sam3VideoFrameResult& prompt_result) {
        if (prompt_result.frame_idx != 0 || prompt_result.height <= 0 || prompt_result.width <= 0) {
            throw std::invalid_argument(
                "SAM3 frame-zero propagation received an invalid prompt result");
        }
        const auto mask_values =
            static_cast<std::size_t>(config_.low_res_mask_size) * config_.low_res_mask_size;
        std::map<int32_t, std::vector<float>> masks;
        for (auto& [object_id, track] : tracks_) {
            if (track.first_frame_idx != 0)
                continue;
            if (track.pending_frame_zero_soft_mask.size() != mask_values) {
                throw std::runtime_error(
                    "SAM3 frame-zero tracker state has no consolidated propagation mask");
            }
            if (!track.pending_frame_zero_object_score_logit.has_value()) {
                throw std::runtime_error(
                    "SAM3 frame-zero tracker state has no mask-input object score");
            }
            fill_small_components(track.pending_frame_zero_soft_mask, config_.low_res_mask_size,
                                  config_.low_res_mask_size, config_.fill_hole_area,
                                  mask_cleanup_workspace_);
            track.tracker_score = sigmoid(*track.pending_frame_zero_object_score_logit);
            masks.emplace(object_id, track.pending_frame_zero_soft_mask);
        }

        Sam3VideoFrame frame;
        frame.frame_idx = 0;
        frame.height = prompt_result.height;
        frame.width = prompt_result.width;
        return finish_result(make_deferred_result(frame, std::move(masks), {}));
    }

    struct FrameZeroSoftMemoryInputs {
        std::vector<int32_t> object_ids;
        std::map<int32_t, std::vector<float>> masks;
    };

    struct FrameZeroMemoryReplacement {
        int32_t object_id{-1};
        EncodedMemory memory;
    };

    FrameZeroSoftMemoryInputs collect_frame_zero_soft_memory_inputs() {
        FrameZeroSoftMemoryInputs inputs;
        const auto mask_values =
            static_cast<std::size_t>(config_.low_res_mask_size) * config_.low_res_mask_size;
        for (auto& [object_id, track] : tracks_) {
            if (track.first_frame_idx != 0)
                continue;
            (void)frame_zero_conditioning_record(track);
            if (track.pending_frame_zero_soft_mask.size() != mask_values) {
                throw std::runtime_error(
                    "SAM3 frame-zero tracker state has no consolidated propagation mask");
            }
            inputs.object_ids.push_back(object_id);
            inputs.masks.emplace(object_id, track.pending_frame_zero_soft_mask);
        }
        return inputs;
    }

    bool complete_empty_frame_zero_soft_memory_refresh(const FrameZeroSoftMemoryInputs& inputs) {
        if (!inputs.object_ids.empty())
            return false;
        if (!tracks_.empty())
            throw std::runtime_error("SAM3 frame-zero tracker state cannot be refreshed");
        frame_zero_soft_memory_refreshed_ = true;
        release_frame_zero_soft_refresh_inputs();
        return true;
    }

    std::vector<FrameZeroMemoryReplacement>
    encode_frame_zero_soft_memories(const Sam3FrameFeatures& features,
                                    const FrameZeroSoftMemoryInputs& inputs,
                                    const std::vector<int32_t>& shrink_memory) {
        std::vector<FrameZeroMemoryReplacement> replacements;
        replacements.reserve(inputs.object_ids.size());
        std::size_t index = 0;
        for (; index + kTrackerStepBatch2Size <= inputs.object_ids.size();
             index += kTrackerStepBatch2Size) {
            std::array<const std::vector<float>*, kTrackerStepBatch2Size> batch_masks{};
            std::array<float, kTrackerStepBatch2Size> scores{};
            std::array<int32_t, kTrackerStepBatch2Size> suppressions{};
            for (std::size_t batch = 0; batch < kTrackerStepBatch2Size; ++batch) {
                const auto item = index + batch;
                batch_masks[batch] = &inputs.masks.at(inputs.object_ids[item]);
                suppressions[batch] = shrink_memory[item];
                scores[batch] = shrink_memory[item] != 0 ? -10.0F : 10.0F;
            }
            auto encoded =
                encode_final_memories_batch2(features, batch_masks, scores, suppressions);
            for (std::size_t batch = 0; batch < kTrackerStepBatch2Size; ++batch) {
                replacements.push_back(
                    {inputs.object_ids[index + batch], std::move(encoded[batch])});
            }
        }
        for (; index < inputs.object_ids.size(); ++index) {
            const int32_t suppression = shrink_memory[index];
            const float score = suppression != 0 ? -10.0F : 10.0F;
            replacements.push_back(
                {inputs.object_ids[index],
                 encode_final_memory(features, inputs.masks.at(inputs.object_ids[index]), score,
                                     suppression)});
        }
        return replacements;
    }

    void install_frame_zero_soft_memories(std::vector<FrameZeroMemoryReplacement>& replacements) {
        // Every encoder call completed before recurrent state changes. A failure therefore leaves
        // the original hard memories and retained inputs intact, allowing the session failure
        // guard to quarantine only unpublished device outputs.
        for (auto& replacement : replacements) {
            auto& track = tracks_.at(replacement.object_id);
            auto& record = frame_zero_conditioning_record(track);
            record.memory_features = std::move(replacement.memory.features);
            record.memory_position = std::move(replacement.memory.position);
            record.device_memory = std::move(replacement.memory.device);
            track.pending_frame_zero_soft_mask.clear();
            track.pending_frame_zero_object_score_logit.reset();
        }
    }

    void refresh_frame_zero_conditioning_memories() {
        if (frame_zero_soft_memory_refreshed_)
            return;
        if (processed_frames_ != 1) {
            throw std::logic_error(
                "SAM3 frame-zero memory refresh requires exactly one processed frame");
        }

        const auto inputs = collect_frame_zero_soft_memory_inputs();
        if (complete_empty_frame_zero_soft_memory_refresh(inputs))
            return;

        const std::map<int32_t, std::vector<float>> no_overrides;
        const auto shrink_memory =
            apply_memory_area_shrinkage_policy(config_.low_res_mask_size, config_.low_res_mask_size,
                                               inputs.object_ids, inputs.masks, no_overrides);
        if (shrink_memory.size() != inputs.object_ids.size()) {
            throw std::runtime_error("SAM3 frame-zero memory-area policy returned the wrong size");
        }
        auto features = restore_frame_zero_tracker_feature_2();
        auto replacements = encode_frame_zero_soft_memories(features, inputs, shrink_memory);
        install_frame_zero_soft_memories(replacements);
        frame_zero_soft_memory_refreshed_ = true;
        release_frame_zero_soft_refresh_inputs();
    }

    AssociationPlan associate(const std::vector<Detection>& detections,
                              const PropagatedTrackBatch& propagated) const {
        if (propagated.object_ids.empty())
            return all_detections_are_new(detections.size());
        if (detections.empty())
            return all_tracks_are_unmatched(propagated);

        AssociationPlan plan;
        const auto ious = association_iou_matrix(detections, propagated);
        classify_track_associations(plan, propagated, ious,
                                    config_.tracker_association_iou_threshold);
        for (std::size_t detection = 0; detection < detections.size(); ++detection)
            classify_detection_association(plan, detections, propagated.object_ids, ious, detection,
                                           config_);
        return plan;
    }

    void update_hotstart_liveness(int32_t frame_idx, const AssociationPlan& association) {
        std::unordered_set<int32_t> matched;
        for (const auto& [_, track_ids] : association.detection_to_track_ids)
            matched.insert(track_ids.begin(), track_ids.end());
        for (const int32_t object_id : matched) {
            auto& track = tracks_.at(object_id);
            track.keep_alive = std::min(config_.max_tracker_keep_alive, track.keep_alive + 1);
        }
        for (const int32_t object_id : association.unmatched_track_ids) {
            auto& track = tracks_.at(object_id);
            track.unmatched_frame_indices.push_back(frame_idx);
            track.keep_alive = std::max(config_.min_tracker_keep_alive, track.keep_alive - 1);
        }
        if (config_.decrease_keep_alive_for_empty_masks) {
            for (const int32_t object_id : association.empty_track_ids) {
                auto& track = tracks_.at(object_id);
                track.keep_alive = std::max(config_.min_tracker_keep_alive, track.keep_alive - 1);
            }
        }
    }

    void collect_unmatched_hotstart_removals(int32_t frame_idx, bool streaming,
                                             std::vector<int32_t>& removed) const {
        if (streaming)
            return;
        const int32_t boundary = frame_idx - config_.hotstart_delay;
        for (const auto& [object_id, track] : tracks_) {
            const bool past_threshold =
                track.unmatched_frame_indices.size() >=
                static_cast<std::size_t>(config_.hotstart_unmatch_threshold);
            if (past_threshold && track.first_frame_idx > boundary)
                removed.push_back(object_id);
        }
    }

    void record_duplicate_hotstart_overlaps(int32_t frame_idx, const AssociationPlan& association) {
        // Track many-to-one matches exactly as upstream hotstart duplicate
        // metadata does. New objects do not participate until the next frame.
        for (const auto& [_, matched_ids] : association.detection_to_track_ids) {
            if (matched_ids.size() < 2)
                continue;
            const int32_t earliest = *std::min_element(
                matched_ids.begin(), matched_ids.end(), [&](int32_t lhs, int32_t rhs) {
                    return tracks_.at(lhs).first_frame_idx < tracks_.at(rhs).first_frame_idx;
                });
            for (const int32_t object_id : matched_ids) {
                if (object_id != earliest)
                    overlap_frames_[{earliest, object_id}].push_back(frame_idx);
            }
        }
    }

    void collect_duplicate_hotstart_removals(int32_t frame_idx, bool streaming,
                                             std::vector<int32_t>& removed) const {
        if (streaming)
            return;
        const int32_t boundary = frame_idx - config_.hotstart_delay;
        for (const auto& [pair, frame_indices] : overlap_frames_) {
            const int32_t object_id = pair.second;
            const auto track = tracks_.find(object_id);
            if (track == tracks_.end())
                continue;
            const bool past_threshold =
                frame_indices.size() >=
                static_cast<std::size_t>(config_.hotstart_duplicate_threshold);
            if (track->second.first_frame_idx > boundary && past_threshold)
                removed.push_back(object_id);
        }
    }

    std::vector<int32_t> apply_hotstart(int32_t frame_idx, bool streaming,
                                        const AssociationPlan& association) {
        update_hotstart_liveness(frame_idx, association);
        std::vector<int32_t> newly_removed;
        collect_unmatched_hotstart_removals(frame_idx, streaming, newly_removed);
        record_duplicate_hotstart_overlaps(frame_idx, association);
        collect_duplicate_hotstart_removals(frame_idx, streaming, newly_removed);
        std::sort(newly_removed.begin(), newly_removed.end());
        newly_removed.erase(std::unique(newly_removed.begin(), newly_removed.end()),
                            newly_removed.end());
        return newly_removed;
    }

    void prune_records(TrackState& track, int32_t current_frame_idx) const {
        std::unordered_set<int32_t> keep_conditioning;
        for (auto iter = track.records.rbegin();
             iter != track.records.rend() &&
             keep_conditioning.size() < static_cast<std::size_t>(config_.max_conditioning_frames);
             ++iter) {
            if (iter->conditioning)
                keep_conditioning.insert(iter->frame_idx);
        }

        std::unordered_set<int32_t> keep_nonconditioning;
        for (auto iter = track.records.rbegin();
             iter != track.records.rend() &&
             keep_nonconditioning.size() <
                 static_cast<std::size_t>(config_.max_object_pointers - 1);
             ++iter) {
            if (!iter->conditioning && iter->effective_iou_score > kMemoryQualityThreshold) {
                keep_nonconditioning.insert(iter->frame_idx);
            }
        }
        // The newest record is the next frame's mandatory t-1 memory even if
        // its quality score is below the selection threshold.
        keep_nonconditioning.insert(current_frame_idx);

        track.records.erase(
            std::remove_if(track.records.begin(), track.records.end(),
                           [&](const auto& record) {
                               return record.conditioning
                                          ? keep_conditioning.count(record.frame_idx) == 0
                                          : keep_nonconditioning.count(record.frame_idx) == 0;
                           }),
            track.records.end());
    }

    void add_record(TrackState& track, int32_t frame_idx, bool conditioning,
                    TrackerNeuralOutput output,
                    std::optional<float> cohort_effective_iou_score = std::nullopt) const {
        TrackerFrameRecord record;
        record.frame_idx = frame_idx;
        record.conditioning = conditioning;
        record.individual_effective_iou_score = output.effective_iou_score;
        record.effective_iou_score =
            cohort_effective_iou_score.value_or(output.effective_iou_score);
        record.object_pointer = std::move(output.object_pointer);
        record.memory_features = std::move(output.memory_features);
        record.memory_position = std::move(output.memory_position);
        record.device_memory = std::move(output.device_memory);
        track.records.push_back(std::move(record));
        if (conditioning)
            ++track.conditioning_records_seen;
        prune_records(track, frame_idx);
    }

    std::map<int32_t, float>
    cohort_effective_iou_scores(const std::vector<int32_t>& object_ids,
                                const std::map<int32_t, TrackerNeuralOutput>& outputs) const {
        struct ScoreTotal {
            float sum{0.0F};
            std::size_t count{0};
        };
        std::map<int32_t, ScoreTotal> totals;
        for (const int32_t object_id : object_ids) {
            const auto& track = tracks_.at(object_id);
            const auto output = outputs.find(object_id);
            if (track.cohort_id < 0 || output == outputs.end()) {
                throw std::runtime_error(
                    "SAM3 tracker cohort quality received incomplete recurrent outputs");
            }
            auto& total = totals[track.cohort_id];
            total.sum += output->second.effective_iou_score;
            ++total.count;
        }

        // Every live row in a Meta tracker inference state participates in
        // cal_mem_score's mean. A partial cohort here would silently turn the
        // shared frame filter back into object-local selection.
        std::map<int32_t, std::size_t> expected_counts;
        for (const auto& [_, track] : tracks_) {
            ++expected_counts[track.cohort_id];
            const auto total = totals.find(track.cohort_id);
            if (total == totals.end() || total->second.count == 0)
                throw std::runtime_error("SAM3 tracker cohort is missing from recurrent outputs");
        }
        for (const auto& [cohort_id, expected] : expected_counts) {
            if (totals.at(cohort_id).count != expected) {
                throw std::runtime_error(
                    "SAM3 tracker cohort has a partial recurrent output batch");
            }
        }

        std::map<int32_t, float> means;
        for (const auto& [cohort_id, total] : totals)
            means.emplace(cohort_id, total.sum / static_cast<float>(total.count));
        return means;
    }

    void refresh_surviving_cohort_record_qualities(int32_t cohort_id) {
        struct ScoreTotal {
            float sum{0.0F};
            std::size_t count{0};
        };
        std::map<int32_t, ScoreTotal> totals;
        for (const auto& [_, track] : tracks_) {
            if (track.cohort_id != cohort_id)
                continue;
            for (const auto& record : track.records) {
                auto& total = totals[record.frame_idx];
                total.sum += record.individual_effective_iou_score;
                ++total.count;
            }
        }
        for (auto& [_, track] : tracks_) {
            if (track.cohort_id != cohort_id)
                continue;
            for (auto& record : track.records) {
                const auto& total = totals.at(record.frame_idx);
                record.effective_iou_score = total.sum / static_cast<float>(total.count);
            }
        }
    }

    int32_t object_to_suppress_for_overlap(int32_t lhs_id, int32_t rhs_id,
                                           const std::unordered_set<int32_t>& removed,
                                           const std::vector<std::uint64_t>& lhs_mask,
                                           const std::vector<std::uint64_t>& rhs_mask) const {
        if (packed_mask_iou(lhs_mask, rhs_mask) < config_.overlap_suppression_threshold) {
            return -1;
        }
        // Native hotstart temporarily marks newly removed tracks as
        // ALWAYS_OCCLUDED while overlap suppression and memory encoding still
        // operate on the complete pre-removal object set.
        const int32_t lhs_occluded =
            removed.count(lhs_id) != 0 ? 100000 : tracks_.at(lhs_id).last_occluded;
        const int32_t rhs_occluded =
            removed.count(rhs_id) != 0 ? 100000 : tracks_.at(rhs_id).last_occluded;
        if (lhs_occluded > rhs_occluded && rhs_occluded > -1)
            return lhs_id;
        if (rhs_occluded > lhs_occluded && lhs_occluded > -1)
            return rhs_id;
        return -1;
    }

    std::vector<int32_t> collect_recent_overlap_suppressions(
        const std::vector<int32_t>& object_ids, const std::unordered_set<int32_t>& removed,
        const std::vector<std::vector<std::uint64_t>>& packed_masks) const {
        std::vector<int32_t> suppressed;
        for (std::size_t lhs = 0; lhs < object_ids.size(); ++lhs) {
            for (std::size_t rhs = lhs + 1; rhs < object_ids.size(); ++rhs) {
                const int32_t object_id =
                    object_to_suppress_for_overlap(object_ids[lhs], object_ids[rhs], removed,
                                                   packed_masks[lhs], packed_masks[rhs]);
                if (object_id >= 0)
                    suppressed.push_back(object_id);
            }
        }
        std::sort(suppressed.begin(), suppressed.end());
        suppressed.erase(std::unique(suppressed.begin(), suppressed.end()), suppressed.end());
        return suppressed;
    }

    void apply_recent_overlap_suppressions(const std::vector<int32_t>& suppressed,
                                           std::map<int32_t, std::vector<float>>& masks) const {
        for (const int32_t object_id : suppressed) {
            auto& mask = masks.at(object_id);
            std::fill(mask.begin(), mask.end(), kNoObjectLogit);
        }
    }

    void update_recent_occlusion_metadata(int32_t frame_idx, const std::vector<int32_t>& object_ids,
                                          const std::vector<int32_t>& suppressed,
                                          const std::vector<bool>& foreground) {
        for (std::size_t index = 0; index < object_ids.size(); ++index) {
            const int32_t object_id = object_ids[index];
            if (!foreground[index] ||
                std::binary_search(suppressed.begin(), suppressed.end(), object_id)) {
                tracks_.at(object_id).last_occluded = frame_idx;
            }
        }
    }

    void suppress_recent_overlaps(int32_t frame_idx, const std::vector<int32_t>& object_ids,
                                  const std::vector<int32_t>& newly_removed,
                                  const std::vector<std::vector<std::uint64_t>>& packed_masks,
                                  const std::vector<bool>& foreground,
                                  std::map<int32_t, std::vector<float>>& current_masks) {
        const std::unordered_set<int32_t> removed(newly_removed.begin(), newly_removed.end());
        const auto suppressed =
            collect_recent_overlap_suppressions(object_ids, removed, packed_masks);
        apply_recent_overlap_suppressions(suppressed, current_masks);
        update_recent_occlusion_metadata(frame_idx, object_ids, suppressed, foreground);
    }

    const std::vector<float>&
    memory_source_mask(int32_t object_id,
                       const std::map<int32_t, std::vector<float>>& current_masks,
                       const std::map<int32_t, std::vector<float>>& memory_overrides) const {
        const auto override = memory_overrides.find(object_id);
        return override == memory_overrides.end() ? current_masks.at(object_id) : override->second;
    }

    struct MemoryAreaGeometry {
        std::size_t low_res_area{0};
        int32_t high_res_height{0};
        int32_t high_res_width{0};
        std::size_t high_res_area{0};
    };

    static MemoryAreaGeometry memory_area_geometry(int32_t low_res_height, int32_t low_res_width) {
        if (low_res_height <= 0 || low_res_width <= 0 ||
            low_res_height > std::numeric_limits<int32_t>::max() / kMemoryMaskInterpolationScale ||
            low_res_width > std::numeric_limits<int32_t>::max() / kMemoryMaskInterpolationScale) {
            throw std::runtime_error("SAM3 video memory-area policy received invalid geometry");
        }
        MemoryAreaGeometry geometry;
        geometry.low_res_area = static_cast<std::size_t>(low_res_height) * low_res_width;
        if (geometry.low_res_area / static_cast<std::size_t>(low_res_height) !=
            static_cast<std::size_t>(low_res_width)) {
            throw std::overflow_error("SAM3 video memory-area policy geometry overflow");
        }
        geometry.high_res_height = low_res_height * kMemoryMaskInterpolationScale;
        geometry.high_res_width = low_res_width * kMemoryMaskInterpolationScale;
        geometry.high_res_area =
            static_cast<std::size_t>(geometry.high_res_height) * geometry.high_res_width;
        if (geometry.high_res_area / static_cast<std::size_t>(geometry.high_res_height) !=
            static_cast<std::size_t>(geometry.high_res_width)) {
            throw std::overflow_error("SAM3 video memory-area policy geometry overflow");
        }
        return geometry;
    }

    static void update_memory_area_winners(std::size_t index, std::vector<float> high_res,
                                           std::vector<std::size_t>& area_before,
                                           std::vector<float>& winner_logits,
                                           std::vector<std::size_t>& winner_indices) {
        area_before[index] = static_cast<std::size_t>(std::count_if(
            high_res.begin(), high_res.end(), [](float value) { return value > 0.0F; }));
        if (index == 0) {
            winner_logits = std::move(high_res);
            return;
        }
        for (std::size_t pixel = 0; pixel < high_res.size(); ++pixel) {
            if (high_res[pixel] > winner_logits[pixel]) {
                winner_logits[pixel] = high_res[pixel];
                winner_indices[pixel] = index;
            }
        }
    }

    static std::vector<std::size_t>
    winning_memory_areas(const std::vector<float>& winner_logits,
                         const std::vector<std::size_t>& winner_indices, std::size_t object_count) {
        std::vector<std::size_t> area_after(object_count, 0);
        for (std::size_t pixel = 0; pixel < winner_logits.size(); ++pixel) {
            if (winner_logits[pixel] > 0.0F)
                ++area_after[winner_indices[pixel]];
        }
        return area_after;
    }

    static std::vector<int32_t>
    memory_area_suppressions(const std::vector<std::size_t>& area_before,
                             const std::vector<std::size_t>& area_after) {
        std::vector<int32_t> shrink(area_before.size(), 0);
        for (std::size_t index = 0; index < area_before.size(); ++index) {
            const float denominator =
                static_cast<float>(std::max<std::size_t>(area_before[index], 1));
            const float retained = static_cast<float>(area_after[index]) / denominator;
            shrink[index] = retained < kMemoryAreaRetentionThreshold ? 1 : 0;
        }
        return shrink;
    }

    std::vector<int32_t> apply_memory_area_shrinkage_policy(
        int32_t low_res_height, int32_t low_res_width, const std::vector<int32_t>& object_ids,
        const std::map<int32_t, std::vector<float>>& current_masks,
        const std::map<int32_t, std::vector<float>>& memory_overrides) const {
        if (object_ids.empty())
            return {};
        const auto geometry = memory_area_geometry(low_res_height, low_res_width);

        std::vector<std::size_t> area_before(object_ids.size(), 0);
        std::vector<float> winner_logits;
        std::vector<std::size_t> winner_indices(geometry.high_res_area, 0);
        for (std::size_t index = 0; index < object_ids.size(); ++index) {
            const auto& source =
                memory_source_mask(object_ids[index], current_masks, memory_overrides);
            if (source.size() != geometry.low_res_area) {
                throw std::runtime_error(
                    "SAM3 video memory-area policy received an unexpected mask size");
            }
            auto high_res =
                resize_float_mask_bilinear(source, low_res_height, low_res_width,
                                           geometry.high_res_height, geometry.high_res_width);
            update_memory_area_winners(index, std::move(high_res), area_before, winner_logits,
                                       winner_indices);
        }
        const auto area_after =
            winning_memory_areas(winner_logits, winner_indices, object_ids.size());
        return memory_area_suppressions(area_before, area_after);
    }

    DeferredFrameResult make_deferred_result(const Sam3VideoFrame& frame,
                                             std::map<int32_t, std::vector<float>> masks,
                                             const std::vector<int32_t>& suppressed) const {
        DeferredFrameResult deferred;
        deferred.frame_idx = frame.frame_idx;
        deferred.height = frame.height;
        deferred.width = frame.width;
        deferred.low_res_height = config_.low_res_mask_size;
        deferred.low_res_width = config_.low_res_mask_size;
        deferred.removed_object_ids.assign(removed_object_ids_.begin(), removed_object_ids_.end());
        std::sort(deferred.removed_object_ids.begin(), deferred.removed_object_ids.end());
        deferred.suppressed_object_ids = suppressed;
        const std::unordered_set<int32_t> hidden(suppressed.begin(), suppressed.end());
        deferred.objects.reserve(tracks_.size());
        for (const auto& [object_id, track] : tracks_) {
            if (hidden.count(object_id) != 0)
                continue;
            auto mask = masks.find(object_id);
            if (mask == masks.end())
                continue;
            DeferredResultObject object;
            object.object_id = object_id;
            object.detection_score = track.detection_score;
            object.tracker_score = track.tracker_score;
            object.low_res_mask = std::move(mask->second);
            deferred.objects.push_back(std::move(object));
        }
        return deferred;
    }

    static std::size_t result_mask_winner(const Sam3VideoFrameResult& result, std::size_t mask_area,
                                          std::size_t pixel) {
        std::size_t winner = result.object_ids.size();
        float winner_score = 0.0F;
        for (std::size_t object = 0; object < result.object_ids.size(); ++object) {
            if (result.masks[object * mask_area + pixel] <= 0.0F)
                continue;
            const float score = result.tracker_scores[object];
            if (winner == result.object_ids.size() || score > winner_score) {
                winner = object;
                winner_score = score;
            }
        }
        return winner;
    }

    static void keep_only_result_mask_winner(Sam3VideoFrameResult& result, std::size_t mask_area,
                                             std::size_t pixel, std::size_t winner) {
        for (std::size_t object = 0; object < result.object_ids.size(); ++object) {
            if (object != winner)
                result.masks[object * mask_area + pixel] = 0.0F;
        }
    }

    static void resolve_result_mask_overlaps(Sam3VideoFrameResult& result) {
        if (result.object_ids.size() <= 1)
            return;
        const std::size_t mask_area = static_cast<std::size_t>(result.height) * result.width;
        parallel_for_host_ranges(mask_area, [&](std::size_t begin, std::size_t end) {
            for (std::size_t pixel = begin; pixel < end; ++pixel) {
                const std::size_t winner = result_mask_winner(result, mask_area, pixel);
                if (winner != result.object_ids.size())
                    keep_only_result_mask_winner(result, mask_area, pixel, winner);
            }
        });
    }

    Sam3VideoFrameResult finish_result(DeferredFrameResult deferred) const {
        Sam3VideoFrameResult result;
        result.frame_idx = deferred.frame_idx;
        result.height = deferred.height;
        result.width = deferred.width;
        result.removed_object_ids = std::move(deferred.removed_object_ids);
        result.suppressed_object_ids = std::move(deferred.suppressed_object_ids);
        const auto mask_area = static_cast<std::size_t>(deferred.height) * deferred.width;
        result.object_ids.reserve(deferred.objects.size());
        result.masks.reserve(deferred.objects.size() * mask_area);
        result.detection_scores.reserve(deferred.objects.size());
        result.tracker_scores.reserve(deferred.objects.size());
        result.boxes.reserve(deferred.objects.size() * 4);
        if (deferred.objects.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::length_error("SAM3 result object count exceeds fused overlap indexing");
        }
        std::vector<std::uint32_t> pixel_owners;
        if (deferred.objects.size() > 1) {
            pixel_owners.assign(mask_area, std::numeric_limits<std::uint32_t>::max());
        }
        const auto resize_plan = make_mask_resize_plan(
            deferred.low_res_height, deferred.low_res_width, deferred.height, deferred.width);
        for (auto& object : deferred.objects) {
            const auto current_object = static_cast<std::uint32_t>(result.object_ids.size());
            const auto summary = append_resized_mask_to_frame(
                object.low_res_mask, resize_plan, result.masks,
                pixel_owners.empty() ? nullptr : &pixel_owners,
                pixel_owners.empty() ? nullptr : &result.tracker_scores, object.tracker_score,
                current_object);
            if (!summary.has_foreground)
                continue;
            result.object_ids.push_back(object.object_id);
            result.detection_scores.push_back(object.detection_score);
            result.tracker_scores.push_back(object.tracker_score);
            result.boxes.insert(result.boxes.end(), summary.box.begin(), summary.box.end());
        }

        // Native video postprocessing computes boxes first, then resolves
        // visible pixel ownership by tracker probability within each prompt
        // group. This session has one text prompt, so all visible objects share
        // a group. Strict comparison preserves torch.argmax's first-object tie
        // behavior (object IDs are accumulated in ascending order).
        if (pixel_owners.empty())
            resolve_result_mask_overlaps(result);
        // Meta drops masks that are empty before ownership, but retains rows
        // that lose their final pixel during non-overlap resolution. Their
        // mask becomes all-zero while scores and the pre-overlap box remain.
        return result;
    }

    void validate_tracker_mask_geometry(const Sam3FrameFeatures& features) const {
        if (features.mask_height != config_.low_res_mask_size ||
            features.mask_width != config_.low_res_mask_size) {
            throw std::runtime_error("SAM3 video detector mask size does not match tracker config");
        }
    }

    void postprocess_propagated_masks(PropagatedTrackBatch& batch,
                                      const Sam3FrameFeatures& features) {
        if (batch.object_ids.size() < 2) {
            for (const int32_t object_id : batch.object_ids) {
                auto& output = batch.outputs.at(object_id);
                fill_small_components(output.mask, features.mask_height, features.mask_width,
                                      config_.fill_hole_area, mask_cleanup_workspace_);
            }
            batch.packed_masks.reserve(batch.object_ids.size());
            batch.has_foreground.reserve(batch.object_ids.size());
            for (const int32_t object_id : batch.object_ids) {
                const auto& mask = batch.outputs.at(object_id).mask;
                auto packed = pack_binary_mask(mask.data(), mask.size());
                batch.has_foreground.push_back(packed_mask_has_foreground(packed));
                batch.packed_masks.push_back(std::move(packed));
            }
            return;
        }

        // Gather map elements in canonical ascending object-ID order before
        // workers start. No worker touches the map itself or mutates its shape.
        std::vector<TrackerNeuralOutput*> ordered_outputs;
        ordered_outputs.reserve(batch.object_ids.size());
        for (const int32_t object_id : batch.object_ids)
            ordered_outputs.push_back(&batch.outputs.at(object_id));

        auto& workspaces = vision_workspace_->propagated_mask_cleanup_workspaces;
        parallel_for_host_lanes(
            ordered_outputs.size(), [&](std::size_t lane, std::size_t begin, std::size_t end) {
                auto& workspace = workspaces[lane];
                for (std::size_t index = begin; index < end; ++index) {
                    fill_small_components(ordered_outputs[index]->mask, features.mask_height,
                                          features.mask_width, config_.fill_hole_area, workspace);
                }
            });

        // Preserve the original phase and allocation order: every cleanup
        // succeeds before either output vector is reserved or any mask is
        // packed. Fixed slots keep object-ID alignment independent of worker
        // completion order; vector<bool> remains a serial commit.
        batch.packed_masks.reserve(ordered_outputs.size());
        batch.has_foreground.reserve(ordered_outputs.size());
        batch.packed_masks.resize(ordered_outputs.size());
        parallel_for_host_lanes(
            ordered_outputs.size(), [&](std::size_t /*lane*/, std::size_t begin, std::size_t end) {
                for (std::size_t index = begin; index < end; ++index) {
                    const auto& mask = ordered_outputs[index]->mask;
                    batch.packed_masks[index] = pack_binary_mask(mask.data(), mask.size());
                }
            });
        for (const auto& packed : batch.packed_masks)
            batch.has_foreground.push_back(packed_mask_has_foreground(packed));
    }

    std::vector<TrackerStepGroup> build_tracker_step_groups(PropagatedTrackBatch& batch,
                                                            int32_t frame_idx) const {
        std::vector<TrackerStepGroup> groups;
        for (const auto& [object_id, track] : tracks_) {
            batch.object_ids.push_back(object_id);
            auto request = make_tracker_step_request(object_id, track, frame_idx);
            const auto group =
                std::find_if(groups.begin(), groups.end(), [&](const auto& candidate) {
                    return candidate.memory_count == request.memory_records.size() &&
                           candidate.pointer_count == request.pointer_records.size();
                });
            if (group == groups.end()) {
                TrackerStepGroup next;
                next.memory_count = request.memory_records.size();
                next.pointer_count = request.pointer_records.size();
                next.requests.push_back(std::move(request));
                groups.push_back(std::move(next));
            } else {
                group->requests.push_back(std::move(request));
            }
        }
        return groups;
    }

    std::size_t tracker_step_call_count(const std::vector<TrackerStepGroup>& groups) const {
        std::size_t calls = 0;
        for (const auto& group : groups) {
            // Use B2 for complete pairs and the fixed B1 engine for an odd
            // tail. Both engines share the same vision stream, so the final
            // event still covers every earlier tracker-step enqueue.
            calls += (group.requests.size() + kTrackerStepBatch2Size - 1) / kTrackerStepBatch2Size;
        }
        return calls;
    }

    static const EngineEnqueueCallback&
    tracker_step_enqueue_callback(std::size_t remaining_step_calls,
                                  const EngineEnqueueCallback& final_step_enqueued,
                                  const EngineEnqueueCallback& no_enqueue_callback) {
        return remaining_step_calls == 1 ? final_step_enqueued : no_enqueue_callback;
    }

    static void store_tracker_step_outputs(PropagatedTrackBatch& batch,
                                           const std::vector<TrackerStepRequest>& requests,
                                           std::size_t begin,
                                           std::vector<TrackerNeuralOutput> outputs) {
        for (std::size_t index = 0; index < kTrackerStepBatch2Size; ++index) {
            batch.outputs.emplace(requests[begin + index].object_id, std::move(outputs[index]));
        }
    }

    std::size_t execute_full_batch2_tracker_slices(
        PropagatedTrackBatch& batch, const Sam3FrameFeatures& features,
        const TrackerStepGroup& group, int32_t frame_idx, bool streaming, int32_t total_frames,
        std::size_t& remaining_step_calls, const EngineEnqueueCallback& final_step_enqueued,
        const EngineEnqueueCallback& no_enqueue_callback) {
        std::size_t begin = 0;
        for (; begin + kTrackerStepBatch2Size <= group.requests.size();
             begin += kTrackerStepBatch2Size) {
            const auto& enqueue_callback = tracker_step_enqueue_callback(
                remaining_step_calls, final_step_enqueued, no_enqueue_callback);
            auto outputs = propagate_track_requests(
                features, group.requests, begin, kTrackerStepBatch2Size, frame_idx, streaming,
                total_frames, tracker_step_batch2_engine_, true, enqueue_callback);
            --remaining_step_calls;
            store_tracker_step_outputs(batch, group.requests, begin, std::move(outputs));
        }
        return begin;
    }

    void execute_singleton_tracker_slices(PropagatedTrackBatch& batch,
                                          const Sam3FrameFeatures& features,
                                          const TrackerStepGroup& group, std::size_t begin,
                                          int32_t frame_idx, bool streaming, int32_t total_frames,
                                          std::size_t& remaining_step_calls,
                                          const EngineEnqueueCallback& final_step_enqueued,
                                          const EngineEnqueueCallback& no_enqueue_callback) {
        for (; begin < group.requests.size(); ++begin) {
            const auto& enqueue_callback = tracker_step_enqueue_callback(
                remaining_step_calls, final_step_enqueued, no_enqueue_callback);
            auto outputs = propagate_track_requests(features, group.requests, begin, 1, frame_idx,
                                                    streaming, total_frames, tracker_step_engine_,
                                                    false, enqueue_callback);
            --remaining_step_calls;
            batch.outputs.emplace(group.requests[begin].object_id, std::move(outputs.front()));
        }
    }

    void execute_tracker_step_group(PropagatedTrackBatch& batch, const Sam3FrameFeatures& features,
                                    const TrackerStepGroup& group, int32_t frame_idx,
                                    bool streaming, int32_t total_frames,
                                    std::size_t& remaining_step_calls,
                                    const EngineEnqueueCallback& final_step_enqueued,
                                    const EngineEnqueueCallback& no_enqueue_callback) {
        const auto begin = execute_full_batch2_tracker_slices(
            batch, features, group, frame_idx, streaming, total_frames, remaining_step_calls,
            final_step_enqueued, no_enqueue_callback);
        execute_singleton_tracker_slices(batch, features, group, begin, frame_idx, streaming,
                                         total_frames, remaining_step_calls, final_step_enqueued,
                                         no_enqueue_callback);
    }

    PropagatedTrackBatch
    propagate_existing_tracks(const Sam3FrameFeatures& features, int32_t frame_idx, bool streaming,
                              int32_t total_frames, std::shared_ptr<std::promise<void>> neural_done,
                              const EngineEnqueueCallback& final_step_enqueued = {}) {
        PropagatedTrackBatch batch;
        batch.object_ids.reserve(tracks_.size());
        const auto groups = build_tracker_step_groups(batch, frame_idx);
        std::size_t remaining_step_calls = tracker_step_call_count(groups);
        if (remaining_step_calls == 0 && final_step_enqueued)
            final_step_enqueued(false, nullptr);
        const EngineEnqueueCallback no_enqueue_callback;
        for (const auto& group : groups) {
            execute_tracker_step_group(batch, features, group, frame_idx, streaming, total_frames,
                                       remaining_step_calls, final_step_enqueued,
                                       no_enqueue_callback);
        }
        if (neural_done != nullptr)
            neural_done->set_value();
        postprocess_propagated_masks(batch, features);
        return batch;
    }

    void limit_new_detections(AssociationPlan& association,
                              const std::vector<Detection>& detections) const {
        if (tracks_.size() + association.new_detection_indices.size() <=
            static_cast<std::size_t>(config_.max_tracked_objects)) {
            return;
        }
        const auto available =
            static_cast<std::size_t>(config_.max_tracked_objects) -
            std::min(tracks_.size(), static_cast<std::size_t>(config_.max_tracked_objects));
        std::stable_sort(association.new_detection_indices.begin(),
                         association.new_detection_indices.end(), [&](int32_t lhs, int32_t rhs) {
                             return detections[static_cast<std::size_t>(lhs)].score >
                                    detections[static_cast<std::size_t>(rhs)].score;
                         });
        association.new_detection_indices.resize(available);
    }

    std::map<int32_t, std::vector<float>> collect_tracker_masks(PropagatedTrackBatch& propagated) {
        std::map<int32_t, std::vector<float>> masks;
        for (const int32_t object_id : propagated.object_ids) {
            tracks_.at(object_id).tracker_score =
                sigmoid(propagated.outputs.at(object_id).object_score_logit);
            masks[object_id] = std::move(propagated.outputs.at(object_id).mask);
        }
        return masks;
    }

    std::size_t tracker_state_size(const TrackState& track) const {
        return static_cast<std::size_t>(
            std::count_if(tracks_.begin(), tracks_.end(), [&](const auto& candidate) {
                return candidate.second.cohort_id == track.cohort_id;
            }));
    }

    void clear_singleton_nonconditioning_memory_around(int32_t object_id, int32_t frame_idx) {
        auto& track = tracks_.at(object_id);
        // Meta groups objects introduced on the same frame in one inference
        // state. Its default correction policy clears surrounding non-cond
        // memories only when that state currently contains one object.
        if (tracker_state_size(track) > 1)
            return;
        const int32_t begin = frame_idx - config_.num_mask_memory_frames;
        const int32_t end = frame_idx + config_.num_mask_memory_frames;
        track.records.erase(std::remove_if(track.records.begin(), track.records.end(),
                                           [&](const auto& record) {
                                               return !record.conditioning &&
                                                      record.frame_idx >= begin &&
                                                      record.frame_idx <= end;
                                           }),
                            track.records.end());
    }

    std::unordered_set<int32_t> apply_periodic_reconditioning(
        const Sam3FrameFeatures& features, int32_t frame_idx,
        const std::vector<int32_t>& object_ids, const std::vector<Detection>& detections,
        const AssociationPlan& association, std::map<int32_t, TrackerNeuralOutput>& propagated,
        std::map<int32_t, std::vector<float>>& current_masks,
        std::map<int32_t, std::vector<float>>& memory_overrides) {
        std::unordered_set<int32_t> reconditioned;
        const bool periodic = frame_idx % config_.recondition_every_nth_frame == 0;
        for (const int32_t object_id : object_ids) {
            const auto match = association.track_to_recondition_detection.find(object_id);
            if (!periodic || match == association.track_to_recondition_detection.end() ||
                propagated.at(object_id).object_score_logit <= config_.high_confidence_threshold) {
                continue;
            }
            const auto& detection = detections[static_cast<std::size_t>(match->second)];
            // Meta's add_new_mask path reruns the mask-input decoder on the
            // current frame. The visible mask and object pointer come from the
            // matched detection, while the later global memory update still
            // encodes the original propagated tracker mask.
            auto refreshed = initialize_track(features, detection.mask);
            auto& output = propagated.at(object_id);
            output.object_pointer = std::move(refreshed.object_pointer);
            output.object_score_logit = refreshed.object_score_logit;
            output.selected_iou = 1.0F;
            const float normalized_score = output.object_score_logit > 0.0F
                                               ? sigmoid(output.object_score_logit) * 2.0F - 1.0F
                                               : 0.0F;
            output.effective_iou_score = normalized_score;
            memory_overrides[object_id] = current_masks.at(object_id);
            current_masks[object_id] = detection.mask;
            clear_singleton_nonconditioning_memory_around(object_id, frame_idx);
            reconditioned.insert(object_id);
        }
        return reconditioned;
    }

    std::unordered_set<int32_t>
    reconditioned_cohort_ids(const std::unordered_set<int32_t>& reconditioned) const {
        std::unordered_set<int32_t> cohorts;
        for (const int32_t object_id : reconditioned)
            cohorts.insert(tracks_.at(object_id).cohort_id);
        return cohorts;
    }

    void
    update_existing_track_memories(const Sam3FrameFeatures& features, int32_t frame_idx,
                                   const std::vector<int32_t>& object_ids,
                                   std::map<int32_t, TrackerNeuralOutput>& propagated,
                                   const std::map<int32_t, std::vector<float>>& current_masks,
                                   const std::map<int32_t, std::vector<float>>& memory_overrides,
                                   const std::vector<int32_t>& shrink_memory,
                                   const std::unordered_set<int32_t>& reconditioned) {
        if (shrink_memory.size() != object_ids.size())
            throw std::runtime_error("SAM3 video memory-area policy returned the wrong size");
        const auto cohort_scores = cohort_effective_iou_scores(object_ids, propagated);
        const auto conditioning_cohorts = reconditioned_cohort_ids(reconditioned);
        const auto add_memory_record = [&](std::size_t index, EncodedMemory memory) {
            const int32_t object_id = object_ids[index];
            const auto& track = tracks_.at(object_id);
            auto output = std::move(propagated.at(object_id));
            output.memory_features = std::move(memory.features);
            output.memory_position = std::move(memory.position);
            output.device_memory = std::move(memory.device);
            add_record(tracks_.at(object_id), frame_idx,
                       conditioning_cohorts.count(track.cohort_id) != 0, std::move(output),
                       cohort_scores.at(track.cohort_id));
        };

        std::size_t index = 0;
        for (; index + kTrackerStepBatch2Size <= object_ids.size();
             index += kTrackerStepBatch2Size) {
            std::array<const std::vector<float>*, kTrackerStepBatch2Size> final_masks{};
            std::array<float, kTrackerStepBatch2Size> object_scores{};
            std::array<int32_t, kTrackerStepBatch2Size> suppress_area_shrinkage{};
            for (std::size_t batch = 0; batch < kTrackerStepBatch2Size; ++batch) {
                const auto item = index + batch;
                final_masks[batch] =
                    &memory_source_mask(object_ids[item], current_masks, memory_overrides);
                suppress_area_shrinkage[batch] = shrink_memory[item];
                object_scores[batch] = shrink_memory[item] != 0 ? -10.0F : 10.0F;
            }
            auto memories = encode_final_memories_batch2(features, final_masks, object_scores,
                                                         suppress_area_shrinkage);
            for (std::size_t batch = 0; batch < kTrackerStepBatch2Size; ++batch)
                add_memory_record(index + batch, std::move(memories[batch]));
        }
        for (; index < object_ids.size(); ++index) {
            const auto& final_mask =
                memory_source_mask(object_ids[index], current_masks, memory_overrides);
            const float object_score = shrink_memory[index] != 0 ? -10.0F : 10.0F;
            add_memory_record(index, encode_final_memory(features, final_mask, object_score,
                                                         shrink_memory[index]));
        }
    }

    void remove_hotstart_tracks(const std::vector<int32_t>& removed) {
        for (const int32_t object_id : removed) {
            removed_object_ids_.insert(object_id);
            const auto track = tracks_.find(object_id);
            if (track == tracks_.end())
                continue;
            const int32_t cohort_id = track->second.cohort_id;
            tracks_.erase(track);
            // Meta's remove_object slices every stored frame in this tracker
            // state and recalculates cal_mem_score over the surviving rows.
            refresh_surviving_cohort_record_qualities(cohort_id);
        }
    }

    int32_t
    commit_prepared_new_detection_track(const Detection& detection, int32_t frame_idx,
                                        int32_t cohort_id, TrackerNeuralOutput initialized,
                                        std::vector<float> output_mask,
                                        std::vector<float> conditioning_mask,
                                        std::map<int32_t, std::vector<float>>& current_masks) {
        TrackState track;
        track.object_id = next_object_id_++;
        track.cohort_id = cohort_id;
        track.first_frame_idx = frame_idx;
        track.detection_score = detection.score;
        track.tracker_score = detection.score;
        track.keep_alive = config_.initial_tracker_keep_alive;
        track.records.reserve(static_cast<std::size_t>(config_.max_pointer_inputs));
        if (frame_idx == 0) {
            track.pending_frame_zero_soft_mask = std::move(conditioning_mask);
            track.pending_frame_zero_object_score_logit = initialized.object_score_logit;
        }
        current_masks[track.object_id] = std::move(output_mask);
        add_record(track, frame_idx, true, std::move(initialized));
        const int32_t object_id = track.object_id;
        tracks_.emplace(track.object_id, std::move(track));
        return object_id;
    }

    void launch_parallel_tracker_init(std::array<std::future<TrackerNeuralOutput>, 2>& futures,
                                      const Sam3FrameFeatures& features,
                                      const std::array<const Detection*, 2>& detections) {
        const std::array<ITrtModule*, 2> engines{&tracker_init_engine_,
                                                 &parallel_tracker_init_engine_};
        for (std::size_t lane = 0; lane < futures.size(); ++lane) {
            futures[lane] =
                std::async(std::launch::async, [this, &features, detection = detections[lane],
                                                engine = engines[lane]] {
                    check_cuda(cudaSetDevice(cuda_device_),
                               "parallel tracker-init worker CUDA device selection");
                    return initialize_track_with_engine(features, detection->mask, *engine);
                });
        }
    }

    static std::array<TrackerNeuralOutput, 2>
    collect_parallel_tracker_init_outputs(std::array<std::future<TrackerNeuralOutput>, 2>& futures,
                                          std::exception_ptr& failure) {
        std::array<TrackerNeuralOutput, 2> initialized;
        for (std::size_t lane = 0; lane < futures.size(); ++lane) {
            try {
                initialized[lane] = futures[lane].get();
            } catch (...) {
                if (failure == nullptr)
                    failure = std::current_exception();
            }
        }
        return initialized;
    }

    static void rethrow_parallel_tracker_init_failure(const std::exception_ptr& init_failure) {
        // Drain every init lane and retain lane-order exception precedence.
        if (init_failure != nullptr)
            std::rethrow_exception(init_failure);
    }

    std::array<TrackerNeuralOutput, 2>
    initialize_two_tracks_in_parallel(const Sam3FrameFeatures& features,
                                      const std::array<const Detection*, 2>& detections) {
        if (!parallel_tracker_init_enabled_)
            throw std::logic_error("SAM3 parallel tracker-init pair is unavailable");

        std::array<std::future<TrackerNeuralOutput>, 2> futures;
        FutureArrayJoinGuard join_guard(futures);
        launch_parallel_tracker_init(futures, features, detections);
        std::exception_ptr failure;
        auto initialized = collect_parallel_tracker_init_outputs(futures, failure);
        rethrow_parallel_tracker_init_failure(failure);
        return initialized;
    }

    std::vector<PreparedNewTrackMasks> prepare_new_track_masks(
        const Sam3FrameFeatures& features, const std::vector<Detection>& detections,
        const AssociationPlan& association, int32_t video_height, int32_t video_width) {
        std::vector<const Detection*> selected;
        selected.reserve(association.new_detection_indices.size());
        for (const int32_t detection_index : association.new_detection_indices) {
            selected.push_back(&detections[static_cast<std::size_t>(detection_index)]);
        }
        auto tracker_masks = prepare_initial_tracker_masks(
            selected, features.mask_height, features.mask_width, video_height, video_width);
        std::vector<PreparedNewTrackMasks> prepared(selected.size());
        auto& workspaces = vision_workspace_->propagated_mask_cleanup_workspaces;
        parallel_for_host_lanes(
            prepared.size(), [&](std::size_t lane, std::size_t begin, std::size_t end) {
                auto& workspace = workspaces[lane];
                for (std::size_t index = begin; index < end; ++index) {
                    prepared[index].detection = selected[index];
                    prepared[index].prompt_mask = selected[index]->mask;
                    fill_small_components(prepared[index].prompt_mask, features.mask_height,
                                          features.mask_width, config_.fill_hole_area, workspace);
                    prepared[index].tracker_mask = std::move(tracker_masks[index]);
                }
            });
        return prepared;
    }

    std::vector<TrackerNeuralOutput>
    initialize_new_track_pointers(const Sam3FrameFeatures& features,
                                  const std::vector<PreparedNewTrackMasks>& prepared) {
        std::vector<TrackerNeuralOutput> initialized(prepared.size());
        if (parallel_tracker_init_enabled_ && prepared.size() == 2) {
            std::array<const Detection*, 2> selected{};
            for (std::size_t index = 0; index < selected.size(); ++index) {
                selected[index] = prepared[index].detection;
            }
            auto pair = initialize_two_tracks_in_parallel(features, selected);
            for (std::size_t index = 0; index < selected.size(); ++index)
                initialized[index] = std::move(pair[index]);
            return initialized;
        }
        for (std::size_t index = 0; index < prepared.size(); ++index) {
            initialized[index] = initialize_track(features, prepared[index].detection->mask);
        }
        return initialized;
    }

    static void attach_encoded_memory(TrackerNeuralOutput& output, EncodedMemory memory) {
        output.memory_features = std::move(memory.features);
        output.memory_position = std::move(memory.position);
        output.device_memory = std::move(memory.device);
    }

    static std::vector<std::vector<float>> copy_resized_hard_mask_output(const TensorMap& outputs,
                                                                         std::size_t batch_size,
                                                                         int32_t tracker_image_size,
                                                                         const char* producer) {
        const auto output = outputs.find("resized_tracker_mask");
        if (tracker_image_size <= 0)
            throw std::runtime_error(std::string(producer) + " has invalid geometry");
        const auto image_size = static_cast<std::size_t>(tracker_image_size);
        if (image_size > std::numeric_limits<std::size_t>::max() / image_size)
            throw std::overflow_error(std::string(producer) + " size overflow");
        const auto expected_area = image_size * image_size;
        if (batch_size > std::numeric_limits<std::size_t>::max() / expected_area)
            throw std::overflow_error(std::string(producer) + " batch size overflow");
        const auto expected_values = batch_size * expected_area;
        if (output == outputs.end() || output->second.dtype != DType::kFloat32 ||
            output->second.data == nullptr ||
            output->second.shape != std::vector<int64_t>{static_cast<int64_t>(batch_size), 1,
                                                         tracker_image_size, tracker_image_size} ||
            output->second.numel() != expected_values) {
            throw std::runtime_error(std::string(producer) + " returned an invalid output");
        }
        const auto* values = static_cast<const float*>(output->second.data);
        std::vector<std::vector<float>> result(batch_size);
        for (std::size_t batch = 0; batch < batch_size; ++batch) {
            result[batch].assign(values + batch * expected_area,
                                 values + (batch + 1) * expected_area);
        }
        return result;
    }

    void validate_hard_mask_resize_geometry(int32_t source_height, int32_t source_width) const {
        if (source_height <= 0 || source_width <= 0 || config_.image_size <= 0)
            throw std::runtime_error("SAM3 hard-mask resize received invalid geometry");
    }

    std::vector<std::vector<float>>
    resize_prepared_hard_mask_pair(const std::vector<PreparedNewTrackMasks>& prepared,
                                   std::size_t index, int32_t source_height, int32_t source_width,
                                   std::size_t source_area) {
        hard_mask_resize_batch2_input_scratch_.resize(kTrackerStepBatch2Size * source_area);
        for (std::size_t batch = 0; batch < kTrackerStepBatch2Size; ++batch) {
            const auto& source = prepared[index + batch].tracker_mask;
            if (source.size() != source_area)
                throw std::runtime_error("SAM3 hard-mask resize received an invalid B2 row");
            std::copy(source.begin(), source.end(),
                      hard_mask_resize_batch2_input_scratch_.begin() +
                          static_cast<std::ptrdiff_t>(batch * source_area));
        }
        Tensor input;
        input.data = hard_mask_resize_batch2_input_scratch_.data();
        input.shape = {2, 1, source_height, source_width};
        input.dtype = DType::kFloat32;
        return copy_resized_hard_mask_output(
            hard_mask_resize_batch2_engine_.forward({{"tracker_mask", input}}), 2,
            config_.image_size, "SAM3 hard-mask resize B2 engine");
    }

    std::vector<float> resize_prepared_hard_mask(const PreparedNewTrackMasks& prepared,
                                                 int32_t source_height, int32_t source_width,
                                                 std::size_t source_area) {
        const auto& source = prepared.tracker_mask;
        if (source.size() != source_area)
            throw std::runtime_error("SAM3 hard-mask resize received an invalid B1 row");
        Tensor input;
        input.data = const_cast<float*>(source.data());
        input.shape = {1, 1, source_height, source_width};
        input.dtype = DType::kFloat32;
        auto batch = copy_resized_hard_mask_output(
            hard_mask_resize_engine_.forward({{"tracker_mask", input}}), 1, config_.image_size,
            "SAM3 hard-mask resize B1 engine");
        return std::move(batch.front());
    }

    GlobalHardMaskOwnership
    select_prepared_hard_mask_owners_exact(const std::vector<PreparedNewTrackMasks>& prepared,
                                           int32_t source_height, int32_t source_width) {
        validate_hard_mask_resize_geometry(source_height, source_width);
        const auto source_area =
            static_cast<std::size_t>(source_height) * static_cast<std::size_t>(source_width);
        auto ownership = make_global_hard_mask_ownership(prepared.size(), config_.image_size);
        std::size_t index = 0;
        for (; index + kTrackerStepBatch2Size <= prepared.size(); index += kTrackerStepBatch2Size) {
            auto batch = resize_prepared_hard_mask_pair(prepared, index, source_height,
                                                        source_width, source_area);
            for (std::size_t row = 0; row < kTrackerStepBatch2Size; ++row)
                update_global_hard_mask_owners(index + row, batch[row], ownership);
        }
        for (; index < prepared.size(); ++index) {
            auto resized = resize_prepared_hard_mask(prepared[index], source_height, source_width,
                                                     source_area);
            update_global_hard_mask_owners(index, resized, ownership);
        }
        return ownership;
    }

    void encode_prepared_new_track_hard_memories(const Sam3FrameFeatures& features,
                                                 const std::vector<PreparedNewTrackMasks>& prepared,
                                                 std::vector<TrackerNeuralOutput>& initialized) {
        if (initialized.size() != prepared.size())
            throw std::logic_error("SAM3 hard-memory initialization count mismatch");
        if (prepared.empty())
            return;
        // Meta consolidates every object in the tracker state, performs one
        // object-axis argmax at the 1008 tracker grid, and only then slices
        // rows for memory encoding. Build that ownership map once so the B1
        // tail competes with all preceding B2 rows.
        const auto ownership = select_prepared_hard_mask_owners_exact(
            prepared, features.mask_height, features.mask_width);
        std::size_t index = 0;
        for (; index + kTrackerStepBatch2Size <= prepared.size(); index += kTrackerStepBatch2Size) {
            std::array<std::vector<float>, kTrackerStepBatch2Size> owned_masks{};
            std::array<const std::vector<float>*, kTrackerStepBatch2Size> masks{};
            std::array<float, kTrackerStepBatch2Size> scores{};
            scores.fill(10.0F);
            for (std::size_t batch = 0; batch < kTrackerStepBatch2Size; ++batch) {
                owned_masks[batch] = materialize_owned_hard_mask(ownership, index + batch);
                masks[batch] = &owned_masks[batch];
            }
            auto memories = encode_hard_memories_batch2(masks, scores);
            for (std::size_t batch = 0; batch < kTrackerStepBatch2Size; ++batch)
                attach_encoded_memory(initialized[index + batch], std::move(memories[batch]));
        }
        for (; index < prepared.size(); ++index) {
            auto owned_mask = materialize_owned_hard_mask(ownership, index);
            attach_encoded_memory(initialized[index], encode_hard_memory(owned_mask, 10.0F));
        }
    }

    void add_new_detection_tracks(const Sam3FrameFeatures& features,
                                  const std::vector<Detection>& detections,
                                  const AssociationPlan& association, int32_t frame_idx,
                                  int32_t video_height, int32_t video_width,
                                  std::map<int32_t, std::vector<float>>& current_masks) {
        auto prepared =
            prepare_new_track_masks(features, detections, association, video_height, video_width);
        auto initialized = initialize_new_track_pointers(features, prepared);
        // Meta finishes the new object's conditioning memory before the prompt call returns.  The
        // prompt call exposes the cleaned detector mask; propagate_in_video later emits the
        // separately consolidated tracker mask and re-encodes it with soft-memory semantics.
        encode_prepared_new_track_hard_memories(features, prepared, initialized);

        if (prepared.empty())
            return;
        if (next_cohort_id_ == std::numeric_limits<int32_t>::max())
            throw std::overflow_error("SAM3 tracker cohort identifier overflow");
        const int32_t cohort_id = next_cohort_id_++;
        for (std::size_t index = 0; index < prepared.size(); ++index) {
            commit_prepared_new_detection_track(
                *prepared[index].detection, frame_idx, cohort_id, std::move(initialized[index]),
                std::move(prepared[index].prompt_mask), std::move(prepared[index].tracker_mask),
                current_masks);
        }
    }

    std::vector<int32_t> frame_suppressed_objects(bool streaming) const {
        std::vector<int32_t> suppressed;
        if (streaming || config_.suppress_unmatched_only_within_hotstart)
            return suppressed;
        for (const auto& [object_id, track] : tracks_) {
            if (track.keep_alive <= 0)
                suppressed.push_back(object_id);
        }
        return suppressed;
    }

    DeferredFrameResult advance_frame_with_features(const Sam3VideoFrame& frame,
                                                    Sam3FrameFeatures features, bool streaming,
                                                    int32_t total_frames) {
        std::future<PropagatedTrackBatch> propagated_future;
        FutureJoinGuard<PropagatedTrackBatch> propagated_join_guard(propagated_future);
        const bool propagate_concurrently = !tracks_.empty();
        if (propagate_concurrently) {
            const int32_t cuda_device = cuda_device_;
            propagated_future =
                std::async(std::launch::async, [this, &features, frame_idx = frame.frame_idx,
                                                streaming, total_frames, cuda_device] {
                    check_cuda(cudaSetDevice(cuda_device), "tracker worker CUDA device selection");
                    return propagate_existing_tracks(features, frame_idx, streaming, total_frames,
                                                     nullptr);
                });
        }
        run_frame_detector(features);
        validate_tracker_mask_geometry(features);
        const auto detections = select_detections(features, config_);
        auto propagated = propagate_concurrently ? propagated_future.get() : PropagatedTrackBatch{};

        auto association = associate(detections, propagated);
        limit_new_detections(association, detections);
        const auto newly_removed = apply_hotstart(frame.frame_idx, streaming, association);
        auto tracker_masks = collect_tracker_masks(propagated);
        suppress_recent_overlaps(frame.frame_idx, propagated.object_ids, newly_removed,
                                 propagated.packed_masks, propagated.has_foreground, tracker_masks);
        std::map<int32_t, std::vector<float>> current_masks = std::move(tracker_masks);
        std::map<int32_t, std::vector<float>> memory_overrides;
        const auto reconditioned = apply_periodic_reconditioning(
            features, frame.frame_idx, propagated.object_ids, detections, association,
            propagated.outputs, current_masks, memory_overrides);
        const auto shrink_memory = apply_memory_area_shrinkage_policy(
            features.mask_height, features.mask_width, propagated.object_ids, current_masks,
            memory_overrides);
        update_existing_track_memories(features, frame.frame_idx, propagated.object_ids,
                                       propagated.outputs, current_masks, memory_overrides,
                                       shrink_memory, reconditioned);
        // Native SAM3 updates overlap metadata and encodes this frame's
        // policy-final memory for every pre-existing object before removing
        // hotstart casualties from recurrent state.
        remove_hotstart_tracks(newly_removed);
        add_new_detection_tracks(features, detections, association, frame.frame_idx, frame.height,
                                 frame.width, current_masks);
        if (frame.frame_idx == 0 && !tracks_.empty())
            retain_frame_zero_tracker_feature_2();
        const auto suppressed = frame_suppressed_objects(streaming);
        ++processed_frames_;
        return make_deferred_result(frame, std::move(current_masks), suppressed);
    }

    DeferredFrameResult advance_frame(const Sam3VideoFrame& frame, bool streaming,
                                      int32_t total_frames,
                                      const std::vector<float>* preprocessed_pixels = nullptr,
                                      bool cuda_preprocessed_input_ready = false) {
        auto features =
            run_frame_backbone(frame, preprocessed_pixels, cuda_preprocessed_input_ready);
        return advance_frame_with_features(frame, std::move(features), streaming, total_frames);
    }

    TrackerHeadOutputStaging& tracker_step_output_staging() {
        return vision_workspace_->tracker_step_output_staging;
    }

    TrackerHeadOutputStaging& tracker_step_batch2_output_staging() {
        return vision_workspace_->tracker_step_batch2_output_staging;
    }

    Sam3VideoFrameResult process_frame(const Sam3VideoFrame& frame, bool streaming,
                                       int32_t total_frames,
                                       const std::vector<float>* preprocessed_pixels = nullptr,
                                       bool cuda_preprocessed_input_ready = false) {
        return finish_result(advance_frame(frame, streaming, total_frames, preprocessed_pixels,
                                           cuda_preprocessed_input_ready));
    }

    ITrtModule& vision_encoder_;
    ITrtModule& core_engine_;
    ITrtModule& tracker_init_engine_;
    ITrtModule& parallel_tracker_init_engine_;
    ITrtModule& tracker_step_engine_;
    ITrtModule& tracker_step_batch2_engine_;
    ITrtModule& tracker_memory_engine_;
    ITrtModule& tracker_memory_batch2_engine_;
    ITrtModule& tracker_hard_memory_engine_;
    ITrtModule& tracker_hard_memory_batch2_engine_;
    ITrtModule& hard_mask_resize_engine_;
    ITrtModule& hard_mask_resize_batch2_engine_;
    Sam3Config config_;
    Sam3VideoTextInput text_input_;
    std::map<int32_t, TrackState> tracks_;
    std::unordered_set<int32_t> removed_object_ids_;
    std::map<std::pair<int32_t, int32_t>, std::vector<int32_t>> overlap_frames_;
    int32_t next_object_id_{0};
    int32_t next_cohort_id_{0};
    std::size_t processed_frames_{0};
    int32_t cuda_device_{0};
    std::shared_ptr<Sam3VideoVisionWorkspace> vision_workspace_;
    bool parallel_tracker_init_enabled_{false};
    Sam3ImagePreprocessWorkspace preprocess_workspace_;
    MaskCleanupWorkspace mask_cleanup_workspace_;
    TrackerStepPositionTemplate tracker_step_position_template_;
    TrackerStepPositionTemplate tracker_step_batch2_position_template_;
    std::vector<int32_t> memory_offsets_scratch_;
    std::vector<float> pointers_scratch_;
    std::vector<int32_t> pointer_offsets_scratch_;
    std::vector<float> memory_batch2_input_scratch_;
    std::vector<float> hard_mask_resize_batch2_input_scratch_;
    // The vision plan's externally-bound feature outputs are shared by every session. Preserve
    // frame zero per session so another prompt cannot overwrite the feature needed by Meta's
    // propagation-time soft-memory transition.
    DeviceTensor frame_zero_tracker_feature_2_;
    std::vector<int64_t> frame_zero_tracker_feature_2_shape_;
    bool frame_zero_soft_memory_refreshed_{false};
    // The owning pool keeps these addresses stable. Tracking every allocation
    // touched by this state lets a whole-call failure quarantine successful
    // producers even if the exception occurs after their record was pruned or
    // before a local result was inserted into recurrent state.
    std::unordered_set<DeviceEncodedMemory*> acquired_device_memories_;
    std::mutex device_memory_pool_mutex_;
    std::mutex mutex_;
};

} // namespace

std::vector<float> preprocess_sam3_image(const float* hwc_pixels, int32_t height, int32_t width,
                                         const Sam3Config& config) {
    Sam3ImagePreprocessWorkspace workspace;
    preprocess_sam3_image_into(hwc_pixels, height, width, config, workspace);
    return std::move(workspace.normalized);
}

std::shared_ptr<Sam3VideoVisionWorkspace> make_sam3_video_vision_workspace(
    ITrtModule& vision_encoder, ITrtModule& core_engine, ITrtModule& tracker_init_engine,
    ITrtModule& tracker_step_engine, ITrtModule& tracker_memory_engine,
    ITrtModule& tracker_hard_memory_engine, ITrtModule& tracker_hard_memory_batch2_engine,
    ITrtModule& tracker_step_batch2_engine, ITrtModule& tracker_memory_batch2_engine,
    ITrtModule& parallel_tracker_init_engine) {
    return make_vision_workspace_impl(
        vision_encoder, core_engine, tracker_init_engine, tracker_step_engine,
        tracker_memory_engine, tracker_hard_memory_engine, tracker_hard_memory_batch2_engine,
        tracker_step_batch2_engine, tracker_memory_batch2_engine, parallel_tracker_init_engine);
}

Sam3VideoFrameProcessor make_sam3_video_frame_processor(
    ITrtModule& vision_encoder, ITrtModule& core_engine, ITrtModule& tracker_init_engine,
    ITrtModule& tracker_step_engine, ITrtModule& tracker_memory_engine, Sam3Config config,
    Sam3VideoTextInput text_input, std::shared_ptr<Sam3VideoVisionWorkspace> vision_workspace,
    ITrtModule& tracker_hard_memory_engine, ITrtModule& tracker_hard_memory_batch2_engine,
    ITrtModule& tracker_step_batch2_engine, ITrtModule& tracker_memory_batch2_engine,
    ITrtModule& parallel_tracker_init_engine, ITrtModule& hard_mask_resize_engine,
    ITrtModule& hard_mask_resize_batch2_engine) {
    auto state = std::make_shared<Sam3VideoProcessorState>(
        vision_encoder, core_engine, tracker_init_engine, tracker_step_engine,
        tracker_memory_engine, tracker_hard_memory_engine, tracker_hard_memory_batch2_engine,
        std::move(config), std::move(text_input), std::move(vision_workspace),
        tracker_step_batch2_engine, tracker_memory_batch2_engine, parallel_tracker_init_engine,
        hard_mask_resize_engine, hard_mask_resize_batch2_engine);
    Sam3VideoFrameProcessor processor;
    processor.accept_prompt = [state](const Sam3VideoFrame& frame) {
        return state->accept_prompt(frame);
    };
    processor.continue_borrowed = [state](Sam3VideoFrameResult prompt_result,
                                          const std::vector<Sam3VideoFrame>& remaining_frames,
                                          int32_t total_frames) {
        return state->continue_borrowed(std::move(prompt_result), remaining_frames, total_frames);
    };
    return processor;
}

} // namespace trtmc
