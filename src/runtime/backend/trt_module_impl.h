/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// TrtModuleImpl: concrete ITrtModule backed by a TRT engine.
// Compiled inside backend DSOs only (libtrtmc_backend_trt.so / _rtx.so).

#include "runtime/backend/prebound_backend.h"
#include "runtime/backend/runtime_memory_backend.h"
#include "runtime/backend/trt_logger.h"

#include <NvInfer.h>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace trtmc {

class CudaGraphExec;
class TrtModuleImplTestPeer;

class TrtModuleImpl : public ITrtModule {
  public:
    // Backend creates engine + context, passes them in.
    // The engine must outlive this module (caller manages lifetime via keep_alive).
    TrtModuleImpl(nvinfer1::ICudaEngine* engine, nvinfer1::IExecutionContext* ctx,
                  cudaStream_t stream, int32_t profile_idx = 0,
                  void* distributed_communicator = nullptr,
                  const std::vector<ModuleExternalBinding>& external_bindings = {},
                  bool runtime_managed_context = false,
                  std::vector<std::string> deferred_runtime_tensors = {},
                  std::vector<RuntimeMemoryAliasPairV1> runtime_alias_pairs = {});
    ~TrtModuleImpl() override;

    TrtModuleImpl(const TrtModuleImpl&) = delete;
    TrtModuleImpl& operator=(const TrtModuleImpl&) = delete;

    // ITrtModule interface
    TensorMap forward(const TensorMap& inputs) override;
    DeviceTensorMap forward_device(const DeviceTensorMap& inputs) override;
    void forward_device_async(const DeviceTensorMap& inputs) override;
    void forward_async(const TensorMap& inputs) override;
    void sync() override;
    cudaStream_t stream() const override { return stream_; }
    void enable_cuda_graph() override;
    bool cuda_graph_active() const override { return use_cuda_graph_; }
    int32_t profile_idx() const override { return profile_idx_; }
    std::vector<TensorInfo> input_info() const override;
    std::vector<TensorInfo> output_info() const override;
    bool has_input(const std::string& name) const override;
    bool has_output(const std::string& name) const override;
    DType tensor_dtype(const std::string& name) const override;
    std::vector<int64_t> tensor_shape(const std::string& name) const override;
    std::vector<int64_t> input_profile_shape(const std::string& name, int32_t profile_idx,
                                             ProfileShapeSelector selector) const override;
    int32_t optimization_profile_count() const override;
    void* device_ptr(const std::string& name) const override;
    void bind_external(const std::string& name, void* ptr) override;
    void bind_external(const std::string& name, void* ptr,
                       const std::vector<int64_t>& shape) override;
    int32_t input_rank(const std::string& name) const override;
    bool input_is_dynamic(const std::string& name) const override;
    void reset_execution_context() override;
    void set_timing_label(std::string label) override;
    bool ok() const override {
        return ctx_ != nullptr && !allocation_failed_ && !runtime_binding_poisoned_ &&
               !execution_failed_;
    }
    void keep_alive(std::shared_ptr<void> resource) override;

    // Runtime-memory implementation used only through
    // RuntimeMemoryTrtModuleImpl in the standard TensorRT backend.
    void set_runtime_binding_shape(const RuntimeMemoryShapeV1& shape);
    void set_runtime_alias_pair_shape(const RuntimeMemoryAliasShapeV1& shape);
    void set_runtime_input_shape(const RuntimeInputShapeV1& shape);
    void bind_runtime_memory(const RuntimeMemoryBindingV1& binding);
    void bind_runtime_memory_alias_pair(const RuntimeMemoryAliasBindingV1& binding);
    RuntimeMemoryContextRequirementV1 context_memory_requirement();
    void bind_context_memory(const RuntimeMemoryContextBlockV1& block);
    bool runtime_memory_ready() const noexcept;
    RuntimeMemoryEngineStatsV1 runtime_memory_engine_stats() const noexcept;
    RuntimeMemoryTransferSnapshotV1 runtime_memory_transfer_snapshot() const;
    TensorMap forward_selected(const TensorMap& inputs,
                               const std::vector<std::string>& host_output_names);

  private:
    friend class TrtModuleImplTestPeer;

    struct BufferEntry {
        void* d_ptr{nullptr};
        std::vector<int64_t> shape;
        DType dtype{DType::kFloat32};
        std::size_t nbytes{0};
        bool is_input{true};
        bool is_external{false};
        bool is_dynamic{false};
        bool is_runtime_deferred{false};
        bool runtime_descriptor_bound{false};
        bool runtime_shape_declared{false};
        bool runtime_input_shape_explicit{false};
        uint64_t valid_tokens{0};
        uint64_t bound_tokens{0};
        uint64_t capacity_tokens{0};
        int32_t sequence_axis{-1};
        uint64_t runtime_shape_generation{0};
        std::shared_ptr<void> lifetime;
    };
    struct TimingEvent {
        cudaEvent_t start{nullptr};
        cudaEvent_t stop{nullptr};
    };

    nvinfer1::ICudaEngine* engine_{nullptr};
    nvinfer1::IExecutionContext* ctx_{nullptr};
    cudaStream_t stream_{nullptr};
    int32_t profile_idx_{0};
    int32_t device_{-1};
    void* distributed_communicator_{nullptr};
    bool runtime_managed_context_{false};
    // TensorRT execution-context reconfiguration must not race an enqueue
    // submitted through this context. Once sync() has completed, later
    // runtime shape/address/context operations do not need to synchronize
    // unrelated stream-ordered work such as the KV commit copy.
    bool runtime_execution_in_flight_{false};
    uint64_t runtime_input_shape_generation_{1};
    uint64_t context_memory_generation_{0};
    bool context_memory_queried_{false};
    bool context_memory_bound_{false};
    std::size_t context_memory_requirement_bytes_{0};
    void* context_memory_pointer_{nullptr};
    std::size_t context_memory_capacity_bytes_{0};
    std::shared_ptr<void> context_memory_lifetime_;
    std::unordered_set<std::string> deferred_runtime_tensors_;
    std::vector<RuntimeMemoryAliasPairV1> runtime_alias_pairs_;
    std::unordered_map<std::string, std::string> runtime_alias_input_to_output_;
    std::unordered_map<std::string, std::string> runtime_alias_output_to_input_;
    std::unordered_set<std::string> bound_runtime_alias_outputs_;
    bool runtime_binding_poisoned_{false};
    bool execution_failed_{false};
    std::string execution_failure_;
    bool allocation_failed_{false};
    bool has_dynamic_shapes_{false};
    bool use_cuda_graph_{false};
    std::unique_ptr<CudaGraphExec> cuda_graph_;
    std::vector<std::shared_ptr<void>> keep_alive_;
    std::unordered_map<std::string, void*> initial_external_bindings_;
    std::unordered_map<std::string, BufferEntry> buffers_;
    std::unordered_map<std::string, std::vector<uint8_t>> host_output_staging_;
    std::unordered_map<std::string, DeviceTensor> output_device_tensors_;
    std::string timing_label_{"engine"};
    std::vector<TimingEvent> timing_events_;
    std::uint64_t transfer_event_sequence_{0};
    std::unordered_map<std::string, RuntimeMemoryTransferCounterV1> transfer_counters_;

    void allocate_buffers(nvinfer1::ICudaEngine* engine);
    void
    validate_initial_external_bindings(nvinfer1::ICudaEngine* engine,
                                       const std::vector<ModuleExternalBinding>& external_bindings);
    void validate_deferred_runtime_tensors(nvinfer1::ICudaEngine* engine);
    void free_buffers();
    void detect_dynamic_shapes(nvinfer1::ICudaEngine* engine, int32_t num_io);
    void allocate_input_buffers(nvinfer1::ICudaEngine* engine, int32_t num_io,
                                int32_t num_profiles);
    void allocate_single_input(nvinfer1::ICudaEngine* engine, const std::string& name,
                               int32_t num_profiles);
    void allocate_output_buffers(nvinfer1::ICudaEngine* engine, int32_t num_io);
    TensorMap download_host_outputs(const std::unordered_set<std::string>* selected_output_names);
    void set_dynamic_input_shapes(nvinfer1::ICudaEngine* engine, int32_t num_io,
                                  nvinfer1::OptProfileSelector selector);
    void update_dynamic_shape(const std::string& name, BufferEntry& entry,
                              const std::vector<int64_t>& new_shape);
    void validate_runtime_binding_descriptor(const RuntimeMemoryBindingV1& binding,
                                             BufferEntry& entry);
    void validate_runtime_tensor_shape(const RuntimeMemoryBindingV1& binding, BufferEntry& entry);
    void synchronize_runtime_reconfiguration(const char* operation);
    void note_runtime_input_shape_change(const BufferEntry& entry,
                                         const std::vector<int64_t>& new_shape);
    void invalidate_context_memory();
    void ensure_runtime_memory_ready() const;
    void commit_runtime_binding(const RuntimeMemoryBindingV1& binding, BufferEntry& entry);
    void commit_runtime_shape(const RuntimeMemoryShapeV1& shape, BufferEntry& entry);
    bool restore_input_shape(const std::string& name, const BufferEntry& entry);
    void execute_enqueue();
    void require_execution_success(bool success, const std::string& operation);
    void flush_timing_events();
    bool begin_timing_event(TimingEvent& event);
    void finish_timing_event(TimingEvent event) noexcept;
    void record_timed_enqueue();
    bool bind_tensor_address(const std::string& name, const BufferEntry& entry);
    void note_transfer(const std::string& name, const BufferEntry& entry, cudaMemcpyKind kind,
                       std::size_t bytes);
    bool attach_distributed_communicator();
    static bool dims_are_dynamic(const nvinfer1::Dims& dims);
    static std::vector<int64_t> dims_to_shape(const nvinfer1::Dims& dims);
    static std::size_t compute_alloc_bytes(const nvinfer1::Dims& dims, DType dtype,
                                           std::vector<int64_t>& shape_out);
    static std::size_t compute_capacity_bytes(const RuntimeMemoryBindingV1& binding);
    static DType from_trt_dtype(nvinfer1::DataType dt);
};

// Keep the module-side capability out of the RTX and legacy standard-TRT
// paths. Only modules created through IRuntimeMemoryBackendV1 use this type.
class RuntimeMemoryTrtModuleImpl final : public TrtModuleImpl,
                                         public IRuntimeMemoryModuleV1,
                                         public IRuntimeMemoryEngineIntrospectionV1,
                                         public IRuntimeMemoryTransferLedgerV1 {
  public:
    using TrtModuleImpl::TrtModuleImpl;

    void set_runtime_binding_shape(const RuntimeMemoryShapeV1& shape) override {
        TrtModuleImpl::set_runtime_binding_shape(shape);
    }
    void set_runtime_alias_pair_shape(const RuntimeMemoryAliasShapeV1& shape) override {
        TrtModuleImpl::set_runtime_alias_pair_shape(shape);
    }
    void set_runtime_input_shape(const RuntimeInputShapeV1& shape) override {
        TrtModuleImpl::set_runtime_input_shape(shape);
    }
    void bind_runtime_memory(const RuntimeMemoryBindingV1& binding) override {
        TrtModuleImpl::bind_runtime_memory(binding);
    }
    void bind_runtime_memory_alias_pair(const RuntimeMemoryAliasBindingV1& binding) override {
        TrtModuleImpl::bind_runtime_memory_alias_pair(binding);
    }
    RuntimeMemoryContextRequirementV1 context_memory_requirement() override {
        return TrtModuleImpl::context_memory_requirement();
    }
    void bind_context_memory(const RuntimeMemoryContextBlockV1& block) override {
        TrtModuleImpl::bind_context_memory(block);
    }
    bool runtime_memory_ready() const noexcept override {
        return TrtModuleImpl::runtime_memory_ready();
    }
    RuntimeMemoryEngineStatsV1 runtime_memory_engine_stats() const noexcept override {
        return TrtModuleImpl::runtime_memory_engine_stats();
    }
    RuntimeMemoryTransferSnapshotV1 runtime_memory_transfer_snapshot() const override {
        return TrtModuleImpl::runtime_memory_transfer_snapshot();
    }
    TensorMap forward_selected(const TensorMap& inputs,
                               const std::vector<std::string>& host_output_names) override {
        return TrtModuleImpl::forward_selected(inputs, host_output_names);
    }
};

} // namespace trtmc
