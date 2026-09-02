/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// TrtModuleImpl: concrete ITrtModule backed by a TRT engine.
// Compiled inside backend DSOs only (libtrtmc_backend_trt.so / _rtx.so).

#include "runtime/backend/prebound_backend.h"
#include "runtime/backend/trt_logger.h"

#include <NvInfer.h>
#include <cstddef>
#include <cuda_runtime_api.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace trtmc {

class CudaGraphExec;
class TrtModuleImplTestPeer;

class TrtModuleImpl final : public ITrtModule {
  public:
    // Backend creates engine + context, passes them in.
    // The engine must outlive this module (caller manages lifetime via keep_alive).
    TrtModuleImpl(nvinfer1::ICudaEngine* engine, nvinfer1::IExecutionContext* ctx,
                  cudaStream_t stream, int32_t profile_idx = 0,
                  void* distributed_communicator = nullptr,
                  const std::vector<ModuleExternalBinding>& external_bindings = {});
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
    bool cuda_graph_captured() const override;
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
    bool ok() const override { return ctx_ != nullptr; }
    void keep_alive(std::shared_ptr<void> resource) override;

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
    };
    struct TimingEvent {
        cudaEvent_t start{nullptr};
        cudaEvent_t stop{nullptr};
    };

    nvinfer1::ICudaEngine* engine_{nullptr};
    nvinfer1::IExecutionContext* ctx_{nullptr};
    cudaStream_t stream_{nullptr};
    int32_t profile_idx_{0};
    void* distributed_communicator_{nullptr};
    bool has_dynamic_shapes_{false};
    bool use_cuda_graph_{false};
    bool alias_groups_ready_{true};
    std::unique_ptr<CudaGraphExec> cuda_graph_;
    std::vector<std::shared_ptr<void>> keep_alive_;
    std::unordered_map<std::string, void*> initial_external_bindings_;
    std::unordered_map<std::string, std::size_t> initial_external_binding_capacities_;
    std::unordered_map<std::string, BufferEntry> buffers_;
    std::unordered_map<std::string, std::string> alias_input_by_output_;
    std::unordered_map<std::string, std::vector<std::string>> alias_outputs_by_input_;
    std::unordered_map<std::string, std::vector<uint8_t>> host_output_staging_;
    std::unordered_map<std::string, DeviceTensor> output_device_tensors_;
    std::string timing_label_{"engine"};
    std::vector<TimingEvent> timing_events_;

    void allocate_buffers(nvinfer1::ICudaEngine* engine);
    void discover_tensor_aliases(nvinfer1::ICudaEngine* engine);
    void
    validate_initial_external_bindings(nvinfer1::ICudaEngine* engine,
                                       const std::vector<ModuleExternalBinding>& external_bindings);
    void bind_alias_group(const std::string& input_name, void* ptr);
    bool should_allocate_input(const std::string& name, std::size_t nbytes) const;
    void validate_alias_outputs_exist(const std::vector<std::string>& output_names) const;
    void bind_alias_outputs_or_invalidate(const std::vector<std::string>& output_names, void* ptr);
    void reset_cuda_graph_if_rebound(void* previous_ptr, void* ptr);
    void validate_alias_groups_bound() const;
    void free_buffers();
    void detect_dynamic_shapes(nvinfer1::ICudaEngine* engine, int32_t num_io);
    void allocate_input_buffers(nvinfer1::ICudaEngine* engine, int32_t num_io,
                                int32_t num_profiles);
    void allocate_single_input(nvinfer1::ICudaEngine* engine, const std::string& name,
                               int32_t num_profiles);
    void allocate_output_buffers(nvinfer1::ICudaEngine* engine, int32_t num_io);
    void set_dynamic_input_shapes(nvinfer1::ICudaEngine* engine, int32_t num_io,
                                  nvinfer1::OptProfileSelector selector);
    void update_dynamic_shape(const std::string& name, BufferEntry& entry,
                              const std::vector<int64_t>& new_shape);
    void execute_enqueue();
    void flush_timing_events();
    bool begin_timing_event(TimingEvent& event);
    void finish_timing_event(TimingEvent event);
    void record_timed_enqueue();
    bool bind_tensor_address(const std::string& name, const BufferEntry& entry);
    bool attach_distributed_communicator();
    static bool dims_are_dynamic(const nvinfer1::Dims& dims);
    static std::vector<int64_t> dims_to_shape(const nvinfer1::Dims& dims);
    static std::size_t compute_alloc_bytes(const nvinfer1::Dims& dims, DType dtype,
                                           std::vector<int64_t>& shape_out);
    static DType from_trt_dtype(nvinfer1::DataType dt);
};

} // namespace trtmc
