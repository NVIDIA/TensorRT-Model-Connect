/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-SEG-CPP-01-SEGMENT
// Architecture:   ARCH-MODPLUG-001
// Unit Design:    UD-SEG-01
// Intent:         SegmentPipeline construction, class-map parsing, 4D logits,
//                 and single-output mask branch coverage
// Preconditions:  TRT headers and CUDA available
// Postconditions: SegmentPipeline constructs with mock engines and exposes
//                 expected mask outputs
// =============================================================================

#include "runtime/backend/trt_module_impl.h"
#include "runtime/core/trt_common.h"
#include "runtime/models/segformer/segment_pipeline.h"

#include <NvInfer.h>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

static trtmc::TrtLogger g_logger;

class CountingTrtModule final : public trtmc::ITrtModule {
  public:
    explicit CountingTrtModule(std::unique_ptr<trtmc::ITrtModule> delegate)
        : delegate_(std::move(delegate)) {}

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++host_forward_calls;
        return delegate_->forward(inputs);
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap& inputs) override {
        return delegate_->forward_device(inputs);
    }
    void forward_device_async(const trtmc::DeviceTensorMap& inputs) override {
        delegate_->forward_device_async(inputs);
    }
    void forward_async(const trtmc::TensorMap& inputs) override {
        ++async_forward_calls;
        delegate_->forward_async(inputs);
    }
    void sync() override { delegate_->sync(); }
    cudaStream_t stream() const override { return delegate_->stream(); }
    void enable_cuda_graph() override { delegate_->enable_cuda_graph(); }
    bool cuda_graph_active() const override { return delegate_->cuda_graph_active(); }
    int32_t profile_idx() const override { return delegate_->profile_idx(); }
    std::vector<trtmc::TensorInfo> input_info() const override { return delegate_->input_info(); }
    std::vector<trtmc::TensorInfo> output_info() const override { return delegate_->output_info(); }
    bool has_input(const std::string& name) const override { return delegate_->has_input(name); }
    bool has_output(const std::string& name) const override { return delegate_->has_output(name); }
    trtmc::DType tensor_dtype(const std::string& name) const override {
        return delegate_->tensor_dtype(name);
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        return delegate_->tensor_shape(name);
    }
    std::vector<int64_t> input_profile_shape(const std::string& name, int32_t profile_idx,
                                             trtmc::ProfileShapeSelector selector) const override {
        return delegate_->input_profile_shape(name, profile_idx, selector);
    }
    int32_t optimization_profile_count() const override {
        return delegate_->optimization_profile_count();
    }
    void* device_ptr(const std::string& name) const override { return delegate_->device_ptr(name); }
    void bind_external(const std::string& name, void* ptr) override {
        delegate_->bind_external(name, ptr);
    }
    void bind_external(const std::string& name, void* ptr,
                       const std::vector<int64_t>& shape) override {
        delegate_->bind_external(name, ptr, shape);
    }
    int32_t input_rank(const std::string& name) const override {
        return delegate_->input_rank(name);
    }
    bool input_is_dynamic(const std::string& name) const override {
        return delegate_->input_is_dynamic(name);
    }
    void reset_execution_context() override { delegate_->reset_execution_context(); }
    void set_timing_label(std::string label) override {
        delegate_->set_timing_label(std::move(label));
    }
    bool ok() const override { return delegate_->ok(); }
    void keep_alive(std::shared_ptr<void> resource) override {
        delegate_->keep_alive(std::move(resource));
    }

    int32_t host_forward_calls{0};
    int32_t async_forward_calls{0};

  private:
    std::unique_ptr<trtmc::ITrtModule> delegate_;
};

static trtmc::SegformerPreprocessConfig make_test_preprocess_config() {
    trtmc::SegformerPreprocessConfig config;
    config.input_image_h = 4;
    config.input_image_w = 4;
    config.output_h = 4;
    config.output_w = 4;
    return config;
}

// Mock: pixel_values[3,4,4] float -> output_mask[1,16] float.
static trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_segment_engine() {
    auto b = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    auto n = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(b->createNetworkV2(0));
    auto c = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(b->createBuilderConfig());
    c->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* pv =
        n->addInput("pixel_values", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{3, {3, 4, 4}});

    float cv[16];
    for (int i = 0; i < 16; ++i)
        cv[i] = static_cast<float>(i);
    auto* cst = n->addConstant(nvinfer1::Dims{1, {16}},
                               nvinfer1::Weights{nvinfer1::DataType::kFLOAT, cv, 16});
    cst->getOutput(0)->setName("output_mask");
    n->markOutput(*cst->getOutput(0));

    n->addIdentity(*pv)->getOutput(0)->setName("_pv");

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(b->buildSerializedNetwork(*n, *c));
    if (!plan)
        return nullptr;
    auto rt = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
}

// Mock: pixel_values[3,4,4] float -> mask[1] float.
static trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_segment_engine_mask_output() {
    auto b = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    auto n = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(b->createNetworkV2(0));
    auto c = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(b->createBuilderConfig());
    c->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* pv =
        n->addInput("pixel_values", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{3, {3, 4, 4}});

    float cv[1] = {1.0f};
    auto* cst = n->addConstant(nvinfer1::Dims{1, {1}},
                               nvinfer1::Weights{nvinfer1::DataType::kFLOAT, cv, 1});
    cst->getOutput(0)->setName("mask");
    n->markOutput(*cst->getOutput(0));

    n->addIdentity(*pv)->getOutput(0)->setName("_pv");

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(b->buildSerializedNetwork(*n, *c));
    if (!plan)
        return nullptr;
    auto rt = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
}

// Mock: pixel_values[3,4,4] float -> logits[1,2,2,2] float.
static trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_segment_engine_4d() {
    auto b = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    auto n = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(b->createNetworkV2(0));
    auto c = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(b->createBuilderConfig());
    c->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* pv =
        n->addInput("pixel_values", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{3, {3, 4, 4}});

    const float cv[8] = {
        1.0F, 0.0F, 0.0F, 1.0F, // class 0
        0.0F, 1.0F, 1.0F, 0.0F, // class 1
    };
    auto* cst = n->addConstant(nvinfer1::Dims{4, {1, 2, 2, 2}},
                               nvinfer1::Weights{nvinfer1::DataType::kFLOAT, cv, 8});
    cst->getOutput(0)->setName("logits");
    n->markOutput(*cst->getOutput(0));

    n->addIdentity(*pv)->getOutput(0)->setName("_pv");

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(b->buildSerializedNetwork(*n, *c));
    if (!plan)
        return nullptr;
    auto rt = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
}

// Mock: pixel_values[3,4,4] float -> logits[2,4,4] float.
static trtmc::TrtUniquePtr<nvinfer1::ICudaEngine> build_segment_engine_2d() {
    auto b = trtmc::TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(g_logger));
    auto n = trtmc::TrtUniquePtr<nvinfer1::INetworkDefinition>(b->createNetworkV2(0));
    auto c = trtmc::TrtUniquePtr<nvinfer1::IBuilderConfig>(b->createBuilderConfig());
    c->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 20);

    auto* pv =
        n->addInput("pixel_values", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{3, {3, 4, 4}});

    float cv[32];
    for (int i = 0; i < 32; ++i)
        cv[i] = static_cast<float>(i % 2);
    auto* cst = n->addConstant(nvinfer1::Dims{3, {2, 4, 4}},
                               nvinfer1::Weights{nvinfer1::DataType::kFLOAT, cv, 32});
    cst->getOutput(0)->setName("logits");
    n->markOutput(*cst->getOutput(0));

    n->addIdentity(*pv)->getOutput(0)->setName("_pv");

    auto plan = trtmc::TrtUniquePtr<nvinfer1::IHostMemory>(b->buildSerializedNetwork(*n, *c));
    if (!plan)
        return nullptr;
    auto rt = trtmc::TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(g_logger));
    return trtmc::TrtUniquePtr<nvinfer1::ICudaEngine>(
        rt->deserializeCudaEngine(plan->data(), plan->size()));
}

static void test_segment_pipeline() {
    auto engine = build_segment_engine();
    if (!engine) {
        std::cerr << "SKIP segment\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                         engine->createExecutionContext(), stream);
    trtmc::SegmentPipeline pipeline(std::move(module), make_test_preprocess_config());

    check(std::string(pipeline.pipeline_type()) == "SegmentPipeline", "segment name");

    float img[3 * 4 * 4] = {0};
    auto result = pipeline.segment(img, 4, 4);
    check(!result.mask.empty(), "segment output has values");

    cudaStreamDestroy(stream);
}

static void test_segment_with_class_output() {
    auto engine = build_segment_engine_2d();
    if (!engine) {
        std::cerr << "SKIP segment_2d\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                         engine->createExecutionContext(), stream);
    trtmc::SegmentPipeline pipeline(std::move(module), make_test_preprocess_config());

    float img[3 * 2 * 3] = {0};
    auto result = pipeline.segment(img, 2, 3);
    check(result.mask.size() == 6, "segment 2d: mask matches source image size");
    check(result.height == 2, "segment 2d: height matches source image");
    check(result.width == 3, "segment 2d: width matches source image");

    cudaStreamDestroy(stream);
}

static void test_segment_validates() {
    bool threw = false;
    try {
        trtmc::SegmentPipeline pipeline(nullptr);
    } catch (const std::exception&) {
        threw = true;
    }
    check(threw, "segment: null model throws");
}

static void test_segment_4d_output() {
    auto engine = build_segment_engine_4d();
    if (!engine) {
        std::cerr << "SKIP segment_4d\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto delegate = std::make_unique<trtmc::TrtModuleImpl>(
        engine.get(), engine->createExecutionContext(), stream);
    auto module = std::make_unique<CountingTrtModule>(std::move(delegate));
    auto* counting_module = module.get();
    trtmc::SegmentPipeline pipeline(std::move(module), make_test_preprocess_config());

    float img[3 * 3 * 3] = {0};
    auto result = pipeline.segment(img, 3, 3);
    const std::vector<int32_t> expected_mask{0, 0, 1, 0, 0, 0, 1, 0, 0};
    check(result.mask == expected_mask, "segment 4d: GPU bilinear class map matches golden");
    check(result.height == 3, "segment 4d: height = 3");
    check(result.width == 3, "segment 4d: width = 3");
    check(counting_module->host_forward_calls == 0, "segment 4d: avoids host logits forward path");
    check(counting_module->async_forward_calls == 1,
          "segment 4d: uses device-resident async forward path");

    cudaStreamDestroy(stream);
}

static void test_segment_mask_named_output() {
    auto engine = build_segment_engine_mask_output();
    if (!engine) {
        std::cerr << "SKIP segment_mask\n";
        return;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    auto module = std::make_unique<trtmc::TrtModuleImpl>(engine.get(),
                                                         engine->createExecutionContext(), stream);
    trtmc::SegmentPipeline pipeline(std::move(module), make_test_preprocess_config());

    float img[3 * 4 * 4] = {0};
    auto result = pipeline.segment(img, 4, 4);
    check(result.mask.size() == 1, "segment mask: 1 value in mask");

    cudaStreamDestroy(stream);
}

int main() {
    test_segment_pipeline();
    test_segment_with_class_output();
    test_segment_validates();
    test_segment_4d_output();
    test_segment_mask_named_output();

    if (failures > 0)
        std::cerr << failures << " FAILED\n";
    return failures;
}
