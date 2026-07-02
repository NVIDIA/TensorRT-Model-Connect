/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Quick test: does IPluginV2DynamicExt work with createExecutionContext on GB300?
// If yes, we rewrite the TVM-FFI plugin using V2 instead of V3.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#if TRTMC_HAS_TRT && TRTMC_HAS_TVM_FFI

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <tvm/ffi/c_api.h>

static int failures = 0;

static void check(bool cond, const char* name) {
    if (!cond) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

// ---------------------------------------------------------------------------
// TVM-FFI add_one kernel (same as before)
// ---------------------------------------------------------------------------

static int add_one_kernel(void*, const TVMFFIAny* args, int32_t, TVMFFIAny* result) {
    auto* inp = static_cast<DLTensor*>(args[0].v_ptr);
    auto* out = static_cast<DLTensor*>(args[1].v_ptr);
    int64_t n = 1;
    for (int i = 0; i < inp->ndim; ++i)
        n *= inp->shape[i];
    size_t nb = static_cast<size_t>(n) * 4;
    std::vector<float> h(static_cast<size_t>(n));
    cudaMemcpy(h.data(), inp->data, nb, cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < h.size(); ++i)
        h[i] += 1.0f;
    cudaMemcpy(out->data, h.data(), nb, cudaMemcpyHostToDevice);
    result->type_index = kTVMFFINone;
    return 0;
}

// ---------------------------------------------------------------------------
// Minimal IPluginV2DynamicExt + IPluginCreator for TVM-FFI
// ---------------------------------------------------------------------------

class TvmFfiV2Plugin : public nvinfer1::IPluginV2DynamicExt {
  public:
    TvmFfiV2Plugin(const std::string& kernel_name) : kernel_name_(kernel_name) {}
    TvmFfiV2Plugin(const void* data, size_t len) {
        auto* p = static_cast<const char*>(data);
        uint32_t slen = 0;
        std::memcpy(&slen, p, 4);
        p += 4;
        kernel_name_ = std::string(p, slen);
    }

    // IPluginV2
    char const* getPluginType() const noexcept override { return "TvmFfiKernelV2"; }
    char const* getPluginVersion() const noexcept override { return "1"; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    int32_t initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    void destroy() noexcept override { delete this; }
    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }
    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }

    size_t getSerializationSize() const noexcept override { return 4 + kernel_name_.size(); }

    void serialize(void* buf) const noexcept override {
        auto* p = static_cast<char*>(buf);
        uint32_t slen = static_cast<uint32_t>(kernel_name_.size());
        std::memcpy(p, &slen, 4);
        p += 4;
        std::memcpy(p, kernel_name_.data(), slen);
    }

    // IPluginV2Ext
    nvinfer1::DataType getOutputDataType(int32_t, nvinfer1::DataType const* inputTypes,
                                         int32_t) const noexcept override {
        return inputTypes[0];
    }

    // IPluginV2DynamicExt
    TvmFfiV2Plugin* clone() const noexcept override { return new TvmFfiV2Plugin(kernel_name_); }

    nvinfer1::DimsExprs getOutputDimensions(int32_t, nvinfer1::DimsExprs const* inputs, int32_t,
                                            nvinfer1::IExprBuilder&) noexcept override {
        return inputs[0]; // same as input
    }

    bool supportsFormatCombination(int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t,
                                   int32_t) noexcept override {
        return inOut[pos].format == nvinfer1::TensorFormat::kLINEAR &&
               inOut[pos].type == nvinfer1::DataType::kFLOAT;
    }

    void configurePlugin(nvinfer1::DynamicPluginTensorDesc const*, int32_t,
                         nvinfer1::DynamicPluginTensorDesc const*, int32_t) noexcept override {}

    size_t getWorkspaceSize(nvinfer1::PluginTensorDesc const*, int32_t,
                            nvinfer1::PluginTensorDesc const*, int32_t) const noexcept override {
        return 0;
    }

    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const*,
                    void const* const* inputs, void* const* outputs, void*,
                    cudaStream_t) noexcept override {
        if (cached_fn_ == nullptr) {
            TVMFFIByteArray name_arr;
            name_arr.data = kernel_name_.c_str();
            name_arr.size = static_cast<int64_t>(kernel_name_.size());
            if (TVMFFIFunctionGetGlobal(&name_arr, &cached_fn_) != 0 || cached_fn_ == nullptr) {
                std::cerr << "[V2] Failed to resolve: " << kernel_name_ << '\n';
                return -1;
            }
        }

        DLTensor dl_in{}, dl_out{};
        auto fill = [&](DLTensor& t, void* data, const nvinfer1::PluginTensorDesc& d) {
            t.data = data;
            t.device = {kDLCUDA, 0};
            t.ndim = d.dims.nbDims;
            t.shape = const_cast<int64_t*>(reinterpret_cast<const int64_t*>(d.dims.d));
            t.strides = nullptr;
            t.byte_offset = 0;
            t.dtype = {kDLFloat, 32, 1};
        };
        fill(dl_in, const_cast<void*>(inputs[0]), inputDesc[0]);
        fill(dl_out, outputs[0], inputDesc[0]);

        TVMFFIAny args[3];
        args[0].type_index = kTVMFFIDLTensorPtr;
        args[0].v_ptr = &dl_in;
        args[1].type_index = kTVMFFIDLTensorPtr;
        args[1].v_ptr = &dl_out;
        args[2].type_index = kTVMFFIOpaquePtr;
        args[2].v_ptr = nullptr;

        TVMFFIAny result;
        result.type_index = kTVMFFINone;
        if (TVMFFIFunctionCall(cached_fn_, args, 3, &result) != 0) {
            std::cerr << "[V2] Kernel call failed\n";
            return -1;
        }
        return 0;
    }

  private:
    std::string kernel_name_;
    std::string ns_;
    void* cached_fn_{nullptr};
};

class TvmFfiV2Creator : public nvinfer1::IPluginCreator {
  public:
    TvmFfiV2Creator() {
        fields_.push_back({"kernel_name", nullptr, nvinfer1::PluginFieldType::kCHAR, 0});
        fc_.nbFields = 1;
        fc_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override { return "TvmFfiKernelV2"; }
    char const* getPluginVersion() const noexcept override { return "1"; }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override { return &fc_; }
    void setPluginNamespace(char const* ns) noexcept override { ns_ = ns ? ns : ""; }
    char const* getPluginNamespace() const noexcept override { return ns_.c_str(); }

    nvinfer1::IPluginV2* createPlugin(char const*,
                                      nvinfer1::PluginFieldCollection const* fc) noexcept override {
        std::string kernel_name;
        if (fc) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                if (std::strcmp(fc->fields[i].name, "kernel_name") == 0) {
                    kernel_name = std::string(static_cast<const char*>(fc->fields[i].data),
                                              static_cast<size_t>(fc->fields[i].length));
                }
            }
        }
        return new TvmFfiV2Plugin(kernel_name);
    }

    nvinfer1::IPluginV2* deserializePlugin(char const*, void const* data,
                                           size_t len) noexcept override {
        return new TvmFfiV2Plugin(data, len);
    }

  private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection fc_{};
    std::string ns_;
};

// Static registration using the V2 path (proven stable)
REGISTER_TENSORRT_PLUGIN(TvmFfiV2Creator);

// ---------------------------------------------------------------------------
// Test logger
// ---------------------------------------------------------------------------

class TestLogger : public nvinfer1::ILogger {
  public:
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kERROR)
            std::cerr << "[TRT] " << msg << '\n';
    }
};

// ---------------------------------------------------------------------------
// Test: full engine round-trip with V2 plugin
// ---------------------------------------------------------------------------

static void test_v2_roundtrip() {
    TestLogger logger;
    auto* builder = nvinfer1::createInferBuilder(logger);
    auto* network = builder->createNetworkV2(0);
    auto* config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 26);

    auto* input = network->addInput("input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {4}});

    // Create plugin via creator
    auto* registry = getPluginRegistry();
    // TRT 11 removed IPluginRegistry::getPluginCreator; replaced by
    // getCreator returning IPluginCreatorInterface*. Downcast preserves
    // the existing createPlugin() call on the concrete creator type.
    auto* creator =
        static_cast<nvinfer1::IPluginCreator*>(registry->getCreator("TvmFfiKernelV2", "1", ""));
    check(creator != nullptr, "V2 creator found");
    if (creator == nullptr) {
        delete config;
        delete network;
        delete builder;
        return;
    }

    std::string kn = "tvm_ffi_test.add_one_v2";
    nvinfer1::PluginField f{"kernel_name", kn.data(), nvinfer1::PluginFieldType::kCHAR,
                            static_cast<int32_t>(kn.size())};
    nvinfer1::PluginFieldCollection fc{1, &f};
    auto* plugin = creator->createPlugin("v2test", &fc);
    check(plugin != nullptr, "V2 createPlugin");

    auto* layer = network->addPluginV2(&input, 1, *plugin);
    auto* output = layer->getOutput(0);
    output->setName("output");
    network->markOutput(*output);

    std::cerr << "Building V2 engine...\n";
    auto* plan = builder->buildSerializedNetwork(*network, *config);
    check(plan != nullptr, "V2 engine built");
    if (plan == nullptr) {
        delete config;
        delete network;
        delete builder;
        return;
    }

    auto* runtime = nvinfer1::createInferRuntime(logger);
    auto* engine = runtime->deserializeCudaEngine(plan->data(), plan->size());
    check(engine != nullptr, "V2 engine deserialized");
    delete plan;
    delete config;
    delete network;
    delete builder;
    delete runtime;
    if (engine == nullptr)
        return;

    std::cerr << "Creating V2 execution context...\n";
    auto* ctx = engine->createExecutionContext();
    check(ctx != nullptr, "V2 createExecutionContext");
    if (ctx == nullptr) {
        delete engine;
        return;
    }

    std::cerr << "Running V2 inference...\n";
    float h_in[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_out[4] = {};
    void *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, 16);
    cudaMalloc(&d_out, 16);
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(d_in, h_in, 16, cudaMemcpyHostToDevice, stream);

    ctx->setTensorAddress("input", d_in);
    ctx->setTensorAddress("output", d_out);
    bool ok = ctx->enqueueV3(stream);
    check(ok, "V2 enqueueV3");

    cudaMemcpyAsync(h_out, d_out, 16, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    float expected[] = {2.0f, 3.0f, 4.0f, 5.0f};
    for (int i = 0; i < 4; ++i) {
        check(std::abs(h_out[i] - expected[i]) < 1e-5f, "V2 output correct");
    }

    std::cerr << "V2 output: [" << h_out[0] << ", " << h_out[1] << ", " << h_out[2] << ", "
              << h_out[3] << "]\n";

    cudaFree(d_in);
    cudaFree(d_out);
    cudaStreamDestroy(stream);
    delete ctx;
    delete engine;
}

#endif

int main() {
#if TRTMC_HAS_TRT && TRTMC_HAS_TVM_FFI
    // Register add_one kernel
    TVMFFIObjectHandle fn = nullptr;
    TVMFFIFunctionCreate(nullptr, &add_one_kernel, nullptr, &fn);
    TVMFFIByteArray name = {"tvm_ffi_test.add_one_v2", 23};
    TVMFFIFunctionSetGlobal(&name, fn, 1);

    test_v2_roundtrip();

    if (failures > 0) {
        std::cerr << failures << " FAILED\n";
        return 1;
    }
    std::cerr << "All V2 plugin tests passed.\n";
#else
    std::cerr << "Skipping (no TRT or TVM-FFI)\n";
#endif
    return 0;
}
