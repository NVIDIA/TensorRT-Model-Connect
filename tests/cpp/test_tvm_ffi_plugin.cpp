// =============================================================================
// test_tvm_ffi_plugin.cpp — Full round-trip test for TVM-FFI kernel plugin
// =============================================================================
//
// Intent:
//   Validates TvmFfiKernelPlugin (IPluginV2DynamicExt) end-to-end: register
//   a trivial add_one kernel via TVM-FFI C API, build a TRT engine, run
//   inference, verify output, and test serialize/deserialize round-trip.
//
// Preconditions:
//   - TRTMC_HAS_TRT=1 and TRTMC_HAS_TVM_FFI=1
//   - GPU with CUDA runtime available
//
// Postconditions:
//   - Input [1,2,3,4] produces output [2,3,4,5]
//   - Serialized engine produces identical results after deserialization
//
// Trace IDs: ARCH-TVM-FFI-001, UD-TVM-FFI-PLUGIN-001, UT-TVM-FFI-ROUNDTRIP-001
// =============================================================================

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

extern "C" void tvm_ffi_plugin_force_link();

static int failures = 0;

static void check(bool cond, const char* name) {
    if (!cond) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

// ---------------------------------------------------------------------------
// Trivial add_one kernel
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
    for (auto& v : h)
        v += 1.0f;
    cudaMemcpy(out->data, h.data(), nb, cudaMemcpyHostToDevice);
    result->type_index = kTVMFFINone;
    return 0;
}

class TestLogger : public nvinfer1::ILogger {
  public:
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kERROR)
            std::cerr << "[TRT] " << msg << '\n';
    }
};

// ---------------------------------------------------------------------------
// Build engine with TvmFfiKernel plugin
// ---------------------------------------------------------------------------

static nvinfer1::ICudaEngine* build_engine(TestLogger& logger) {
    auto* builder = nvinfer1::createInferBuilder(logger);
    auto* network = builder->createNetworkV2(0);
    auto* config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 26);

    auto* input = network->addInput("input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims{1, {4}});

    std::string kn = "tvm_ffi_test.add_one";
    std::string ss =
        R"({"num_inputs":1,"num_outputs":1,"outputs":[{"dims":"same_as_input_0","dtype":"float32"}],"workspace_bytes":0})";

    nvinfer1::PluginField fields[] = {
        {"kernel_name", kn.data(), nvinfer1::PluginFieldType::kCHAR,
         static_cast<int32_t>(kn.size())},
        {"shape_spec", ss.data(), nvinfer1::PluginFieldType::kCHAR,
         static_cast<int32_t>(ss.size())},
    };
    nvinfer1::PluginFieldCollection fc{2, fields};

    auto* registry = getPluginRegistry();
    // TRT 11 removed IPluginRegistry::getPluginCreator and replaced it with
    // getCreator returning IPluginCreatorInterface*; downcast to the still-
    // supported (deprecated) IPluginCreator so createPlugin() is callable.
    auto* creator =
        static_cast<nvinfer1::IPluginCreator*>(registry->getCreator("TvmFfiKernel", "1", ""));
    check(creator != nullptr, "creator found");
    if (!creator) {
        delete config;
        delete network;
        delete builder;
        return nullptr;
    }

    auto* plugin = creator->createPlugin("add_one", &fc);
    auto* layer = network->addPluginV2(&input, 1, *plugin);
    auto* output = layer->getOutput(0);
    output->setName("output");
    network->markOutput(*output);

    auto* plan = builder->buildSerializedNetwork(*network, *config);
    check(plan != nullptr, "engine built");
    if (!plan) {
        delete config;
        delete network;
        delete builder;
        return nullptr;
    }

    auto* runtime = nvinfer1::createInferRuntime(logger);
    auto* engine = runtime->deserializeCudaEngine(plan->data(), plan->size());

    delete plan;
    delete config;
    delete network;
    delete builder;
    delete runtime;
    return engine;
}

// ---------------------------------------------------------------------------
// Run engine and verify output
// ---------------------------------------------------------------------------

static void run_and_verify(nvinfer1::ICudaEngine* engine, const char* name) {
    auto* ctx = engine->createExecutionContext();
    check(ctx != nullptr, (std::string(name) + " context").c_str());
    if (!ctx)
        return;

    float h_in[] = {1, 2, 3, 4}, h_out[4] = {};
    void *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, 16);
    cudaMalloc(&d_out, 16);
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    cudaMemcpyAsync(d_in, h_in, 16, cudaMemcpyHostToDevice, stream);

    ctx->setTensorAddress("input", d_in);
    ctx->setTensorAddress("output", d_out);
    check(ctx->enqueueV3(stream), (std::string(name) + " enqueue").c_str());

    cudaMemcpyAsync(h_out, d_out, 16, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    for (int i = 0; i < 4; ++i)
        check(std::abs(h_out[i] - (h_in[i] + 1.0f)) < 1e-5f,
              (std::string(name) + " output[" + std::to_string(i) + "]").c_str());

    cudaFree(d_in);
    cudaFree(d_out);
    cudaStreamDestroy(stream);
    delete ctx;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

static void test_roundtrip() {
    TestLogger logger;
    auto* engine = build_engine(logger);
    if (engine) {
        run_and_verify(engine, "roundtrip");
        delete engine;
    }
}

static void test_serialize_deserialize() {
    TestLogger logger;
    auto* engine = build_engine(logger);
    if (!engine)
        return;

    auto* plan = engine->serialize();
    check(plan != nullptr, "serialize");
    delete engine;
    if (!plan)
        return;

    auto* runtime = nvinfer1::createInferRuntime(logger);
    auto* engine2 = runtime->deserializeCudaEngine(plan->data(), plan->size());
    check(engine2 != nullptr, "deserialize");
    delete plan;
    delete runtime;
    if (!engine2)
        return;

    run_and_verify(engine2, "serde");
    delete engine2;
}

#endif

int main() {
#if TRTMC_HAS_TRT && TRTMC_HAS_TVM_FFI
    tvm_ffi_plugin_force_link();

    // Register add_one kernel
    TVMFFIObjectHandle fn = nullptr;
    TVMFFIFunctionCreate(nullptr, &add_one_kernel, nullptr, &fn);
    TVMFFIByteArray name = {"tvm_ffi_test.add_one", 20};
    TVMFFIFunctionSetGlobal(&name, fn, 1);

    test_roundtrip();
    test_serialize_deserialize();

    if (failures > 0) {
        std::cerr << failures << " FAILED\n";
        return 1;
    }
    std::cerr << "All tvm_ffi_plugin tests passed.\n";
#else
    std::cerr << "test_tvm_ffi_plugin: skipping (no TRT/TVM-FFI)\n";
#endif
    return 0;
}
