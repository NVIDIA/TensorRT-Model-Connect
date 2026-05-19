#include "common/trtUtils.h"
#include "runtime/llmInferenceRuntime.h"
#include "trtmc/pipeline.h"

#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#ifndef TRTMC_DEFAULT_EDGE_LLM_PLUGIN_PATH
#define TRTMC_DEFAULT_EDGE_LLM_PLUGIN_PATH ""
#endif

namespace {

class EdgeLlmPipeline final : public trtmc::IPipeline {
  public:
    EdgeLlmPipeline(std::filesystem::path engine_dir, std::string model_id)
        : engine_dir_(std::move(engine_dir)), model_id_(std::move(model_id)) {
        const std::string default_plugin = TRTMC_DEFAULT_EDGE_LLM_PLUGIN_PATH;
        if (std::getenv("EDGELLM_PLUGIN_PATH") == nullptr && !default_plugin.empty()) {
            setenv("EDGELLM_PLUGIN_PATH", default_plugin.c_str(), 0);
        }

        plugin_handle_ = trt_edgellm::loadEdgellmPluginLib();
        if (!plugin_handle_) {
            throw std::runtime_error(
                "Failed to load TensorRT Edge-LLM plugin library. Set EDGELLM_PLUGIN_PATH "
                "to libNvInfer_edgellm_plugin.so from the Edge-LLM build.");
        }
        const auto stream_status = cudaStreamCreate(&stream_);
        if (stream_status != cudaSuccess) {
            throw std::runtime_error("Failed to create CUDA stream for TensorRT Edge-LLM runtime");
        }
        runtime_ = std::make_unique<trt_edgellm::rt::LLMInferenceRuntime>(
            engine_dir_.string(), std::string{}, std::unordered_map<std::string, std::string>{},
            stream_);
    }

    ~EdgeLlmPipeline() override {
        runtime_.reset();
        if (stream_ != nullptr)
            cudaStreamDestroy(stream_);
    }

    trtmc::TextResult generate(const std::string& prompt,
                               const trtmc::GenerateConfig& cfg) override {
        trt_edgellm::rt::LLMGenerationRequest request;
        trt_edgellm::rt::LLMGenerationRequest::Request item;
        trt_edgellm::rt::Message message;
        message.role = "user";
        message.contents.push_back({"text", prompt});
        item.messages.push_back(std::move(message));
        request.requests.push_back(std::move(item));
        request.temperature = cfg.temperature;
        request.topP = cfg.top_p;
        request.topK = cfg.top_k;
        request.maxGenerateLength = cfg.max_new_tokens;
        request.applyChatTemplate = cfg.use_chat_template;
        request.enableThinking = cfg.enable_thinking;

        trt_edgellm::rt::LLMGenerationResponse response;
        const auto t0 = std::chrono::steady_clock::now();
        if (!runtime_->handleRequest(request, response, stream_))
            throw std::runtime_error("TensorRT Edge-LLM request failed");
        cudaStreamSynchronize(stream_);
        const auto t1 = std::chrono::steady_clock::now();

        trtmc::TextResult out;
        if (!response.outputTexts.empty())
            out.text = response.outputTexts.front();
        if (!response.outputIds.empty())
            out.token_ids = response.outputIds.front();
        out.decode_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return out;
    }

    const char* model_id() const override { return model_id_.c_str(); }

    const char* pipeline_type() const override { return "tensorrt-edge-llm"; }

  private:
    std::filesystem::path engine_dir_;
    std::string model_id_;
    cudaStream_t stream_{nullptr};
    std::unique_ptr<void, trt_edgellm::DlDeleter> plugin_handle_;
    std::unique_ptr<trt_edgellm::rt::LLMInferenceRuntime> runtime_;
};

} // namespace

extern "C" trtmc::IPipeline* trtmc_create_deployment_provider_pipeline(const char* engine_dir,
                                                                       const char* bundle_path) {
    return new EdgeLlmPipeline(engine_dir == nullptr ? "" : engine_dir,
                               bundle_path == nullptr ? "" : bundle_path);
}

extern "C" void trtmc_destroy_deployment_provider_pipeline(trtmc::IPipeline* pipeline) {
    delete pipeline;
}

extern "C" const char* trtmc_deployment_provider_name() {
    return "tensorrt-edge-llm";
}
