#pragma once

// TensorRT-Model-Connect MVP user API proposal.
//
// Design rule:
//   Users always load one Model from a bundle. Common tasks use typed request
//   objects with model.run(request). Plugin-specific and low-level control uses
//   discoverable endpoints. An endpoint is not an arbitrary plugin method: in
//   the MVP it is a named DLPack tensor IO boundary, normally one TensorRT
//   engine's inputs and outputs. Host-side text/token/image preprocessing stays
//   in the high-level Model APIs. There is no separate TextSession API;
//   token/logits control is represented as tensor endpoints implemented by the
//   loaded plugin.
//
// Normal end-to-end text generation:
//
//     #include <trtmc/model.h>
//     #include <iostream>
//
//     int main() {
//         trtmc::Model model = trtmc::load("qwen3.trtfb");
//
//         trtmc::TextGenerationRequest req{"The capital of France is"};
//         req.options.max_new_tokens = 64;
//         req.options.temperature = 0.7F;
//
//         trtmc::TextResult out = model.run(req);
//         std::cout << out.text << "\n";
//     }
//
// Batched text generation:
//
//     std::vector<trtmc::TextGenerationRequest> batch;
//     batch.emplace_back("One sentence about GPUs.");
//     batch.emplace_back("One sentence about TensorRT.");
//
//     std::vector<trtmc::TextResult> out = model.run_batch(batch);
//
// Streaming text generation:
//
//     model.stream(trtmc::TextGenerationRequest{"Count to ten."},
//         [](const trtmc::TextChunk& chunk) {
//             std::cout << chunk.text << std::flush;
//             return trtmc::StreamControl::Continue;
//         });
//
// Multimodal text generation:
//
//     trtmc::ImageView image;
//     image.data = pixels.data();
//     image.width = width;
//     image.height = height;
//     image.channels = 3;
//
//     trtmc::TextGenerationRequest req;
//     req.input.push_back(std::string{"Describe this image"});
//     req.input.push_back(image);
//
//     trtmc::TextResult out = model.run(req);
//
// Discovering plugin endpoints:
//
//     for (const std::string& name : model.list_endpoints()) {
//         std::cout << name << "\n";
//     }
//
//     trtmc::Endpoint decode = model.endpoint("text.decode");
//     trtmc::EndpointDescription desc = decode.describe();
//     std::cout << desc.summary << "\n";
//
// Diffusion endpoint shape:
//
//     // A diffusion bundle with T5, DiT, and VAE engines may publish one endpoint
//     // per compiled tensor boundary, for example:
//     //   "text_encoder.t5"
//     //   "denoiser.dit"
//     //   "vae.decode"
//     //
//     // Do not publish a separate "embedding" endpoint unless embedding is a
//     // separately declared component boundary in the bundle. If embedding is
//     // part of the T5 engine, it belongs behind "text_encoder.t5".
//     //
//     // model.run(ImageGenerationRequest{...}) still remains the normal E2E
//     // API. Endpoints let advanced users drive the component boundaries.
//
// Proposed decoder endpoint shape:
//
//     // Today's decoder bundles contain one "engine_plan" section, and the
//     // runtime may create separate prefill/decode execution contexts from
//     // optimization profiles. They do not yet publish "llm.prefill" or
//     // "llm.decode" endpoint records in the bundle metadata.
//     //
//     // If we promote that boundary into the endpoint API, the endpoint names
//     // could look like:
//     //   "llm.prefill"   profile 0 of engine_plan
//     //   "llm.decode"    profile 1 or default profile of engine_plan
//
// Logits-level text control through proposed endpoints:
//
//     trtmc::Endpoint prefill = model.endpoint("llm.prefill");
//     trtmc::Endpoint decode = model.endpoint("llm.decode");
//
//     // Applications that need logits-level control own tokenization and pass
//     // token IDs to the endpoint as a DLPack int32 tensor.
//     std::vector<int32_t> prompt_ids = my_tokenizer("The answer is");
//     int64_t token_ids_shape[] = {
//         static_cast<int64_t>(prompt_ids.size())};
//     DLTensor token_ids{};
//     token_ids.data = prompt_ids.data();
//     token_ids.device = {kDLCPU, 0};
//     token_ids.ndim = 1;
//     token_ids.shape = token_ids_shape;
//     token_ids.strides = nullptr;
//     token_ids.byte_offset = 0;
//     token_ids.dtype = {kDLInt, 32, 1};
//
//     trtmc::EndpointRequest prefill_req;
//     prefill_req.inputs.emplace_back("token_ids", &token_ids);
//     trtmc::EndpointResult prefill_out = prefill.run(prefill_req);
//     trtmc::NamedTensor state = prefill_out.output("state");
//     int32_t next_token = start_token_id;
//
//     for (int i = 0; i < 32; ++i) {
//         int64_t token_id_shape[] = {1};
//         DLTensor token_id{};
//         token_id.data = &next_token;
//         token_id.device = {kDLCPU, 0};
//         token_id.ndim = 1;
//         token_id.shape = token_id_shape;
//         token_id.strides = nullptr;
//         token_id.byte_offset = 0;
//         token_id.dtype = {kDLInt, 32, 1};
//
//         trtmc::EndpointRequest decode_req;
//         decode_req.inputs.push_back(state);
//         decode_req.inputs.emplace_back("token_id", &token_id);
//         decode_req.requested_outputs.push_back("logits");
//         decode_req.requested_outputs.push_back("state");
//         trtmc::EndpointResult step = decode.run(decode_req);
//
//         const DLTensor& logits = step.output("logits").tensor.dl_tensor();
//         next_token = my_sampler(logits);
//         state = step.output("state");
//     }
//
// Image generation:
//
//     trtmc::ImageGenerationRequest req{"a red chair in a studio"};
//     req.options.width = 1024;
//     req.options.height = 1024;
//     req.options.steps = 28;
//
//     trtmc::ImageResult image = model.run(req);
//
// Speech-to-text:
//
//     trtmc::AudioView audio{
//         samples.data(), static_cast<int64_t>(samples.size()), 16000};
//
//     trtmc::TranscriptionResult transcript =
//         model.run(trtmc::TranscriptionRequest{audio});
//
// Implementation model:
//   Model is the public C++ user handle. Internally, load() reads bundle
//   metadata, selects one registered runtime factory from runtime_strategy, and
//   stores the created model-specific runtime behind Model. The runtime
//   overrides only the run(...) methods and endpoints it supports. Endpoint
//   names, schemas, and descriptions are runtime-provided bundle contract, but
//   endpoints should correspond to declared bundle components rather than
//   private helper functions.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

// DLPack tensor structs live in the global namespace. The final SDK header
// should include <dlpack/dlpack.h>; this proposal forward-declares them so the
// API shape is visible without vendoring that dependency into the docs tree.
struct DLTensor;
struct DLManagedTensorVersioned;

namespace trtmc {

// Error thrown by the C++ SDK when loading fails, a requested task is
// unsupported by the bundle, or a runtime operation reports a recoverable
// user-visible failure.
class Error : public std::runtime_error {
  public:
    explicit Error(std::string message) : std::runtime_error(std::move(message)) {}
};

// Options shared by load().
//
// runtime_cache_path:
//   Optional directory for runtime-generated cache artifacts. For example, a
//   backend may store JIT-compiled kernels, deserialized runtime blobs, or other
//   per-machine acceleration artifacts here. Empty means the backend chooses its
//   default or disables persistent caching.
//
// device_id:
//   CUDA device ordinal to use for model execution. -1 means use the process or
//   backend default device.
//
struct LoadOptions {
    std::string runtime_cache_path;
    int32_t device_id{-1};
};

// Capability advertised by a bundle.
//
// Normal application code does not need this; it can call model.run(request)
// and handle trtmc::Error if the bundle does not support that request. This is
// for generic tools, model browsers, and servers that need to inspect a bundle
// before choosing which request UI or route to expose.
//
// Stable SDK capability names should be lowercase dotted strings, for example:
//   "text-generation"
//   "text-generation.batch"
//   "text-generation.stream"
//   "image-generation"
//   "transcription"
//   "embedding"
//   "endpoint"
//
// Plugins may publish namespaced capabilities for model-specific behavior, for
// example "qwen-image.edit" or "llm.logits".
struct Capability {
    std::string name;
    std::string description;
    bool stable{true};
};

// Execution options for high-level requests and endpoint execution.
//
// stream:
//   cudaStream_t when using CUDA, passed as void* to keep CUDA headers out of
//   the public MVP header. nullptr means the runtime default stream.
//
// synchronize:
//   If true, the call does not return until work is complete.
struct RunOptions {
    void* stream{nullptr};
    bool synchronize{true};
};

// Borrowed image view for multimodal input.
//
// data:
//   Caller-owned HWC RGB float32 image memory in [0, 1]. The runtime reads it
//   during the call.
//
// width/height:
//   Logical image dimensions.
//
// Advanced layouts:
//   Use a DLPack tensor through an endpoint when calling a component that
//   expects CHW/NCHW/NHWC, device memory, uint8, or model-specific tensor
//   layout.
struct ImageView {
    const float* data{nullptr};
    int32_t width{0};
    int32_t height{0};
    int32_t channels{3};
};

// Borrowed audio view for speech or multimodal input.
//
// samples:
//   Interleaved float32 PCM samples owned by the caller.
//
// num_samples:
//   Number of float samples, not bytes.
//
// sample_rate:
//   Input sample rate in Hz. 0 means model default if supported.
//
// channels:
//   Number of interleaved channels. 1 means mono.
struct AudioView {
    const float* samples{nullptr};
    int64_t num_samples{0};
    int32_t sample_rate{0};
    int32_t channels{1};
};

// One user-facing input item in a multimodal request.
//
// This is intentionally limited to natural input modalities. Token IDs and raw
// tensors are endpoint-level values, not high-level request content.
using Input = std::variant<std::string, ImageView, AudioView>;

// Text generation sampling options.
//
// max_new_tokens:
//   Maximum number of generated tokens after the prompt.
//
// temperature:
//   Sampling temperature. 0 can be interpreted by the runtime as greedy if the
//   model supports that convention.
//
// top_k/top_p:
//   Common nucleus/top-k sampling controls.
//
// seed:
//   -1 means non-deterministic or runtime default.
//
// return_token_ids:
//   Controls whether TextResult::token_ids is populated.
struct TextOptions {
    int32_t max_new_tokens{128};
    float temperature{1.0F};
    int32_t top_k{1};
    float top_p{1.0F};
    int32_t seed{-1};
    bool return_token_ids{true};
};

// Text generation request. Passing this request type selects text generation.
//
// input:
//   Ordered prompt input. A plain text prompt is represented as one string.
//
// options:
//   Sampling and result-shaping options.
struct TextGenerationRequest {
    std::vector<Input> input;
    TextOptions options;
    RunOptions run;

    TextGenerationRequest() = default;
    explicit TextGenerationRequest(std::string prompt) {
        input.emplace_back(std::move(prompt));
    }
};

// Final text generation result.
//
// text:
//   Decoded generated text.
//
// token_ids:
//   Generated token IDs when requested by TextOptions::return_token_ids.
struct TextResult {
    std::string text;
    std::vector<int32_t> token_ids;
};

// Incremental text chunk delivered by Model::stream(TextGenerationRequest, ...).
//
// text/token_ids:
//   Newly emitted content for this callback invocation.
//
// is_final:
//   True on the final callback when the runtime can provide a clean terminal
//   event. Callers should still treat stream() returning normally as completion.
struct TextChunk {
    std::string text;
    std::vector<int32_t> token_ids;
    bool is_final{false};
};

// Callback decision for Model::stream().
enum class StreamControl {
    Continue,
    Stop,
};

using TextStreamCallback = std::function<StreamControl(const TextChunk&)>;

// Image generation options.
//
// width/height:
//   0 means model default.
//
// steps:
//   <0 means model default.
//
// guidance_scale:
//   <0 means model default or disabled.
//
// seed:
//   -1 means non-deterministic or runtime default.
struct ImageOptions {
    int32_t width{0};
    int32_t height{0};
    int32_t steps{-1};
    float guidance_scale{-1.0F};
    int32_t seed{-1};
};

// Image generation request. Passing this request type selects image generation.
struct ImageGenerationRequest {
    std::string prompt;
    ImageOptions options;
    RunOptions run;

    ImageGenerationRequest() = default;
    explicit ImageGenerationRequest(std::string prompt_text)
        : prompt(std::move(prompt_text)) {}
};

// Image generation result.
//
// pixels:
//   HWC RGB float32 pixels in [0, 1] for the MVP. A later API can add encoded image
//   formats without changing the request/response shape.
struct ImageResult {
    std::vector<float> pixels;
    int32_t width{0};
    int32_t height{0};
    int32_t channels{3};
};

// Transcription options.
//
// language:
//   Optional language hint, such as "en". Empty means auto-detect or runtime
//   default.
//
// translate_to_english:
//   Requests translation instead of source-language transcription when the model
//   supports it.
//
// return_segments:
//   Requests timestamped segments in TranscriptionResult::segments.
struct TranscriptionOptions {
    std::string language;
    bool translate_to_english{false};
    bool return_segments{false};
};

// Transcription request. Passing this request type selects transcription.
struct TranscriptionRequest {
    AudioView audio;
    TranscriptionOptions options;
    RunOptions run;

    TranscriptionRequest() = default;
    explicit TranscriptionRequest(AudioView input_audio) : audio(input_audio) {}
};

// Timestamped transcription segment.
struct TranscriptionSegment {
    int64_t start_ms{0};
    int64_t end_ms{0};
    std::string text;
};

// Speech-to-text result.
struct TranscriptionResult {
    std::string text;
    std::vector<TranscriptionSegment> segments;
};

// Embedding request. Passing this request type selects embedding.
struct EmbeddingRequest {
    std::vector<Input> input;
    RunOptions run;

    EmbeddingRequest() = default;
    explicit EmbeddingRequest(std::string text) {
        input.emplace_back(std::move(text));
    }
};

// Embedding result.
struct EmbeddingResult {
    std::vector<float> values;
    int32_t dimensions{0};
};

// Shape contract for endpoint tensor values.
//
// dims:
//   Tensor dimensions. Use -1 for dynamic dimensions.
//
// rank_dynamic:
//   True when rank itself can vary.
struct ShapeSpec {
    std::vector<int64_t> dims;
    bool rank_dynamic{false};
};

// Endpoint tensor IO schema entry.
//
// required:
//   False means the input or output is optional.
//
// allow_user_buffer:
//   For outputs, true means callers may provide output storage in
//   EndpointRequest::output_buffers.
struct TensorSpec {
    std::string name;
    // Human-readable description of the DLPack DLDataType expected here, such
    // as "float16", "float32", "int32", or "uint8". Actual endpoint tensors
    // carry the authoritative DLDataType in DLTensor::dtype.
    std::string dtype;
    // Human-readable description of the expected DLPack DLDevice, such as
    // "cuda" or "cpu". Actual endpoint tensors carry the authoritative device
    // in DLTensor::device.
    std::string device;
    ShapeSpec shape;
    bool required{true};
    bool allow_user_buffer{false};
    std::string description;
};

// Full endpoint description returned by Endpoint::describe().
struct EndpointDescription {
    std::string name;
    std::string summary;
    std::string details;
    std::vector<TensorSpec> inputs;
    std::vector<TensorSpec> outputs;
};

// DLPack tensor handle used by endpoint inputs and outputs.
//
// Borrowed input:
//   Tensor(DLTensor*) references caller-owned DLPack metadata and memory. The
//   caller must keep the DLTensor, its shape/strides arrays, and data alive for
//   the duration of the endpoint call.
//
// Runtime-owned output:
//   Tensor::take(DLManagedTensorVersioned*) takes ownership of a DLPack managed
//   tensor returned by the runtime. Copies of Tensor share that ownership.
//
// The final SDK header should include dlpack.h so users can inspect DLTensor
// fields directly.
class Tensor {
  public:
    Tensor() = default;
    explicit Tensor(DLTensor* borrowed_tensor) : tensor_(borrowed_tensor) {}

    static Tensor take(DLManagedTensorVersioned* managed_tensor);

    bool valid() const { return tensor_ != nullptr; }
    explicit operator bool() const { return valid(); }
    bool owns() const { return owner_ != nullptr; }

    DLTensor* get() { return tensor_; }
    const DLTensor* get() const { return tensor_; }
    DLTensor& dl_tensor();
    const DLTensor& dl_tensor() const;

  private:
    std::shared_ptr<void> owner_;
    DLTensor* tensor_{nullptr};
};

// Named DLPack tensor passed to or returned from an endpoint.
struct NamedTensor {
    std::string name;
    Tensor tensor;

    NamedTensor() = default;
    NamedTensor(std::string tensor_name, DLTensor* borrowed_tensor)
        : name(std::move(tensor_name)), tensor(borrowed_tensor) {}
    NamedTensor(std::string tensor_name, Tensor tensor_value)
        : name(std::move(tensor_name)), tensor(std::move(tensor_value)) {}
};

// Endpoint request.
//
// inputs:
//   Named values required by EndpointDescription::inputs.
//
// requested_outputs:
//   Optional list of output names to return. Empty means endpoint default.
//
// output_buffers:
//   Optional caller-provided output buffers. Use this for tensor outputs when
//   TensorSpec::allow_user_buffer is true.
struct EndpointRequest {
    std::vector<NamedTensor> inputs;
    std::vector<std::string> requested_outputs;
    std::vector<NamedTensor> output_buffers;
    RunOptions run;

    EndpointRequest() = default;
    explicit EndpointRequest(std::vector<NamedTensor> request_inputs)
        : inputs(std::move(request_inputs)) {}
};

// Endpoint result.
struct EndpointResult {
    std::vector<NamedTensor> outputs;

    const NamedTensor& output(const std::string& name) const;
};

// Discoverable plugin endpoint.
//
// Endpoints are the low-level escape hatch for driving declared tensor
// boundaries directly. They use named DLPack tensor inputs and outputs, with
// schema advertised by describe(). Higher-level text/image/audio APIs stay on
// Model.
class Endpoint {
  public:
    Endpoint();
    Endpoint(const Endpoint&) = delete;
    Endpoint& operator=(const Endpoint&) = delete;
    Endpoint(Endpoint&&) noexcept;
    Endpoint& operator=(Endpoint&&) noexcept;
    ~Endpoint();

    bool valid() const;
    explicit operator bool() const { return valid(); }

    EndpointDescription describe() const;
    EndpointResult run(const EndpointRequest& request) const;

  private:
    friend class Model;

    class Impl;
    explicit Endpoint(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

// Public model handle returned by load().
//
// Model exposes:
//   1. Common metadata.
//   2. Stable task-level run(request) overloads.
//   3. Plugin-provided endpoints for low-level or model-specific control.
//
// The request type selects the high-level task. Endpoint names select plugin
// functionality after the user discovers available endpoints.
class Model {
  public:
    Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    Model(Model&&) noexcept;
    Model& operator=(Model&&) noexcept;
    ~Model();

    bool valid() const;
    explicit operator bool() const { return valid(); }

    std::string info_json() const;
    std::vector<Capability> capabilities() const;
    bool supports(const std::string& capability) const;

    TextResult run(const TextGenerationRequest& request) const;
    std::vector<TextResult>
    run_batch(const std::vector<TextGenerationRequest>& requests) const;
    void stream(const TextGenerationRequest& request,
                const TextStreamCallback& callback) const;

    ImageResult run(const ImageGenerationRequest& request) const;
    std::vector<ImageResult>
    run_batch(const std::vector<ImageGenerationRequest>& requests) const;

    TranscriptionResult run(const TranscriptionRequest& request) const;
    std::vector<TranscriptionResult>
    run_batch(const std::vector<TranscriptionRequest>& requests) const;

    EmbeddingResult run(const EmbeddingRequest& request) const;
    std::vector<EmbeddingResult>
    run_batch(const std::vector<EmbeddingRequest>& requests) const;

    std::vector<std::string> list_endpoints() const;
    Endpoint endpoint(const std::string& name) const;

  private:
    friend Model load(const std::string& bundle_path, const LoadOptions& options);
    friend Model make_model_from_runtime(void* runtime); // Implementation helper.

    class Impl;
    explicit Model(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

// Load a bundle through the linked C++ SDK path.
Model load(const std::string& bundle_path, const LoadOptions& options = {});

} // namespace trtmc
