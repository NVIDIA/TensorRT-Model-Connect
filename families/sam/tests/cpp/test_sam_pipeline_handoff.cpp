/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// SamPipeline's handoff between its two engines, driven through
// segment_prompted with mock modules whose "device" buffers are host memory.
// The mask decoder reads the image encoder's output in place: each request's
// decoder must read exactly the embedding the encoder computed for that
// request, and nothing is downloaded or uploaded. A pair of engines that
// cannot share the buffer (a dynamic input, a shape or dtype mismatch, a
// decoder input the encoder does not produce) is refused at construction.

#include "families/sam/runtime/sam_pipeline.h"

#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int g_failures = 0;

void check(bool condition, const std::string& test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++g_failures;
    }
}

bool bit_identical(const std::vector<float>& actual, const std::vector<float>& expected) {
    return actual.size() == expected.size() &&
           (actual.empty() ||
            std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)) == 0);
}

constexpr int32_t kImageSize = 16;
constexpr int32_t kHidden = 4;
constexpr int32_t kMaskSide = kImageSize / 4;
constexpr int32_t kMaskOutputs = 4;
const std::vector<int64_t> kEmbeddingShape = {1, 2, 4, 4};
constexpr std::size_t kEmbeddingCount = 32;
const char* const kPixels = "pixel_values";
const char* const kEmbeddings = "image_embeddings";
const char* const kSparse = "sparse_prompt_embeddings";

// Every ITrtModule member the tests do not use; both overloads of
// bind_external reach bound().
class MockModule : public trtmc::ITrtModule {
  public:
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void bind_external(const std::string& name, void* ptr) override { bound(name, ptr); }
    void bind_external(const std::string& name, void* ptr, const std::vector<int64_t>&) override {
        bound(name, ptr);
    }
    int32_t input_rank(const std::string&) const override { return 4; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

  protected:
    virtual void bound(const std::string&, void*) {}
};

// The image encoder: one output, a deterministic function of the pixels it
// was given, in its own "device" buffer. Like an engine's asynchronous launch,
// the output is written only once the caller syncs.
class MockEncoder final : public MockModule {
  public:
    std::vector<int64_t> output_shape = kEmbeddingShape;
    trtmc::DType output_dtype = trtmc::DType::kFloat32;
    bool pixels_dynamic = false;
    bool exposes_device_buffer = true;
    int downloads = 0;
    std::vector<float> device = std::vector<float>(kEmbeddingCount, 0.0F);
    std::vector<std::vector<float>> computed; // each request's output, at its sync

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        forward_async(inputs);
        sync();
        ++downloads;
        staging_ = device;
        trtmc::Tensor tensor;
        tensor.data = staging_.data();
        tensor.shape = output_shape;
        tensor.dtype = output_dtype;
        return {{kEmbeddings, tensor}};
    }
    void forward_async(const trtmc::TensorMap& inputs) override {
        const auto& pixels = inputs.at(kPixels);
        const auto* data = static_cast<const float*>(pixels.data);
        pending_.assign(data, data + pixels.numel());
    }
    void sync() override {
        if (pending_.empty())
            return;
        for (std::size_t i = 0; i < device.size(); ++i) {
            device[i] =
                pending_[(i * 37U) % pending_.size()] * 0.5F + static_cast<float>(i) * 0.125F;
        }
        computed.push_back(device);
        pending_.clear();
    }
    std::vector<trtmc::TensorInfo> input_info() const override {
        return {{kPixels, {1, 3, kImageSize, kImageSize}, trtmc::DType::kFloat32, true}};
    }
    std::vector<trtmc::TensorInfo> output_info() const override {
        return {{kEmbeddings, output_shape, output_dtype, false}};
    }
    bool has_input(const std::string& name) const override { return name == kPixels; }
    bool has_output(const std::string& name) const override { return name == kEmbeddings; }
    trtmc::DType tensor_dtype(const std::string& name) const override {
        return name == kEmbeddings ? output_dtype : trtmc::DType::kFloat32;
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        if (name == kEmbeddings)
            return output_shape;
        return {1, 3, kImageSize, kImageSize};
    }
    void* device_ptr(const std::string& name) const override {
        if (name != kEmbeddings || !exposes_device_buffer)
            return nullptr;
        return const_cast<float*>(device.data());
    }
    bool input_is_dynamic(const std::string& name) const override {
        return name == kPixels && pixels_dynamic;
    }

  private:
    std::vector<float> pending_;
    std::vector<float> staging_;
};

// The mask decoder: its image_embeddings input reads whatever buffer it is
// bound to (its own zeros until then); it records what it read.
class MockDecoder final : public MockModule {
  public:
    std::vector<int64_t> embedding_shape = kEmbeddingShape;
    trtmc::DType embedding_dtype = trtmc::DType::kFloat32;
    bool embedding_dynamic = false;
    bool takes_dense_prompt = false;
    const float* bound_embeddings = nullptr;
    int uploads_of_embeddings = 0;
    std::vector<std::vector<float>> read; // the embedding each request read

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        if (inputs.count(kEmbeddings) != 0)
            ++uploads_of_embeddings;
        const float* source =
            bound_embeddings != nullptr ? bound_embeddings : own_embeddings_.data();
        read.emplace_back(source, source + kEmbeddingCount);
        const auto* sparse = static_cast<const float*>(inputs.at(kSparse).data);
        const auto mask_values = static_cast<std::size_t>(kMaskOutputs * kMaskSide * kMaskSide);
        masks_.resize(mask_values);
        for (std::size_t i = 0; i < mask_values; ++i)
            masks_[i] = source[i % kEmbeddingCount] - sparse[i % (2 * kHidden)];
        iou_.resize(kMaskOutputs);
        for (std::size_t i = 0; i < iou_.size(); ++i)
            iou_[i] = source[i] * 0.25F + sparse[i];
        trtmc::Tensor masks;
        masks.data = masks_.data();
        masks.shape = {1, kMaskOutputs, kMaskSide, kMaskSide};
        masks.dtype = trtmc::DType::kFloat32;
        trtmc::Tensor iou;
        iou.data = iou_.data();
        iou.shape = {1, kMaskOutputs};
        iou.dtype = trtmc::DType::kFloat32;
        return {{"masks", masks}, {"iou_scores", iou}};
    }
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    std::vector<trtmc::TensorInfo> input_info() const override {
        std::vector<trtmc::TensorInfo> inputs = {
            {kEmbeddings, embedding_shape, embedding_dtype, true},
            {kSparse, {2, kHidden}, trtmc::DType::kFloat32, true}};
        if (takes_dense_prompt)
            inputs.push_back(
                {"dense_prompt_embeddings", kEmbeddingShape, trtmc::DType::kFloat32, true});
        return inputs;
    }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string& name) const override {
        return name == kEmbeddings || name == kSparse ||
               (takes_dense_prompt && name == "dense_prompt_embeddings");
    }
    bool has_output(const std::string& name) const override {
        return name == "masks" || name == "iou_scores";
    }
    trtmc::DType tensor_dtype(const std::string& name) const override {
        return name == kEmbeddings ? embedding_dtype : trtmc::DType::kFloat32;
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        if (name == kEmbeddings)
            return embedding_shape;
        if (name == kSparse)
            return {2, kHidden};
        return kEmbeddingShape;
    }
    void* device_ptr(const std::string&) const override { return nullptr; }
    bool input_is_dynamic(const std::string& name) const override {
        return name == kEmbeddings && embedding_dynamic;
    }

  protected:
    void bound(const std::string& name, void* ptr) override {
        if (name == kEmbeddings)
            bound_embeddings = static_cast<const float*>(ptr);
    }

  private:
    std::vector<float> own_embeddings_ = std::vector<float>(kEmbeddingCount, 0.0F);
    std::vector<float> masks_;
    std::vector<float> iou_;
};

trtmc::SamConfig small_config() {
    trtmc::SamConfig config;
    config.image_size = kImageSize;
    config.image_embedding_size = 4;
    config.decoder_hidden_size = kHidden;
    config.num_mask_outputs = kMaskOutputs;
    config.num_multimask_outputs = 3;
    config.point_embed_fg = {0.5F, -0.25F, 0.125F, 1.0F};
    config.point_embed_bg = {-0.5F, 0.25F, -0.125F, -1.0F};
    config.not_a_point_embed = {0.0625F, 0.75F, -0.375F, 0.5F};
    config.shared_image_pe = {0.3F, -0.7F, 1.1F, 0.2F};
    return config;
}

// An HWC RGB image of seeded values.
std::vector<float> seeded_image(int32_t height, int32_t width, uint32_t seed) {
    std::vector<float> pixels(static_cast<std::size_t>(height) * static_cast<std::size_t>(width) *
                              3);
    for (auto& value : pixels) {
        seed = seed * 1664525U + 1013904223U;
        value = static_cast<float>(seed >> 8) / static_cast<float>(1U << 24);
    }
    return pixels;
}

void test_decoder_reads_each_requests_embedding_in_place() {
    auto encoder = std::make_unique<MockEncoder>();
    auto decoder = std::make_unique<MockDecoder>();
    const MockEncoder& enc = *encoder;
    const MockDecoder& dec = *decoder;
    trtmc::SamPipeline pipeline(std::move(encoder), std::move(decoder), small_config());
    check(dec.bound_embeddings == enc.device.data(),
          "handoff: the decoder input is bound to the encoder's buffer");

    // Three requests of different images: each decoder run must read the
    // embedding its own request's encoder run produced, not a stale one.
    const int32_t sizes[][2] = {{5, 7}, {9, 5}, {6, 6}};
    uint32_t seed = 11U;
    for (const auto& size : sizes) {
        const auto image = seeded_image(size[0], size[1], seed++);
        const auto result =
            pipeline.segment_prompted(image.data(), size[0], size[1], 0.4F, 0.6F, seed % 2 == 0);
        check(result.num_masks == 3 && !result.masks.empty(), "handoff: the request returns masks");
    }
    check(enc.computed.size() == 3 && dec.read.size() == 3, "handoff: one run per request");
    for (std::size_t i = 0; i < enc.computed.size() && i < dec.read.size(); ++i) {
        check(bit_identical(dec.read[i], enc.computed[i]),
              "handoff: request " + std::to_string(i) + " reads its own embedding");
    }
    check(enc.downloads == 0, "handoff: the embedding is never downloaded");
    check(dec.uploads_of_embeddings == 0, "handoff: the embedding is never uploaded");
}

void check_refused(const std::function<void(MockEncoder&, MockDecoder&)>& configure,
                   const std::string& needle, const std::string& test_name) {
    auto encoder = std::make_unique<MockEncoder>();
    auto decoder = std::make_unique<MockDecoder>();
    configure(*encoder, *decoder);
    try {
        trtmc::SamPipeline pipeline(std::move(encoder), std::move(decoder), small_config());
        check(false, test_name + ": constructed");
    } catch (const std::runtime_error& error) {
        check(std::string(error.what()).find(needle) != std::string::npos,
              test_name + ": names the problem (" + error.what() + ")");
    }
}

void test_engines_that_cannot_share_the_buffer_are_refused() {
    check_refused([](MockEncoder&, MockDecoder& d) { d.embedding_shape = {1, 2, 2, 8}; },
                  "shape or dtype", "shape mismatch");
    check_refused([](MockEncoder&, MockDecoder& d) { d.embedding_dtype = trtmc::DType::kFloat16; },
                  "shape or dtype", "dtype mismatch");
    check_refused([](MockEncoder&, MockDecoder& d) { d.embedding_dynamic = true; }, "is dynamic",
                  "dynamic decoder input");
    check_refused([](MockEncoder& e, MockDecoder&) { e.pixels_dynamic = true; },
                  "image_encoder input 'pixel_values' is dynamic", "dynamic encoder input");
    check_refused([](MockEncoder&, MockDecoder& d) { d.takes_dense_prompt = true; },
                  "'dense_prompt_embeddings' is not an image_encoder output",
                  "decoder input the encoder does not produce");
    check_refused([](MockEncoder& e, MockDecoder&) { e.exposes_device_buffer = false; },
                  "no image_encoder device buffer", "encoder without a device buffer");
}

} // namespace

int main() {
    test_decoder_reads_each_requests_embedding_in_place();
    test_engines_that_cannot_share_the_buffer_are_refused();

    if (g_failures != 0) {
        std::cerr << g_failures << " SAM pipeline handoff test(s) failed" << '\n';
        return 1;
    }

    std::cout << "SAM pipeline handoff tests passed" << '\n';
    return 0;
}
