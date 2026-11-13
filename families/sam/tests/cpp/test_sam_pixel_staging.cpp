/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// SamPipeline's staging of the encoder's pixel_values in page-locked memory.
// With a CUDA device, every request uploads from the pipeline's one
// page-locked buffer, holding exactly what write_sam_pixel_values writes into
// a fresh buffer (the seam tests hold that to the per-sample definition). The
// pipeline needs page-locked memory: without a CUDA device it is refused at
// construction, naming the allocation.

#include "families/sam/runtime/sam_pipeline.h"
#include "families/sam/runtime/sam_preprocess.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <cuda_runtime_api.h>
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

trtmc::SamConfig small_config() {
    trtmc::SamConfig config;
    config.image_size = 32;
    config.image_embedding_size = 8;
    config.decoder_hidden_size = 4;
    config.num_mask_outputs = 4;
    config.num_multimask_outputs = 3;
    config.image_mean = {0.4F, 0.5F, 0.6F};
    config.image_std = {0.2F, 0.25F, 0.3F};
    config.point_embed_fg = {0.5F, -0.25F, 0.125F, 1.0F};
    config.point_embed_bg = {-0.5F, 0.25F, -0.125F, -1.0F};
    config.not_a_point_embed = {0.0625F, 0.75F, -0.375F, 0.5F};
    config.shared_image_pe = {0.3F, -0.7F, 1.1F, 0.2F};
    return config;
}

// Every ITrtModule member the test does not use.
class MockModule : public trtmc::ITrtModule {
  public:
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string&) const override { return false; }
    bool has_output(const std::string&) const override { return false; }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<int64_t> tensor_shape(const std::string&) const override { return {}; }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<int64_t>&) override {}
    int32_t input_rank(const std::string&) const override { return 4; }
    bool input_is_dynamic(const std::string&) const override { return false; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}
};

// Records the pixel_values each request uploads and where they came from.
class RecordingEncoder final : public MockModule {
  public:
    std::vector<std::vector<float>> received;
    std::vector<const void*> sources;

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        const auto& pixels = inputs.at("pixel_values");
        const auto* data = static_cast<const float*>(pixels.data);
        received.emplace_back(data, data + pixels.numel());
        sources.push_back(pixels.data);
        trtmc::Tensor embedding;
        embedding.data = embedding_.data();
        embedding.shape = {1, 4};
        embedding.dtype = trtmc::DType::kFloat32;
        return {{"image_embeddings", embedding}};
    }

  private:
    std::vector<float> embedding_ = std::vector<float>(4, 0.5F);
};

class FixedDecoder final : public MockModule {
  public:
    trtmc::TensorMap forward(const trtmc::TensorMap&) override {
        trtmc::Tensor masks;
        masks.data = masks_.data();
        masks.shape = {1, 4, 8, 8};
        masks.dtype = trtmc::DType::kFloat32;
        trtmc::Tensor iou;
        iou.data = iou_.data();
        iou.shape = {1, 4};
        iou.dtype = trtmc::DType::kFloat32;
        return {{"masks", masks}, {"iou_scores", iou}};
    }

  private:
    std::vector<float> masks_ = std::vector<float>(4 * 8 * 8, 1.0F);
    std::vector<float> iou_ = std::vector<float>(4, 0.75F);
};

bool cuda_device_present() {
    int count = 0;
    const bool present = cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    (void)cudaGetLastError();
    return present;
}

bool page_locked(const void* pointer) {
    cudaPointerAttributes attributes{};
    const bool host = cudaPointerGetAttributes(&attributes, pointer) == cudaSuccess &&
                      attributes.type == cudaMemoryTypeHost;
    (void)cudaGetLastError();
    return host;
}

void test_without_a_device_the_pipeline_is_refused() {
    try {
        trtmc::SamPipeline pipeline(std::make_unique<RecordingEncoder>(),
                                    std::make_unique<FixedDecoder>(), small_config());
        check(false, "no device: the pipeline was constructed");
    } catch (const std::runtime_error& error) {
        check(std::string(error.what()).find("page-locked pixel_values allocation") !=
                  std::string::npos,
              std::string("no device: names the allocation (") + error.what() + ")");
    }
}

// Requests in an order that leaves stale values where the next one pads.
void test_requests_upload_from_one_page_locked_buffer() {
    const auto config = small_config();
    auto encoder = std::make_unique<RecordingEncoder>();
    const RecordingEncoder& recorded = *encoder;
    trtmc::SamPipeline pipeline(std::move(encoder), std::make_unique<FixedDecoder>(), config);

    const std::vector<std::array<int32_t, 2>> sizes = {
        {19, 32}, {32, 19}, {1, 7}, {40, 33}, {5, 3}};
    uint32_t seed = 101U;
    for (const auto& [height, width] : sizes) {
        const auto image = seeded_image(height, width, seed++);
        (void)pipeline.segment_prompted(image.data(), height, width, 0.5F, 0.5F, true);
        const auto plan = trtmc::plan_sam_preprocess(image.data(), height, width, config);
        std::vector<float> expected(3 * static_cast<std::size_t>(config.image_size) *
                                    static_cast<std::size_t>(config.image_size));
        trtmc::write_sam_pixel_values(expected.data(), plan, image.data(), config);
        const auto& actual = recorded.received.back();
        check(actual.size() == expected.size() &&
                  std::memcmp(actual.data(), expected.data(), expected.size() * sizeof(float)) == 0,
              "upload: " + std::to_string(height) + "x" + std::to_string(width));
    }
    check(recorded.received.size() == sizes.size(), "one upload per request");
    bool one_page_locked_buffer = !recorded.sources.empty();
    for (const void* source : recorded.sources)
        one_page_locked_buffer =
            one_page_locked_buffer && source == recorded.sources[0] && page_locked(source);
    check(one_page_locked_buffer, "every request uploads from one page-locked buffer");
}

} // namespace

int main() {
    if (cuda_device_present()) {
        test_requests_upload_from_one_page_locked_buffer();
    } else {
        std::cout << "no CUDA device: checking the refusal only" << '\n';
        test_without_a_device_the_pipeline_is_refused();
    }

    if (g_failures != 0) {
        std::cerr << g_failures << " SAM pixel staging test(s) failed" << '\n';
        return 1;
    }

    std::cout << "SAM pixel staging tests passed" << '\n';
    return 0;
}
