/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/esmc/runtime/pipeline.h"

#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

class Encoder final : public trtmc::ITrtModule {
  public:
    int calls{0};
    std::vector<float> logits = std::vector<float>(8 * 64, 0.0F);
    std::vector<std::int32_t> received;

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        ++calls;
        const auto& ids = inputs.at("input_ids");
        const auto* data = static_cast<const std::int32_t*>(ids.data);
        received.assign(data, data + ids.numel());
        trtmc::Tensor result;
        result.data = logits.data();
        result.shape = {8, 64};
        result.dtype = trtmc::DType::kFloat32;
        return {{"logits", result}};
    }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    bool cuda_graph_captured() const override { return false; }
    std::int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override { return {}; }
    std::vector<trtmc::TensorInfo> output_info() const override { return {}; }
    bool has_input(const std::string&) const override { return true; }
    bool has_output(const std::string&) const override { return true; }
    trtmc::DType tensor_dtype(const std::string&) const override { return trtmc::DType::kFloat32; }
    std::vector<std::int64_t> tensor_shape(const std::string&) const override { return {}; }
    std::vector<std::int64_t> input_profile_shape(const std::string&, std::int32_t,
                                                  trtmc::ProfileShapeSelector) const override {
        return {};
    }
    std::int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    void bind_external(const std::string&, void*, const std::vector<std::int64_t>&) override {}
    std::int32_t input_rank(const std::string&) const override { return 1; }
    bool input_is_dynamic(const std::string&) const override { return false; }
    void reset_execution_context() override {}
    void set_timing_label(std::string) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}
};

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

} // namespace

int main() {
    try {
        std::string vocabulary = "<cls>\n<pad>\n<eos>\n<unk>\n";
        for (int index = 4; index < 64; ++index)
            vocabulary += index == 32 ? "<mask>\n" : "<token_" + std::to_string(index) + ">\n";
        auto tokenizer = trtmc::CreateEsmcTokenizer(vocabulary.data(), vocabulary.size());
        require(static_cast<bool>(tokenizer), "tokenizer must load");
        auto encoder = std::make_unique<Encoder>();
        auto* observed = encoder.get();
        trtmc::EsmcRuntimeConfig config;
        config.max_sequence_length = 8;
        trtmc::EsmcPipeline pipeline(std::move(encoder), std::move(tokenizer), config,
                                     "masked_text_to_token_scores");

        const std::vector<std::vector<std::int32_t>> invalid = {
            {},
            {0, -1, 2},
            {0, 64, 2},
            {0, std::numeric_limits<std::int32_t>::min(), 2},
            {0, std::numeric_limits<std::int32_t>::max(), 2},
            {0, 4, 4, 4, 4, 4, 4, 4, 64, 2},
        };
        for (const auto& ids : invalid) {
            bool rejected = false;
            try {
                pipeline.run({trtmc::Span<const std::int32_t>(ids.data(), ids.size())}, {});
            } catch (const std::invalid_argument&) {
                rejected = true;
            }
            require(rejected, "invalid token ids must be rejected");
            require(observed->calls == 0, "invalid token ids must not reach the encoder");
        }

        const std::vector<std::int32_t> ids = {0, 32, 63, 2};
        const auto result =
            pipeline.run({trtmc::Span<const std::int32_t>(ids.data(), ids.size())}, {});
        require(observed->calls == 1, "valid token ids must reach the encoder");
        require(observed->received == std::vector<std::int32_t>({0, 32, 63, 2, 1, 1, 1, 1}),
                "valid ids including the vocabulary boundary must be preserved");
        require(result.logits.rows == 1 && result.logits.columns == 64,
                "masked logits must retain their shape");
        require(result.positions.size() == 1 && result.positions[0].token_index == 1,
                "the masked position must be preserved");
        std::cout << "pipeline: all checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
