// EncoderPipeline host-side padding tests.
//
// Intent: verify static encoder inputs are padded/truncated before forward().
// Preconditions: no TensorRT SDK required.
// Postconditions: input_ids use the configured pad token and masks mark only
//                 real tokens.

#include "runtime/pipelines/encoder_pipeline.h"

#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

class RecordingModule final : public trtmc::ITrtModule {
  public:
    explicit RecordingModule(trtmc::DType mask_dtype) : mask_dtype_(mask_dtype) {}

    trtmc::TensorMap forward(const trtmc::TensorMap& inputs) override {
        record_ids(inputs.at("input_ids"));
        record_mask(inputs.at("attention_mask"));

        trtmc::Tensor hidden;
        hidden.data = output_.data();
        hidden.shape = {4, 2};
        hidden.dtype = trtmc::DType::kFloat32;
        return {{"hidden_states", hidden}};
    }

    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
    int32_t profile_idx() const override { return 0; }
    std::vector<trtmc::TensorInfo> input_info() const override {
        return {
            {"input_ids", {4}, trtmc::DType::kInt32, true},
            {"attention_mask", {4}, mask_dtype_, true},
        };
    }
    std::vector<trtmc::TensorInfo> output_info() const override {
        return {{"hidden_states", {4, 2}, trtmc::DType::kFloat32, false}};
    }
    bool has_input(const std::string& name) const override {
        return name == "input_ids" || name == "attention_mask";
    }
    bool has_output(const std::string& name) const override { return name == "hidden_states"; }
    trtmc::DType tensor_dtype(const std::string& name) const override {
        return name == "attention_mask" ? mask_dtype_ : trtmc::DType::kInt32;
    }
    std::vector<int64_t> tensor_shape(const std::string& name) const override {
        return name == "hidden_states" ? std::vector<int64_t>{4, 2} : std::vector<int64_t>{4};
    }
    std::vector<int64_t> input_profile_shape(const std::string&, int32_t,
                                             trtmc::ProfileShapeSelector) const override {
        return {4};
    }
    int32_t optimization_profile_count() const override { return 1; }
    void* device_ptr(const std::string&) const override { return nullptr; }
    void bind_external(const std::string&, void*) override {}
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}

    std::vector<int32_t> seen_ids;
    std::vector<int32_t> seen_mask_i32;
    std::vector<float> seen_mask_f32;

  private:
    void record_ids(const trtmc::Tensor& tensor) {
        const auto* ptr = static_cast<const int32_t*>(tensor.data);
        seen_ids.assign(ptr, ptr + tensor.numel());
    }

    void record_mask(const trtmc::Tensor& tensor) {
        if (tensor.dtype == trtmc::DType::kInt32) {
            const auto* ptr = static_cast<const int32_t*>(tensor.data);
            seen_mask_i32.assign(ptr, ptr + tensor.numel());
            return;
        }
        const auto* ptr = static_cast<const float*>(tensor.data);
        seen_mask_f32.assign(ptr, ptr + tensor.numel());
    }

    trtmc::DType mask_dtype_;
    std::vector<float> output_{0.1F, 0.2F, 0.3F, 0.4F, 0.5F, 0.6F, 0.7F, 0.8F};
};

void test_encoder_padding_uses_configured_pad_token() {
    auto module = std::make_unique<RecordingModule>(trtmc::DType::kInt32);
    auto* raw = module.get();
    trtmc::EncoderPipeline pipeline(std::move(module), "encoder_only", nullptr, "fnet", 3);

    auto result = pipeline.encode_ids({4, 14275});

    check(result.dim == 8, "padding: output copied");
    check((raw->seen_ids == std::vector<int32_t>{4, 14275, 3, 3}), "padding: ids use pad token");
    check((raw->seen_mask_i32 == std::vector<int32_t>{1, 1, 0, 0}),
          "padding: int32 mask marks real tokens");
}

void test_encoder_padding_uses_float_mask_when_engine_expects_float() {
    auto module = std::make_unique<RecordingModule>(trtmc::DType::kFloat32);
    auto* raw = module.get();
    trtmc::EncoderPipeline pipeline(std::move(module), "encoder_only", nullptr, "fnet", 3);

    pipeline.encode_ids({4});

    check((raw->seen_ids == std::vector<int32_t>{4, 3, 3, 3}), "float mask: ids use pad token");
    check((raw->seen_mask_f32 == std::vector<float>{1.0F, 0.0F, 0.0F, 0.0F}),
          "float mask: mask marks real tokens");
}

void test_encoder_input_truncates_to_engine_length() {
    auto module = std::make_unique<RecordingModule>(trtmc::DType::kInt32);
    auto* raw = module.get();
    trtmc::EncoderPipeline pipeline(std::move(module), "encoder_only", nullptr, "fnet", 3);

    pipeline.encode_ids({1, 2, 3, 4, 5});

    check((raw->seen_ids == std::vector<int32_t>{1, 2, 3, 4}), "truncate: ids fit engine length");
    check((raw->seen_mask_i32 == std::vector<int32_t>{1, 1, 1, 1}),
          "truncate: mask stays full for real tokens");
}

} // namespace

int main() {
    test_encoder_padding_uses_configured_pad_token();
    test_encoder_padding_uses_float_mask_when_engine_expects_float();
    test_encoder_input_truncates_to_engine_length();

    if (failures > 0) {
        std::cerr << failures << " failures\n";
        return 1;
    }
    std::cerr << "All EncoderPipeline padding tests passed\n";
    return 0;
}
