#include "runtime/models/patchtsmixer/pipeline.h"

#include "runtime/plugins/shared/plugin_helpers.h"
#include "utils/json_helpers.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace trtmc {
namespace {

std::string normalize_task_kind(const std::string& raw) {
    std::string task = raw;
    std::transform(task.begin(), task.end(), task.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (task.find("regress") != std::string::npos)
        return "regression";
    if (task.find("class") != std::string::npos)
        return "classification";
    if (task.find("pretrain") != std::string::npos)
        return "pretraining";
    if (task.find("forecast") != std::string::npos || task.find("predict") != std::string::npos)
        return "prediction";
    return task;
}

std::vector<float> align_window(const float* src, int32_t src_len, int32_t expected_len,
                                float fill_value) {
    std::vector<float> out(static_cast<std::size_t>(expected_len), fill_value);
    if (!src || src_len <= 0 || expected_len <= 0)
        return out;

    const int32_t copy_len = std::min(src_len, expected_len);
    const int32_t src_offset = src_len - copy_len;
    const int32_t dst_offset = expected_len - copy_len;
    std::memcpy(out.data() + dst_offset, src + src_offset,
                static_cast<std::size_t>(copy_len) * sizeof(float));
    return out;
}

std::vector<float> align_optional_window(const float* src, int32_t src_len, int32_t expected_len,
                                         float fill_value) {
    if (!src || src_len <= 0)
        return align_window(nullptr, 0, expected_len, fill_value);
    return align_window(src, src_len, expected_len, fill_value);
}

const Tensor* select_output_tensor(const TensorMap& outputs) {
    const Tensor* fallback = nullptr;
    for (const auto& [name, tensor] : outputs) {
        if (!fallback)
            fallback = &tensor;
        if (name.find("prediction") != std::string::npos ||
            name.find("regression") != std::string::npos ||
            name.find("classification") != std::string::npos ||
            name.find("score") != std::string::npos || name.find("logits") != std::string::npos) {
            return &tensor;
        }
    }
    return fallback;
}

std::string mask_input_name(const TrtModule& model) {
    if (model.has_input("observed_mask"))
        return "observed_mask";
    if (model.has_input("past_observed_mask"))
        return "past_observed_mask";
    return "";
}

int32_t extract_positive_json_int(const std::string& config_json,
                                  std::initializer_list<const char*> keys, int32_t fallback) {
    for (const char* key : keys) {
        const int32_t value = extract_json_int(config_json, key, 0);
        if (value > 0)
            return value;
    }
    return fallback;
}

std::string extract_task_kind(const std::string& config_json) {
    std::string task_kind = extract_json_string(config_json, "task_type", "");
    if (task_kind.empty()) {
        auto architectures = extract_json_string_array(config_json, "architectures");
        if (!architectures.empty())
            task_kind = architectures.front();
    }
    if (task_kind.empty())
        task_kind = "prediction";
    return normalize_task_kind(task_kind);
}

Tensor make_feature_tensor(float* data, int32_t context_length, int32_t num_input_channels) {
    Tensor tensor;
    tensor.data = data;
    tensor.shape = {1, context_length, num_input_channels};
    tensor.dtype = DType::kFloat32;
    return tensor;
}

EmbeddingResult tensor_to_embedding_result(const Tensor& tensor) {
    EmbeddingResult result;
    const auto n = tensor.numel();
    result.data.resize(n);
    std::memcpy(result.data.data(), tensor.data, n * sizeof(float));
    result.dim = static_cast<int32_t>(n);
    return result;
}

void add_mask_input_if_present(TensorMap& inputs, const TrtModule& model,
                               std::vector<float>& mask_host, int32_t context_length,
                               int32_t num_input_channels) {
    const std::string mask_name = mask_input_name(model);
    if (mask_name.empty())
        return;
    inputs[mask_name] = make_feature_tensor(mask_host.data(), context_length, num_input_channels);
}

} // namespace

PatchTSMixerConfig parse_patchtsmixer_config(const std::string& config_json,
                                             int32_t fallback_context_length) {
    PatchTSMixerConfig cfg;
    cfg.context_length = extract_json_int(config_json, "context_length", fallback_context_length);
    if (cfg.context_length <= 0)
        cfg.context_length = fallback_context_length > 0 ? fallback_context_length : 1;

    cfg.num_input_channels = extract_positive_json_int(
        config_json, {"num_input_channels", "input_size", "feature_size"}, 1);

    cfg.prediction_length = extract_json_int(config_json, "prediction_length", 1);
    if (cfg.prediction_length <= 0)
        cfg.prediction_length = 1;

    cfg.num_targets = extract_positive_json_int(config_json, {"num_targets", "num_labels"},
                                                cfg.prediction_length);
    cfg.task_kind = extract_task_kind(config_json);
    return cfg;
}

PatchTSMixerPipeline::PatchTSMixerPipeline(std::unique_ptr<TrtModule> model,
                                           PatchTSMixerConfig config, std::string model_id_str)
    : model_(std::move(model)), config_(std::move(config)), model_id_(std::move(model_id_str)) {
    if (!model_ || !model_->ok())
        throw std::runtime_error("PatchTSMixerPipeline: invalid model module");
}

EmbeddingResult PatchTSMixerPipeline::solve(const float* branch_input, int32_t branch_len,
                                            const float* trunk_input, int32_t trunk_len) {
    if (!branch_input || branch_len <= 0)
        throw std::runtime_error("PatchTSMixerPipeline::solve requires branch_input");

    const int32_t expected_len = std::max<int32_t>(
        1, config_.context_length * std::max<int32_t>(1, config_.num_input_channels));

    // Treat branch_input as the flattened past_values window.
    // If trunk_input is provided, treat it as an optional observed_mask window.
    // This keeps the numeric solve() API usable without introducing a second
    // dedicated mask argument for this model family.
    auto past_host = align_window(branch_input, branch_len, expected_len, 0.0f);
    auto mask_host = align_optional_window(trunk_input, trunk_len, expected_len, 1.0f);

    TensorMap inputs;
    inputs["past_values"] =
        make_feature_tensor(past_host.data(), config_.context_length, config_.num_input_channels);
    add_mask_input_if_present(inputs, *model_, mask_host, config_.context_length,
                              config_.num_input_channels);

    auto outputs = model_->forward(inputs);
    const Tensor* selected = select_output_tensor(outputs);
    if (!selected || !selected->data || selected->numel() == 0)
        throw std::runtime_error("PatchTSMixerPipeline: no output tensor produced");
    return tensor_to_embedding_result(*selected);
}

} // namespace trtmc
