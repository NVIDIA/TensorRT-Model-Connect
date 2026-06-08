#include "runtime/models/voxcpm2/pipeline.h"

#include "runtime/domains/audio/voxcpm2_config.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace trtmc {

namespace audio = runtime::builders::audio;

namespace {

struct OwnedStageTensor {
    std::vector<unsigned char> storage;
    std::vector<int64_t> shape;
    DType dtype{DType::kFloat32};

    Tensor as_tensor() const {
        Tensor tensor;
        tensor.data = storage.empty() ? nullptr : const_cast<unsigned char*>(storage.data());
        tensor.shape = shape;
        tensor.dtype = dtype;
        return tensor;
    }
};

using StageArtifacts = std::unordered_map<std::string, OwnedStageTensor>;
using half_bits_t = uint16_t;

struct KvCacheBinding {
    const char* past;
    const char* present;
};

constexpr KvCacheBinding kTslmKvCache{"tslm_past_kv_cache", "tslm_present_kv_cache"};
constexpr KvCacheBinding kRalmKvCache{"ralm_past_kv_cache", "ralm_present_kv_cache"};
constexpr std::size_t kVoxCPM2MinGenerationSteps = 2;

const char* dtype_name(DType dtype) {
    switch (dtype) {
    case DType::kFloat32:
        return "float32";
    case DType::kFloat16:
        return "float16";
    case DType::kBFloat16:
        return "bfloat16";
    case DType::kInt32:
        return "int32";
    case DType::kInt8:
        return "int8";
    }
    return "unknown";
}

std::string describe_tensor_contract(const audio::VoxCPM2TensorContract& contract) {
    std::ostringstream os;
    os << contract.name << ":" << audio::voxcpm2_dtype_contract_name(contract.dtype_contract) << "["
       << contract.symbolic_shape << "]";
    return os.str();
}

void validate_tensor_contract(const Tensor& tensor, const audio::VoxCPM2TensorContract& contract,
                              const audio::VoxCPM2GenerationStage& stage,
                              const std::string& component_name, const char* direction) {
    if (!audio::voxcpm2_dtype_matches(contract.dtype_contract, tensor.dtype)) {
        throw std::runtime_error("VoxCPM2Pipeline: stage " + component_name + " (" +
                                 stage.engine_section + ") " + direction + " artifact '" +
                                 contract.name + "' has dtype " + dtype_name(tensor.dtype) +
                                 ", expected " + describe_tensor_contract(contract));
    }
    if (tensor.shape.size() != contract.rank) {
        throw std::runtime_error(
            "VoxCPM2Pipeline: stage " + component_name + " (" + stage.engine_section + ") " +
            direction + " artifact '" + contract.name + "' has rank " +
            std::to_string(tensor.shape.size()) + ", expected " + std::to_string(contract.rank) +
            " for " + describe_tensor_contract(contract));
    }
}

OwnedStageTensor copy_stage_tensor(const Tensor& tensor, const std::string& artifact_name,
                                   const std::string& component_name) {
    if (tensor.data == nullptr && tensor.nbytes() > 0) {
        throw std::runtime_error("VoxCPM2Pipeline: stage " + component_name +
                                 " returned null data for output artifact '" + artifact_name + "'");
    }

    OwnedStageTensor owned;
    owned.shape = tensor.shape;
    owned.dtype = tensor.dtype;
    owned.storage.resize(tensor.nbytes());
    if (!owned.storage.empty())
        std::memcpy(owned.storage.data(), tensor.data, owned.storage.size());
    return owned;
}

OwnedStageTensor make_prompt_tensor(const std::string& prompt) {
    OwnedStageTensor tensor;
    tensor.shape = {static_cast<int64_t>(prompt.size())};
    tensor.dtype = DType::kInt8;
    tensor.storage.resize(prompt.size());
    if (!prompt.empty()) {
        std::memcpy(tensor.storage.data(), prompt.data(), prompt.size());
    }
    return tensor;
}

struct PreparedTextTokens {
    OwnedStageTensor tensor;
    std::size_t actual_count{0};
};

bool read_utf8_codepoint(const std::string& text, std::size_t& offset, uint32_t& codepoint,
                         std::string& bytes) {
    if (offset >= text.size())
        return false;

    const auto first = static_cast<unsigned char>(text[offset]);
    std::size_t length = 1;
    if ((first & 0x80U) == 0U) {
        codepoint = first;
    } else if ((first & 0xE0U) == 0xC0U) {
        codepoint = first & 0x1FU;
        length = 2;
    } else if ((first & 0xF0U) == 0xE0U) {
        codepoint = first & 0x0FU;
        length = 3;
    } else if ((first & 0xF8U) == 0xF0U) {
        codepoint = first & 0x07U;
        length = 4;
    } else {
        codepoint = first;
    }

    if (offset + length > text.size()) {
        length = 1;
        codepoint = first;
    }

    for (std::size_t i = 1; i < length; ++i) {
        const auto next = static_cast<unsigned char>(text[offset + i]);
        if ((next & 0xC0U) != 0x80U) {
            length = 1;
            codepoint = first;
            break;
        }
        codepoint = (codepoint << 6U) | (next & 0x3FU);
    }

    bytes.assign(text.data() + offset, length);
    offset += length;
    return true;
}

bool is_cjk_codepoint(uint32_t codepoint) {
    return (codepoint >= 0x4E00U && codepoint <= 0x9FFFU) ||
           (codepoint >= 0x3400U && codepoint <= 0x4DBFU) ||
           (codepoint >= 0xF900U && codepoint <= 0xFAFFU) ||
           (codepoint >= 0x20000U && codepoint <= 0x2A6DFU);
}

std::vector<std::string> voxcpm2_cjk_expansion_chars(const std::string& token) {
    constexpr uint32_t kSentencePieceUnderline = 0x2581U;
    std::vector<std::string> chars;
    std::size_t offset = 0;
    while (offset < token.size()) {
        uint32_t codepoint = 0;
        std::string bytes;
        if (!read_utf8_codepoint(token, offset, codepoint, bytes))
            break;
        if (codepoint == kSentencePieceUnderline)
            continue;
        if (!is_cjk_codepoint(codepoint))
            return {};
        chars.push_back(std::move(bytes));
    }
    if (chars.size() < 2)
        return {};
    return chars;
}

std::vector<int32_t> expand_voxcpm2_multichar_cjk_tokens(const std::vector<int32_t>& ids,
                                                         const ITokenizer& tokenizer) {
    std::vector<int32_t> expanded;
    expanded.reserve(ids.size());
    for (const int32_t id : ids) {
        const auto token = tokenizer.token_for_id(id);
        const auto chars = voxcpm2_cjk_expansion_chars(token);
        if (chars.empty()) {
            expanded.push_back(id);
            continue;
        }

        std::vector<int32_t> char_ids;
        char_ids.reserve(chars.size());
        bool resolved = true;
        for (const auto& ch : chars) {
            const int32_t char_id = tokenizer.id_for_token(ch);
            if (char_id < 0) {
                resolved = false;
                break;
            }
            char_ids.push_back(char_id);
        }
        if (!resolved) {
            expanded.push_back(id);
            continue;
        }
        expanded.insert(expanded.end(), char_ids.begin(), char_ids.end());
    }
    return expanded;
}

int32_t resolve_voxcpm2_audio_start_token(const ITokenizer& tokenizer) {
    constexpr int32_t kDefaultAudioStartToken = 101;
    const int32_t token_id = tokenizer.id_for_token("<|audio_start|>");
    return token_id >= 0 ? token_id : kDefaultAudioStartToken;
}

PreparedTextTokens make_text_tokens_tensor(const std::string& prompt, const ITokenizer* tokenizer,
                                           int32_t max_text_steps) {
    if (tokenizer == nullptr) {
        throw std::runtime_error(
            "VoxCPM2Pipeline: tokenizer.json is required to build text_tokens for TSLM");
    }
    auto ids = expand_voxcpm2_multichar_cjk_tokens(tokenizer->encode(prompt), *tokenizer);
    ids.push_back(resolve_voxcpm2_audio_start_token(*tokenizer));
    if (ids.empty()) {
        throw std::runtime_error("VoxCPM2Pipeline: tokenizer returned no text tokens");
    }

    const auto requested_steps = static_cast<std::size_t>(std::max(max_text_steps, 0));
    const auto padded_steps = requested_steps > 0 ? requested_steps : ids.size();
    if (ids.size() > padded_steps) {
        throw std::runtime_error(
            "VoxCPM2Pipeline: prompt token count " + std::to_string(ids.size()) +
            " exceeds engine text step capacity " + std::to_string(padded_steps));
    }

    std::vector<int32_t> padded_ids(padded_steps, 0);
    std::copy(ids.begin(), ids.end(), padded_ids.begin());

    PreparedTextTokens prepared;
    prepared.actual_count = ids.size();
    prepared.tensor.shape = {static_cast<int64_t>(padded_steps)};
    prepared.tensor.dtype = DType::kInt32;
    prepared.tensor.storage.resize(padded_ids.size() * sizeof(int32_t));
    std::memcpy(prepared.tensor.storage.data(), padded_ids.data(), prepared.tensor.storage.size());
    return prepared;
}

OwnedStageTensor make_float_mask_tensor(std::size_t token_count, std::size_t active_count,
                                        float active_value) {
    OwnedStageTensor tensor;
    tensor.shape = {static_cast<int64_t>(token_count)};
    tensor.dtype = DType::kFloat32;
    std::vector<float> values(token_count, 0.0F);
    std::fill(values.begin(),
              values.begin() + static_cast<std::ptrdiff_t>(std::min(token_count, active_count)),
              active_value);
    tensor.storage.resize(values.size() * sizeof(float));
    if (!values.empty()) {
        std::memcpy(tensor.storage.data(), values.data(), tensor.storage.size());
    }
    return tensor;
}

OwnedStageTensor make_zero_audio_feats_tensor(std::size_t text_steps, const VoxCPM2Config& cfg) {
    if (cfg.patch_size <= 0 || cfg.feat_dim <= 0) {
        throw std::runtime_error("VoxCPM2Pipeline: patch_size and feat_dim must be positive");
    }

    OwnedStageTensor tensor;
    const auto steps = static_cast<int64_t>(std::max<std::size_t>(1, text_steps));
    tensor.shape = {steps, cfg.patch_size, cfg.feat_dim};
    tensor.dtype = DType::kFloat32;
    const auto value_count =
        static_cast<std::size_t>(steps) * static_cast<std::size_t>(cfg.patch_size) *
        static_cast<std::size_t>(cfg.feat_dim);
    tensor.storage.resize(value_count * sizeof(float));
    return tensor;
}

OwnedStageTensor make_initial_feat_cond_tensor(const VoxCPM2Config& cfg) {
    if (cfg.patch_size <= 0 || cfg.feat_dim <= 0) {
        throw std::runtime_error("VoxCPM2Pipeline: patch_size and feat_dim must be positive");
    }

    OwnedStageTensor tensor;
    tensor.shape = {cfg.patch_size, cfg.feat_dim};
    tensor.dtype = DType::kFloat32;
    const auto value_count =
        static_cast<std::size_t>(cfg.patch_size) * static_cast<std::size_t>(cfg.feat_dim);
    tensor.storage.resize(value_count * sizeof(float));
    return tensor;
}

OwnedStageTensor make_int32_scalar_tensor(int32_t value) {
    OwnedStageTensor tensor;
    tensor.shape = {1};
    tensor.dtype = DType::kInt32;
    tensor.storage.resize(sizeof(int32_t));
    std::memcpy(tensor.storage.data(), &value, tensor.storage.size());
    return tensor;
}

OwnedStageTensor make_single_step_mask_tensor(float value) {
    OwnedStageTensor tensor;
    tensor.shape = {1};
    tensor.dtype = DType::kFloat32;
    tensor.storage.resize(sizeof(float));
    std::memcpy(tensor.storage.data(), &value, tensor.storage.size());
    return tensor;
}

OwnedStageTensor make_single_step_text_tokens_tensor() {
    return make_int32_scalar_tensor(0);
}

bool has_positive_shape(const std::vector<int64_t>& shape) {
    if (shape.empty())
        return false;
    return std::all_of(shape.begin(), shape.end(), [](int64_t dim) { return dim > 0; });
}

std::vector<int64_t> resolve_input_shape(const ITrtModule& module, const std::string& name) {
    auto shape = module.tensor_shape(name);
    if (has_positive_shape(shape))
        return shape;

    const int32_t profile_count = module.optimization_profile_count();
    if (profile_count > 0) {
        shape = module.input_profile_shape(name, module.profile_idx(), ProfileShapeSelector::kOpt);
        if (has_positive_shape(shape))
            return shape;
        shape = module.input_profile_shape(name, module.profile_idx(), ProfileShapeSelector::kMax);
        if (has_positive_shape(shape))
            return shape;
    }

    throw std::runtime_error("VoxCPM2Pipeline: cache input binding '" + name +
                             "' does not expose a concrete allocation shape");
}

OwnedStageTensor make_zero_input_tensor(const ITrtModule& module, const std::string& name) {
    OwnedStageTensor tensor;
    tensor.shape = resolve_input_shape(module, name);
    tensor.dtype = module.tensor_dtype(name);
    std::size_t element_count = 1;
    for (const auto dim : tensor.shape) {
        if (element_count > std::numeric_limits<std::size_t>::max() /
                                static_cast<std::size_t>(dim)) {
            throw std::runtime_error("VoxCPM2Pipeline: cache input binding '" + name +
                                     "' shape overflows byte-size calculation");
        }
        element_count *= static_cast<std::size_t>(dim);
    }
    tensor.storage.resize(element_count * dtype_size(tensor.dtype));
    return tensor;
}

std::size_t checked_first_dim_stride(const OwnedStageTensor& tensor,
                                     const std::string& artifact_name) {
    if (tensor.shape.empty()) {
        throw std::runtime_error("VoxCPM2Pipeline: artifact '" + artifact_name +
                                 "' must have at least one dimension");
    }
    for (const auto dim : tensor.shape) {
        if (dim < 0) {
            throw std::runtime_error("VoxCPM2Pipeline: artifact '" + artifact_name +
                                     "' has negative tensor dimension");
        }
    }
    std::size_t stride = 1;
    for (std::size_t i = 1; i < tensor.shape.size(); ++i) {
        const auto dim = static_cast<std::size_t>(tensor.shape[i]);
        if (dim != 0 &&
            stride > std::numeric_limits<std::size_t>::max() / dim) {
            throw std::runtime_error("VoxCPM2Pipeline: artifact '" + artifact_name +
                                     "' shape overflows byte-size calculation");
        }
        stride *= dim;
    }
    return stride;
}

OwnedStageTensor slice_first_dim(const OwnedStageTensor& tensor, std::size_t start,
                                 std::size_t count, const std::string& artifact_name) {
    const auto first_dim_stride = checked_first_dim_stride(tensor, artifact_name);
    const auto first_dim = static_cast<std::size_t>(tensor.shape.front());
    if (start > first_dim || count > first_dim - start) {
        throw std::runtime_error("VoxCPM2Pipeline: artifact '" + artifact_name +
                                 "' cannot slice first dimension at " +
                                 std::to_string(start) + " for " + std::to_string(count) +
                                 " row(s)");
    }
    const auto dtype_bytes = dtype_size(tensor.dtype);
    const auto row_bytes = first_dim_stride * dtype_bytes;
    const auto offset_bytes = start * row_bytes;
    const auto byte_count = count * row_bytes;
    if (offset_bytes > tensor.storage.size() || byte_count > tensor.storage.size() - offset_bytes) {
        throw std::runtime_error("VoxCPM2Pipeline: artifact '" + artifact_name +
                                 "' storage is too small for requested slice");
    }

    OwnedStageTensor out;
    out.shape = tensor.shape;
    out.shape[0] = static_cast<int64_t>(count);
    out.dtype = tensor.dtype;
    out.storage.resize(byte_count);
    if (!out.storage.empty()) {
        std::memcpy(out.storage.data(), tensor.storage.data() + offset_bytes, byte_count);
    }
    return out;
}

OwnedStageTensor make_audio_feats_from_patch(const OwnedStageTensor& patch,
                                             const VoxCPM2Config& cfg,
                                             const std::string& artifact_name) {
    if (patch.shape.size() != 2) {
        throw std::runtime_error("VoxCPM2Pipeline: generated latent patch '" + artifact_name +
                                 "' must be rank 2 before LocEnc refresh");
    }
    if (patch.shape[0] != cfg.patch_size || patch.shape[1] != cfg.feat_dim) {
        throw std::runtime_error("VoxCPM2Pipeline: generated latent patch '" + artifact_name +
                                 "' has shape [" + std::to_string(patch.shape[0]) + "," +
                                 std::to_string(patch.shape[1]) + "], expected [" +
                                 std::to_string(cfg.patch_size) + "," +
                                 std::to_string(cfg.feat_dim) + "]");
    }

    OwnedStageTensor audio_feats;
    audio_feats.shape = {1, patch.shape[0], patch.shape[1]};
    audio_feats.dtype = patch.dtype;
    audio_feats.storage = patch.storage;
    return audio_feats;
}

void append_first_dim(OwnedStageTensor& target, const OwnedStageTensor& chunk,
                      const std::string& artifact_name) {
    if (chunk.shape.empty()) {
        throw std::runtime_error("VoxCPM2Pipeline: cannot append rank-0 artifact '" +
                                 artifact_name + "'");
    }
    if (target.shape.empty()) {
        target = chunk;
        return;
    }
    if (target.dtype != chunk.dtype || target.shape.size() != chunk.shape.size()) {
        throw std::runtime_error("VoxCPM2Pipeline: latent patch for artifact '" + artifact_name +
                                 "' does not match accumulated tensor metadata");
    }
    for (std::size_t i = 1; i < target.shape.size(); ++i) {
        if (target.shape[i] != chunk.shape[i]) {
            throw std::runtime_error("VoxCPM2Pipeline: latent patch for artifact '" +
                                     artifact_name + "' has incompatible trailing shape");
        }
    }
    target.storage.insert(target.storage.end(), chunk.storage.begin(), chunk.storage.end());
    target.shape[0] += chunk.shape[0];
}

float fp16_to_fp32(half_bits_t h) {
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000U) << 16U;
    uint32_t exp = (h >> 10U) & 0x1FU;
    uint32_t mant = h & 0x3FFU;
    uint32_t bits = sign;
    if (exp == 31U) {
        bits |= 0x7F800000U | (mant << 13U);
    } else if (exp != 0U) {
        bits |= (static_cast<uint32_t>(exp - 15U + 127U) << 23U) | (mant << 13U);
    }
    float out = 0.0F;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

float bf16_to_fp32(half_bits_t h) {
    const uint32_t bits = static_cast<uint32_t>(h) << 16U;
    float out = 0.0F;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

std::size_t tensor_element_count(const OwnedStageTensor& tensor,
                                 const std::string& artifact_name) {
    const auto element_size = dtype_size(tensor.dtype);
    if (element_size == 0) {
        throw std::runtime_error("VoxCPM2Pipeline: artifact '" + artifact_name +
                                 "' has unsupported dtype");
    }
    if (tensor.storage.size() % element_size != 0) {
        throw std::runtime_error("VoxCPM2Pipeline: artifact '" + artifact_name +
                                 "' byte size is not element-aligned");
    }
    return tensor.storage.size() / element_size;
}

float tensor_float_value(const OwnedStageTensor& tensor, std::size_t index,
                         const std::string& artifact_name) {
    const auto count = tensor_element_count(tensor, artifact_name);
    if (index >= count) {
        throw std::runtime_error("VoxCPM2Pipeline: artifact '" + artifact_name +
                                 "' scalar index is out of range");
    }
    if (tensor.dtype == DType::kFloat32) {
        float value = 0.0F;
        std::memcpy(&value, tensor.storage.data() + index * sizeof(float), sizeof(value));
        return value;
    }
    if (tensor.dtype == DType::kFloat16 || tensor.dtype == DType::kBFloat16) {
        half_bits_t value = 0;
        std::memcpy(&value, tensor.storage.data() + index * sizeof(half_bits_t), sizeof(value));
        return tensor.dtype == DType::kFloat16 ? fp16_to_fp32(value) : bf16_to_fp32(value);
    }
    throw std::runtime_error("VoxCPM2Pipeline: artifact '" + artifact_name +
                             "' is not a floating-point tensor");
}

bool stop_logits_predict_stop(const StageArtifacts& artifacts) {
    const auto it = artifacts.find("stop_logits");
    if (it == artifacts.end())
        return false;
    const auto count = tensor_element_count(it->second, "stop_logits");
    if (count < 2) {
        throw std::runtime_error(
            "VoxCPM2Pipeline: stop_logits must contain at least two class logits");
    }
    const auto base = count - 2;
    return tensor_float_value(it->second, base + 1, "stop_logits") >
           tensor_float_value(it->second, base, "stop_logits");
}

std::size_t positive_first_dim(const std::vector<int64_t>& shape) {
    if (shape.empty() || shape.front() <= 0)
        return 0;
    return static_cast<std::size_t>(shape.front());
}

std::size_t audio_vae_profile_frame_count(const ITrtModule& module) {
    const auto profile_count = module.optimization_profile_count();
    if (profile_count > 0) {
        auto shape = module.input_profile_shape("audio_vae_latents", module.profile_idx(),
                                                ProfileShapeSelector::kMax);
        if (const auto frames = positive_first_dim(shape); frames > 0)
            return frames;
    }
    return positive_first_dim(module.tensor_shape("audio_vae_latents"));
}

OwnedStageTensor trim_audio_vae_waveform_to_latents(OwnedStageTensor waveform,
                                                    const OwnedStageTensor& latents,
                                                    const ITrtModule& audio_vae_module) {
    if (waveform.shape.size() != 1 || latents.shape.empty() || latents.shape.front() <= 0)
        return waveform;
    const auto latent_frames = static_cast<std::size_t>(latents.shape.front());
    const auto profile_frames = audio_vae_profile_frame_count(audio_vae_module);
    if (profile_frames == 0 || latent_frames >= profile_frames)
        return waveform;

    const auto sample_count = tensor_element_count(waveform, "waveform_f32");
    if (sample_count % profile_frames != 0)
        return waveform;
    const auto samples_per_latent_frame = sample_count / profile_frames;
    const auto target_samples = latent_frames * samples_per_latent_frame;
    if (target_samples == 0 || target_samples >= sample_count)
        return waveform;
    return slice_first_dim(waveform, 0, target_samples, "waveform_f32");
}

std::size_t resolve_latent_generation_steps(std::size_t active_text_token_count,
                                            const GenerateConfig& cfg,
                                            const VoxCPM2Config& voxcpm2_cfg) {
    constexpr std::size_t kUpstreamDefaultMaxLen = 2000;
    if (cfg.max_new_tokens > 0)
        return static_cast<std::size_t>(cfg.max_new_tokens);
    if (active_text_token_count == 0) {
        throw std::runtime_error("VoxCPM2Pipeline: resolved zero active text tokens");
    }
    if (voxcpm2_cfg.retry_badcase_ratio_threshold < 0.0F) {
        throw std::runtime_error(
            "VoxCPM2Pipeline: retry_badcase_ratio_threshold must be non-negative");
    }

    const auto default_steps = static_cast<std::size_t>(
        static_cast<double>(active_text_token_count) *
            static_cast<double>(voxcpm2_cfg.retry_badcase_ratio_threshold) +
        10.0);
    const auto steps = std::min(default_steps, kUpstreamDefaultMaxLen);
    if (steps == 0) {
        throw std::runtime_error("VoxCPM2Pipeline: resolved zero LocDiT generation steps");
    }
    return steps;
}

void validate_lm_state_ready(const StageArtifacts& artifacts) {
    const auto lm_it = artifacts.find("lm_hidden");
    if (lm_it == artifacts.end()) {
        throw std::runtime_error(
            "VoxCPM2Pipeline: LocDiT loop requires prepared lm_hidden artifact");
    }
    const auto residual_it = artifacts.find("residual_hidden");
    if (residual_it == artifacts.end()) {
        throw std::runtime_error(
            "VoxCPM2Pipeline: LocDiT loop requires prepared residual_hidden artifact");
    }
}

OwnedStageTensor extract_locdit_patch(const OwnedStageTensor& locdit_output, std::size_t step,
                                      const VoxCPM2Config& cfg,
                                      const std::string& artifact_name) {
    if (cfg.patch_size <= 0)
        throw std::runtime_error("VoxCPM2Pipeline: patch_size must be positive");
    if (locdit_output.shape.size() != 2) {
        throw std::runtime_error("VoxCPM2Pipeline: LocDiT output artifact '" + artifact_name +
                                 "' must be rank 2");
    }
    if (locdit_output.shape[1] != cfg.feat_dim) {
        throw std::runtime_error("VoxCPM2Pipeline: LocDiT output artifact '" + artifact_name +
                                 "' feature dimension " +
                                 std::to_string(locdit_output.shape[1]) + " does not match " +
                                 std::to_string(cfg.feat_dim));
    }
    const auto patch_rows = static_cast<std::size_t>(cfg.patch_size);
    const auto available_rows = static_cast<std::size_t>(locdit_output.shape[0]);
    const auto start = available_rows == patch_rows ? 0 : step * patch_rows;
    return slice_first_dim(locdit_output, start, patch_rows, artifact_name);
}

OwnedStageTensor latest_hidden_row(const OwnedStageTensor& tensor,
                                   std::size_t active_token_count,
                                   const std::string& artifact_name) {
    if (tensor.shape.empty()) {
        throw std::runtime_error("VoxCPM2Pipeline: artifact '" + artifact_name +
                                 "' must have a sequence dimension");
    }
    const auto rows = static_cast<std::size_t>(tensor.shape.front());
    if (rows == 0) {
        throw std::runtime_error("VoxCPM2Pipeline: artifact '" + artifact_name +
                                 "' has no hidden rows");
    }
    auto row = rows - 1;
    if (active_token_count > 0) {
        row = std::min(active_token_count - 1, rows - 1);
    }
    return slice_first_dim(tensor, row, 1, artifact_name);
}

void keep_latest_hidden_artifacts(StageArtifacts& artifacts, std::size_t active_token_count) {
    const auto lm_it = artifacts.find("lm_hidden");
    if (lm_it == artifacts.end()) {
        throw std::runtime_error(
            "VoxCPM2Pipeline: LocDiT loop requires prepared lm_hidden artifact");
    }
    const auto residual_it = artifacts.find("residual_hidden");
    if (residual_it == artifacts.end()) {
        throw std::runtime_error(
            "VoxCPM2Pipeline: LocDiT loop requires prepared residual_hidden artifact");
    }

    auto lm_hidden = latest_hidden_row(lm_it->second, active_token_count, "lm_hidden");
    auto residual_hidden =
        latest_hidden_row(residual_it->second, active_token_count, "residual_hidden");
    artifacts["lm_hidden"] = std::move(lm_hidden);
    artifacts["residual_hidden"] = std::move(residual_hidden);
}

int32_t checked_position_id(std::size_t active_text_token_count, std::size_t generated_step) {
    if (active_text_token_count >
        static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("VoxCPM2Pipeline: text token count exceeds int32 position range");
    }
    if (generated_step >
        static_cast<std::size_t>(std::numeric_limits<int32_t>::max()) - active_text_token_count) {
        throw std::runtime_error("VoxCPM2Pipeline: generated step exceeds int32 position range");
    }
    return static_cast<int32_t>(active_text_token_count + generated_step);
}

struct RuntimeScalarInputs {
    explicit RuntimeScalarInputs(const VoxCPM2Config& cfg)
        : sample_rate(cfg.sample_rate), reference_sample_rate(cfg.reference_sample_rate),
          cfg_value(cfg.cfg_value), inference_timesteps(cfg.inference_timesteps),
          normalize(cfg.normalize ? 1 : 0), denoise(cfg.denoise ? 1 : 0),
          retry_badcase(cfg.retry_badcase ? 1 : 0),
          retry_badcase_max_times(cfg.retry_badcase_max_times),
          retry_badcase_ratio_threshold(cfg.retry_badcase_ratio_threshold) {
        if (cfg.seed < static_cast<int64_t>(std::numeric_limits<int32_t>::min()) ||
            cfg.seed > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
            throw std::runtime_error("VoxCPM2Pipeline: seed is outside int32 runtime range");
        }
        seed = static_cast<int32_t>(cfg.seed);
    }

    void add_to(const ITrtModule& module, TensorMap& inputs) const {
        add_int32(module, inputs, "sample_rate", sample_rate);
        add_int32(module, inputs, "reference_sample_rate", reference_sample_rate);
        add_float32(module, inputs, "cfg_value", cfg_value);
        add_int32(module, inputs, "inference_timesteps", inference_timesteps);
        add_int32(module, inputs, "normalize", normalize);
        add_int32(module, inputs, "denoise", denoise);
        add_int32(module, inputs, "retry_badcase", retry_badcase);
        add_int32(module, inputs, "retry_badcase_max_times", retry_badcase_max_times);
        add_float32(module, inputs, "retry_badcase_ratio_threshold", retry_badcase_ratio_threshold);
        add_int32(module, inputs, "seed", seed);
    }

    int32_t sample_rate;
    int32_t reference_sample_rate;
    float cfg_value;
    int32_t inference_timesteps;
    int32_t normalize;
    int32_t denoise;
    int32_t retry_badcase;
    int32_t retry_badcase_max_times;
    float retry_badcase_ratio_threshold;
    int32_t seed;

  private:
    static void add_int32(const ITrtModule& module, TensorMap& inputs, const char* name,
                          const int32_t& value) {
        if (!module.has_input(name) || inputs.find(name) != inputs.end())
            return;
        Tensor tensor;
        tensor.data = const_cast<int32_t*>(&value);
        tensor.shape = {1};
        tensor.dtype = DType::kInt32;
        inputs.emplace(name, tensor);
    }

    static void add_float32(const ITrtModule& module, TensorMap& inputs, const char* name,
                            const float& value) {
        if (!module.has_input(name) || inputs.find(name) != inputs.end())
            return;
        Tensor tensor;
        tensor.data = const_cast<float*>(&value);
        tensor.shape = {1};
        tensor.dtype = DType::kFloat32;
        inputs.emplace(name, tensor);
    }
};

void validate_stage_bindings(const audio::VoxCPM2LoadedComponent& component,
                             const audio::VoxCPM2GenerationStage& stage) {
    if (!component.module->has_input(stage.input_artifact)) {
        throw std::runtime_error(
            "VoxCPM2Pipeline: stage " + component.name + " (" + component.engine_section +
            ") is missing required input binding '" + stage.input_artifact + "'");
    }

    const char* output_binding = stage.output_artifact;
    if (!component.module->has_output(output_binding) &&
        stage.kind == audio::VoxCPM2StageKind::kAudioVae &&
        component.module->has_output("output0")) {
        output_binding = "output0";
    }
    if (!component.module->has_output(output_binding)) {
        throw std::runtime_error(
            "VoxCPM2Pipeline: stage " + component.name + " (" + component.engine_section +
            ") is missing required output binding '" + stage.output_artifact + "'");
    }
    for (std::size_t i = 0; i < stage.required_side_input_count; ++i) {
        const auto* name = stage.required_side_inputs[i];
        if (!component.module->has_input(name)) {
            throw std::runtime_error("VoxCPM2Pipeline: stage " + component.name + " (" +
                                     component.engine_section +
                                     ") is missing required side input binding '" + name + "'");
        }
    }
    for (std::size_t i = 0; i < stage.required_control_input_count; ++i) {
        const auto* name = stage.required_control_inputs[i];
        if (!component.module->has_input(name)) {
            throw std::runtime_error("VoxCPM2Pipeline: stage " + component.name + " (" +
                                     component.engine_section +
                                     ") is missing required control input binding '" + name + "'");
        }
    }
}

void add_required_artifact_inputs(const audio::VoxCPM2GenerationStage& stage,
                                  const StageArtifacts& artifacts, TensorMap& inputs,
                                  const std::string& component_name) {
    for (std::size_t i = 0; i < stage.required_side_input_count; ++i) {
        const auto* name = stage.required_side_inputs[i];
        if (inputs.find(name) != inputs.end())
            continue;
        const auto artifact_it = artifacts.find(name);
        if (artifact_it == artifacts.end()) {
            throw std::runtime_error("VoxCPM2Pipeline: stage " + component_name +
                                     " is missing required side artifact '" + name + "'");
        }
        inputs.emplace(name, artifact_it->second.as_tensor());
    }
}

void add_declared_artifact_inputs(const ITrtModule& module, const StageArtifacts& artifacts,
                                  TensorMap& inputs) {
    for (const auto& artifact : artifacts) {
        if (inputs.find(artifact.first) != inputs.end())
            continue;
        if (!module.has_input(artifact.first))
            continue;
        inputs.emplace(artifact.first, artifact.second.as_tensor());
    }
}

bool component_has_cache_binding(const audio::VoxCPM2LoadedComponent& component,
                                 const KvCacheBinding& binding) {
    return component.module->has_input(binding.past) && component.module->has_output(binding.present);
}

bool component_has_partial_cache_binding(const audio::VoxCPM2LoadedComponent& component,
                                         const KvCacheBinding& binding) {
    return component.module->has_input(binding.past) || component.module->has_output(binding.present);
}

bool lm_cache_mode_enabled(const std::vector<audio::VoxCPM2LoadedComponent>& components) {
    return component_has_cache_binding(components[1], kTslmKvCache) &&
           component_has_cache_binding(components[2], kRalmKvCache);
}

void validate_partial_cache_bindings(
    const std::vector<audio::VoxCPM2LoadedComponent>& components) {
    const bool tslm_full = component_has_cache_binding(components[1], kTslmKvCache);
    const bool ralm_full = component_has_cache_binding(components[2], kRalmKvCache);
    const bool tslm_partial = component_has_partial_cache_binding(components[1], kTslmKvCache);
    const bool ralm_partial = component_has_partial_cache_binding(components[2], kRalmKvCache);
    if ((tslm_partial || ralm_partial) && !(tslm_full && ralm_full)) {
        throw std::runtime_error(
            "VoxCPM2Pipeline: TSLM/RALM cache mode requires both complete cache bindings "
            "tslm_past_kv_cache=>tslm_present_kv_cache and "
            "ralm_past_kv_cache=>ralm_present_kv_cache");
    }
}

void ensure_zero_cache_artifact(const audio::VoxCPM2LoadedComponent& component,
                                StageArtifacts& artifacts, const KvCacheBinding& binding) {
    if (!component_has_cache_binding(component, binding))
        return;
    if (artifacts.find(binding.past) != artifacts.end())
        return;
    artifacts[binding.past] = make_zero_input_tensor(*component.module, binding.past);
}

void roll_present_cache(StageArtifacts& artifacts, const KvCacheBinding& binding) {
    auto present_it = artifacts.find(binding.present);
    if (present_it == artifacts.end())
        return;
    artifacts[binding.past] = std::move(present_it->second);
    artifacts.erase(present_it);
}

void roll_stage_cache_outputs(const audio::VoxCPM2GenerationStage& stage,
                              StageArtifacts& artifacts) {
    if (stage.kind == audio::VoxCPM2StageKind::kTslm) {
        roll_present_cache(artifacts, kTslmKvCache);
    } else if (stage.kind == audio::VoxCPM2StageKind::kRalm) {
        roll_present_cache(artifacts, kRalmKvCache);
    }
}

OwnedStageTensor run_stage(const audio::VoxCPM2LoadedComponent& component,
                           const audio::VoxCPM2GenerationStage& stage, StageArtifacts& artifacts,
                           const RuntimeScalarInputs& controls) {
    validate_stage_bindings(component, stage);
    const auto input_it = artifacts.find(stage.input_artifact);
    if (input_it == artifacts.end()) {
        throw std::runtime_error("VoxCPM2Pipeline: stage " + component.name +
                                 " is missing prepared input artifact '" + stage.input_artifact +
                                 "'");
    }
    const auto& input = input_it->second;
    validate_tensor_contract(input.as_tensor(), stage.input_tensor, stage, component.name, "input");
    TensorMap inputs;
    inputs.emplace(stage.input_artifact, input.as_tensor());
    add_required_artifact_inputs(stage, artifacts, inputs, component.name);
    add_declared_artifact_inputs(*component.module, artifacts, inputs);
    controls.add_to(*component.module, inputs);
    auto outputs = component.module->forward(inputs);
    const char* output_binding = stage.output_artifact;
    if (outputs.find(output_binding) == outputs.end() &&
        stage.kind == audio::VoxCPM2StageKind::kAudioVae) {
        output_binding = "output0";
    }
    const auto output_it = outputs.find(output_binding);
    if (output_it == outputs.end()) {
        throw std::runtime_error("VoxCPM2Pipeline: stage " + component.name +
                                 " did not return output artifact '" + stage.output_artifact + "'");
    }

    validate_tensor_contract(output_it->second, stage.output_tensor, stage, component.name,
                             "output");
    artifacts[stage.output_artifact] =
        copy_stage_tensor(output_it->second, stage.output_artifact, component.name);
    for (const auto& output : outputs) {
        if (output.first == stage.output_artifact || output.first == output_binding)
            continue;
        artifacts[output.first] = copy_stage_tensor(output.second, output.first, component.name);
    }
    roll_stage_cache_outputs(stage, artifacts);
    return artifacts.at(stage.output_artifact);
}

OwnedStageTensor run_cache_bound_lm_prefill(
    const std::vector<audio::VoxCPM2LoadedComponent>& components,
    const audio::VoxCPM2GenerationPlan& plan, StageArtifacts& artifacts,
    const RuntimeScalarInputs& controls, std::size_t active_text_token_count) {
    ensure_zero_cache_artifact(components[1], artifacts, kTslmKvCache);
    ensure_zero_cache_artifact(components[2], artifacts, kRalmKvCache);

    const auto local_text_features = artifacts.at("local_text_features");
    const auto text_tokens = artifacts.at("text_tokens");
    const auto text_mask = artifacts.at("text_mask");
    const auto audio_mask = artifacts.at("audio_mask");

    if (active_text_token_count == 0) {
        throw std::runtime_error("VoxCPM2Pipeline: cache prefill requires active text tokens");
    }
    if (static_cast<std::size_t>(local_text_features.shape.front()) < active_text_token_count) {
        throw std::runtime_error(
            "VoxCPM2Pipeline: local_text_features has fewer rows than active text tokens");
    }

    OwnedStageTensor current;
    for (std::size_t pos = 0; pos < active_text_token_count; ++pos) {
        artifacts["local_text_features"] =
            slice_first_dim(local_text_features, pos, 1, "local_text_features");
        artifacts["text_tokens"] = slice_first_dim(text_tokens, pos, 1, "text_tokens");
        artifacts["text_mask"] = slice_first_dim(text_mask, pos, 1, "text_mask");
        artifacts["audio_mask"] = slice_first_dim(audio_mask, pos, 1, "audio_mask");
        artifacts["position_id"] = make_int32_scalar_tensor(checked_position_id(pos, 0));

        (void)run_stage(components[1], plan.stages[1], artifacts, controls);
        current = run_stage(components[2], plan.stages[2], artifacts, controls);
    }
    return current;
}

void refresh_autoregressive_hidden_state(
    const std::vector<audio::VoxCPM2LoadedComponent>& components,
    const audio::VoxCPM2GenerationPlan& plan, StageArtifacts& artifacts,
    const RuntimeScalarInputs& controls, const OwnedStageTensor& generated_patch,
    std::size_t completed_generation_step, std::size_t active_text_token_count,
    const VoxCPM2Config& cfg) {
    artifacts["audio_feats"] =
        make_audio_feats_from_patch(generated_patch, cfg, plan.stages[3].output_artifact);
    artifacts["text_tokens"] = make_single_step_text_tokens_tensor();
    artifacts["text_mask"] = make_single_step_mask_tensor(0.0F);
    artifacts["audio_mask"] = make_single_step_mask_tensor(1.0F);
    artifacts["position_id"] =
        make_int32_scalar_tensor(checked_position_id(active_text_token_count,
                                                     completed_generation_step));

    for (std::size_t i = 0; i < 3; ++i) {
        (void)run_stage(components[i], plan.stages[i], artifacts, controls);
    }
    keep_latest_hidden_artifacts(artifacts, 1);
}

OwnedStageTensor run_locdit_autoregressive(
    const std::vector<audio::VoxCPM2LoadedComponent>& components,
    const audio::VoxCPM2GenerationPlan& plan, StageArtifacts& artifacts,
    const RuntimeScalarInputs& controls, std::size_t generation_steps,
    std::size_t active_text_token_count, const VoxCPM2Config& cfg) {
    OwnedStageTensor generated_latents;
    const auto& component = components[3];
    const auto& stage = plan.stages[3];
    for (std::size_t step = 0; step < generation_steps; ++step) {
        const auto locdit_output = run_stage(component, stage, artifacts, controls);
        auto generated_patch = extract_locdit_patch(locdit_output, step, cfg, stage.output_artifact);
        append_first_dim(generated_latents, generated_patch, stage.output_artifact);
        artifacts["feat_cond"] = generated_patch;
        if (step > kVoxCPM2MinGenerationSteps && stop_logits_predict_stop(artifacts))
            break;
        if (step + 1 < generation_steps) {
            refresh_autoregressive_hidden_state(components, plan, artifacts, controls,
                                                generated_patch, step, active_text_token_count,
                                                cfg);
        }
    }
    artifacts[stage.output_artifact] = generated_latents;
    return generated_latents;
}

AudioResult make_audio_result(const OwnedStageTensor& waveform,
                              const audio::VoxCPM2GenerationPlan& plan) {
    if (waveform.dtype != DType::kFloat32) {
        throw std::runtime_error("VoxCPM2Pipeline: expected waveform_f32 as float32, got " +
                                 std::string(dtype_name(waveform.dtype)));
    }
    if (waveform.storage.size() % sizeof(float) != 0) {
        throw std::runtime_error("VoxCPM2Pipeline: waveform_f32 byte size is not float-aligned");
    }
    const auto sample_count = waveform.storage.size() / sizeof(float);
    if (sample_count > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("VoxCPM2Pipeline: waveform_f32 has too many samples");
    }

    AudioResult out;
    out.samples.resize(sample_count);
    if (!out.samples.empty()) {
        std::memcpy(out.samples.data(), waveform.storage.data(), waveform.storage.size());
    }
    out.num_samples = static_cast<int32_t>(out.samples.size());
    out.sample_rate = plan.config.sample_rate;
    return out;
}

} // namespace

VoxCPM2Pipeline::VoxCPM2Pipeline(std::vector<audio::VoxCPM2LoadedComponent> components,
                                 audio::VoxCPM2GenerationPlan plan, std::string model_id_str,
                                 std::shared_ptr<ITokenizer> tokenizer)
    : components_(std::move(components)), plan_(std::move(plan)),
      model_id_(std::move(model_id_str)), tokenizer_(std::move(tokenizer)) {
    validate_components();
}

void VoxCPM2Pipeline::validate_components() const {
    if (!audio::voxcpm2_generation_plan_matches_component_contract()) {
        throw std::runtime_error(
            "VoxCPM2Pipeline: generation plan no longer matches component contract");
    }
    if (components_.size() != plan_.stages.size()) {
        throw std::runtime_error(
            "VoxCPM2Pipeline: expected " + std::to_string(plan_.stages.size()) +
            " loaded component modules, got " + std::to_string(components_.size()));
    }

    for (std::size_t i = 0; i < plan_.stages.size(); ++i) {
        const auto& component = components_[i];
        const auto& stage = plan_.stages[i];
        if (component.name != stage.name || component.engine_section != stage.engine_section) {
            throw std::runtime_error(
                "VoxCPM2Pipeline: loaded component order does not match generation plan at stage " +
                std::to_string(i) + " (expected " + stage.name + "/" + stage.engine_section +
                ", got " + component.name + "/" + component.engine_section + ")");
        }
        if (component.module == nullptr || !component.module->ok()) {
            throw std::runtime_error("VoxCPM2Pipeline: invalid loaded module for stage " +
                                     component.name);
        }
        validate_stage_bindings(component, stage);
    }
    validate_partial_cache_bindings(components_);
}

AudioResult VoxCPM2Pipeline::generate_audio(const std::string& prompt, const GenerateConfig& cfg) {
    auto effective_cfg = plan_.config;
    if (cfg.cfg_scale >= 0.0F)
        effective_cfg.cfg_value = cfg.cfg_scale;
    if (cfg.num_steps > 0)
        effective_cfg.inference_timesteps = cfg.num_steps;
    if (cfg.seed >= 0)
        effective_cfg.seed = cfg.seed;

    const auto effective_plan = audio::make_voxcpm2_generation_plan(effective_cfg);
    const RuntimeScalarInputs controls(effective_plan.config);
    StageArtifacts artifacts;
    artifacts.emplace("text_utf8", make_prompt_tensor(prompt));
    auto text_tokens = make_text_tokens_tensor(prompt, tokenizer_.get(),
                                               effective_plan.config.max_text_steps);
    const auto text_token_count = static_cast<std::size_t>(text_tokens.tensor.shape.front());
    const auto active_text_token_count = text_tokens.actual_count;
    artifacts.emplace("text_tokens", std::move(text_tokens.tensor));
    artifacts.emplace("text_mask", make_float_mask_tensor(text_token_count, active_text_token_count,
                                                          1.0F));
    artifacts.emplace("audio_mask", make_float_mask_tensor(text_token_count, 0, 1.0F));
    artifacts.emplace("audio_feats",
                      make_zero_audio_feats_tensor(text_token_count, effective_plan.config));
    artifacts.emplace("feat_cond", make_initial_feat_cond_tensor(effective_plan.config));
    OwnedStageTensor current =
        run_stage(components_[0], effective_plan.stages[0], artifacts, controls);
    if (lm_cache_mode_enabled(components_)) {
        current = run_cache_bound_lm_prefill(components_, effective_plan, artifacts, controls,
                                             active_text_token_count);
    } else {
        for (std::size_t i = 1; i < 3; ++i) {
            current = run_stage(components_[i], effective_plan.stages[i], artifacts, controls);
        }
    }
    validate_lm_state_ready(artifacts);
    const auto generation_steps =
        resolve_latent_generation_steps(active_text_token_count, cfg, effective_plan.config);
    keep_latest_hidden_artifacts(artifacts, active_text_token_count);
    current = run_locdit_autoregressive(components_, effective_plan, artifacts, controls,
                                        generation_steps, active_text_token_count,
                                        effective_plan.config);
    current = run_stage(components_[4], effective_plan.stages[4], artifacts, controls);
    current = trim_audio_vae_waveform_to_latents(std::move(current),
                                                 artifacts.at("audio_vae_latents"),
                                                 *components_[4].module);

    auto audio = make_audio_result(current, effective_plan);
    return audio;
}

} // namespace trtmc
