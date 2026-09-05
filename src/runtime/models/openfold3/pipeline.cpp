/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/openfold3/pipeline.h"

#include "utils/sha256.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace trtmc::openfold3 {
namespace {

using Vec3 = std::array<float, 3>;

constexpr float kSigmaMin = 0.0004F;
constexpr float kSigmaMax = 160.0F;
constexpr float kSigmaData = 16.0F;
constexpr float kRho = 7.0F;
constexpr float kGamma0 = 0.8F;
constexpr float kGammaMin = 1.0F;
constexpr float kNoiseScale = 1.003F;
constexpr float kStepScale = 1.5F;

std::unordered_set<std::string> names(const std::vector<TensorInfo>& tensors) {
    std::unordered_set<std::string> result;
    for (const auto& tensor : tensors) {
        if (!result.insert(tensor.name).second)
            throw std::invalid_argument("OpenFold3 engine has duplicate tensor names");
    }
    return result;
}

void requireNames(ITrtModule& module, std::initializer_list<std::string_view> inputs,
                  std::initializer_list<std::string_view> outputs) {
    const auto actual_inputs = names(module.input_info());
    const auto actual_outputs = names(module.output_info());
    for (const auto input : inputs) {
        if (actual_inputs.count(std::string(input)) == 0)
            throw std::invalid_argument("OpenFold3 engine is missing input: " + std::string(input));
    }
    for (const auto output : outputs) {
        if (actual_outputs.count(std::string(output)) == 0)
            throw std::invalid_argument("OpenFold3 engine is missing output: " +
                                        std::string(output));
    }
}

void bindPointer(ITrtModule& consumer, std::string_view input, ITrtModule& producer,
                 std::string_view output) {
    const auto in = std::string(input);
    const auto out = std::string(output);
    if (!consumer.has_input(in) || !producer.has_output(out) ||
        consumer.tensor_dtype(in) != producer.tensor_dtype(out) ||
        consumer.tensor_shape(in) != producer.tensor_shape(out)) {
        throw std::invalid_argument("OpenFold3 graph edge differs: " + out + " -> " + in);
    }
    consumer.bind_external(in, producer.device_ptr(out));
}

void requireStream(const std::unique_ptr<ITrtModule>& module, cudaStream_t stream) {
    if (!module || !module->ok() || module->stream() != stream)
        throw std::invalid_argument("OpenFold3 engines must be valid on one CUDA stream");
}

float bfloat16ToFloat(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16U;
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float float16ToFloat(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16U;
    uint32_t exponent = (value >> 10U) & 0x1FU;
    uint32_t mantissa = value & 0x03FFU;
    uint32_t bits = 0;
    if (exponent == 0U) {
        if (mantissa == 0U) {
            bits = sign;
        } else {
            uint32_t shift = 0;
            while ((mantissa & 0x0400U) == 0U) {
                mantissa <<= 1U;
                ++shift;
            }
            mantissa &= 0x03FFU;
            bits = sign | ((113U - shift) << 23U) | (mantissa << 13U);
        }
    } else if (exponent == 0x1FU) {
        bits = sign | 0x7F800000U | (mantissa << 13U);
    } else {
        bits = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
    }
    float result = 0.0F;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

std::size_t outputElementCount(const std::vector<int64_t>& shape, std::string_view tensor_name) {
    if (shape.empty())
        throw std::runtime_error("OpenFold3 output has no shape: " + std::string(tensor_name));
    std::size_t result = 1;
    for (const auto dimension : shape) {
        if (dimension <= 0 ||
            result > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(dimension))
            throw std::runtime_error("OpenFold3 output has an invalid shape: " +
                                     std::string(tensor_name));
        result *= static_cast<std::size_t>(dimension);
    }
    return result;
}

template <typename T>
std::vector<T> copyDeviceOutput(ITrtModule& module, const std::string& tensor_name,
                                std::size_t count, cudaStream_t stream) {
    std::vector<T> result(count);
    if (cudaMemcpyAsync(result.data(), module.device_ptr(tensor_name), count * sizeof(T),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess)
        throw std::runtime_error("OpenFold3 failed to read output: " + tensor_name);
    return result;
}

template <typename Converter>
std::vector<float> convert16BitOutput(std::vector<uint16_t> storage, Converter converter) {
    std::vector<float> result(storage.size());
    std::transform(storage.begin(), storage.end(), result.begin(), converter);
    return result;
}

std::vector<float> copyOutput(ITrtModule& module, std::string_view name, std::size_t count,
                              cudaStream_t stream) {
    const auto tensor_name = std::string(name);
    const auto dtype = module.tensor_dtype(tensor_name);
    if (outputElementCount(module.tensor_shape(tensor_name), tensor_name) != count)
        throw std::runtime_error("OpenFold3 output extent differs: " + tensor_name);
    if (dtype == DType::kFloat32)
        return copyDeviceOutput<float>(module, tensor_name, count, stream);
    if (dtype == DType::kBFloat16)
        return convert16BitOutput(copyDeviceOutput<uint16_t>(module, tensor_name, count, stream),
                                  bfloat16ToFloat);
    if (dtype == DType::kFloat16)
        return convert16BitOutput(copyDeviceOutput<uint16_t>(module, tensor_name, count, stream),
                                  float16ToFloat);
    throw std::runtime_error("OpenFold3 output has unsupported dtype: " + tensor_name);
}

std::vector<float> expectedBins(const std::vector<float>& logits, std::size_t rows, int bins,
                                float begin, float end) {
    if (logits.size() != rows * static_cast<std::size_t>(bins))
        throw std::invalid_argument("OpenFold3 confidence logits have the wrong extent");
    std::vector<float> result(rows);
    const float width = (end - begin) / static_cast<float>(bins);
    for (std::size_t row = 0; row < rows; ++row) {
        const auto* values = logits.data() + row * static_cast<std::size_t>(bins);
        const float maximum = *std::max_element(values, values + bins);
        float denominator = 0.0F;
        float numerator = 0.0F;
        for (int bin = 0; bin < bins; ++bin) {
            const float probability = std::exp(values[bin] - maximum);
            denominator += probability;
            numerator += probability * (begin + (static_cast<float>(bin) + 0.5F) * width);
        }
        result[row] = numerator / denominator;
    }
    return result;
}

std::vector<float> sigmaSchedule(int steps) {
    std::vector<float> result(static_cast<std::size_t>(steps) + 1U);
    const float begin = std::pow(kSigmaMax, 1.0F / kRho);
    const float end = std::pow(kSigmaMin, 1.0F / kRho);
    for (int step = 0; step <= steps; ++step) {
        const float fraction = static_cast<float>(step) / static_cast<float>(steps);
        result[static_cast<std::size_t>(step)] =
            kSigmaData * std::pow(begin + fraction * (end - begin), kRho);
    }
    return result;
}

void augment(std::vector<Vec3>& coordinates, const std::array<float, 9>& rotation,
             const std::array<float, 3>& translation, int atom_count) {
    Vec3 mean{};
    for (int atom = 0; atom < atom_count; ++atom)
        for (int axis = 0; axis < 3; ++axis)
            mean[axis] += coordinates[static_cast<std::size_t>(atom)][axis];
    for (float& value : mean)
        value /= static_cast<float>(atom_count);
    for (int atom = 0; atom < atom_count; ++atom) {
        const auto source = coordinates[static_cast<std::size_t>(atom)];
        for (int row = 0; row < 3; ++row) {
            float value = translation[static_cast<std::size_t>(row)];
            for (int column = 0; column < 3; ++column)
                value += rotation[static_cast<std::size_t>(row * 3 + column)] *
                         (source[static_cast<std::size_t>(column)] -
                          mean[static_cast<std::size_t>(column)]);
            coordinates[static_cast<std::size_t>(atom)][static_cast<std::size_t>(row)] = value;
        }
    }
    for (std::size_t atom = static_cast<std::size_t>(atom_count); atom < coordinates.size(); ++atom)
        coordinates[atom] = {};
}

std::vector<float> pack(const std::vector<Vec3>& coordinates) {
    std::vector<float> result(coordinates.size() * 3U);
    for (std::size_t atom = 0; atom < coordinates.size(); ++atom)
        for (int axis = 0; axis < 3; ++axis)
            result[atom * 3U + static_cast<std::size_t>(axis)] = coordinates[atom][axis];
    return result;
}

std::vector<Vec3> unpack(const std::vector<float>& coordinates) {
    if (coordinates.size() % 3U != 0)
        throw std::invalid_argument("OpenFold3 coordinate extent is invalid");
    std::vector<Vec3> result(coordinates.size() / 3U);
    for (std::size_t atom = 0; atom < result.size(); ++atom)
        std::copy_n(coordinates.data() + atom * 3U, 3, result[atom].begin());
    return result;
}

float tmScore(const std::vector<float>& pae_logits, int token_count) {
    const float clipped = std::max(static_cast<float>(token_count), 19.0F);
    const float d0 = 1.24F * std::cbrt(clipped - 15.0F) - 1.8F;
    float maximum_score = 0.0F;
    for (int row = 0; row < token_count; ++row) {
        float score = 0.0F;
        for (int column = 0; column < token_count; ++column) {
            const auto offset = static_cast<std::size_t>((row * token_count + column) * 64);
            const float maximum =
                *std::max_element(pae_logits.begin() + offset, pae_logits.begin() + offset + 64U);
            float denominator = 0.0F;
            float numerator = 0.0F;
            for (int bin = 0; bin < 64; ++bin) {
                const float probability = std::exp(pae_logits[offset + bin] - maximum);
                const float distance = (static_cast<float>(bin) + 0.5F) * 0.5F;
                denominator += probability;
                numerator += probability / (1.0F + (distance / d0) * (distance / d0));
            }
            score += numerator / denominator;
        }
        maximum_score = std::max(maximum_score, score / static_cast<float>(token_count));
    }
    return maximum_score;
}

float globalPde(const std::vector<float>& pde, const std::vector<float>& distogram,
                int token_count) {
    double numerator = 0.0;
    double denominator = 0.0;
    const auto rows = static_cast<std::size_t>(token_count) * token_count;
    for (std::size_t row = 0; row < rows; ++row) {
        const auto* logits = distogram.data() + row * 64U;
        const float maximum = *std::max_element(logits, logits + 64);
        double all = 0.0;
        double contact = 0.0;
        for (int bin = 0; bin < 64; ++bin) {
            const double probability = std::exp(logits[bin] - maximum);
            all += probability;
            if (2.0F + static_cast<float>(bin + 1) * (20.0F / 64.0F) <= 8.0F)
                contact += probability;
        }
        const double weight = contact / all;
        numerator += weight * pde[row];
        denominator += weight;
    }
    return static_cast<float>(numerator / (denominator + 1.0e-8));
}

std::string cifToken(std::string value) {
    if (value.empty())
        return ".";
    if (value.find_first_of(" \t\r\n'\"") != std::string::npos)
        return "'" + value + "'";
    return value;
}

bool validTokenShape(const std::vector<int64_t>& shape) {
    return shape.size() == 2 && shape[0] == 1 && shape[1] > 0 && shape[1] <= kMaxTokenCount;
}

bool validAtomShape(const std::vector<int64_t>& shape) {
    return shape.size() == 3 && shape[0] == 1 && shape[1] > 0 && shape[1] % 32 == 0 &&
           shape[2] == 3;
}

bool validRepresentativeShape(const std::vector<int64_t>& shape, int64_t token_count,
                              int64_t padded_atom_count) {
    return shape.size() == 3 && shape[0] == 1 && shape[1] == token_count && shape[2] > 0 &&
           shape[2] <= padded_atom_count;
}

template <typename Values>
bool allFinite(const Values& values) {
    return std::all_of(values.begin(), values.end(), [](const auto& value) {
        if constexpr (std::is_arithmetic_v<std::decay_t<decltype(value)>>)
            return std::isfinite(value);
        else
            return std::all_of(value.begin(), value.end(),
                               [](float component) { return std::isfinite(component); });
    });
}

} // namespace

OpenFold3Pipeline::OpenFold3Pipeline(EngineSet engines, BundleArtifacts artifacts,
                                     std::string model_id)
    : engines_(std::move(engines)), artifacts_(std::move(artifacts)),
      model_id_(std::move(model_id)) {
    if (model_id_.empty())
        model_id_ = "openfold3";
    validateAndBind();
}

const FeatureTensor& OpenFold3Pipeline::feature(std::string_view name) const {
    return artifacts_.features.require(name);
}

void OpenFold3Pipeline::bindFeature(ITrtModule& module, std::string_view name) {
    const auto found = device_features_.find(std::string(name));
    if (found == device_features_.end() || !module.has_input(std::string(name)) ||
        module.tensor_dtype(std::string(name)) != found->second.dtype() ||
        module.tensor_shape(std::string(name)) != found->second.shape()) {
        throw std::invalid_argument("OpenFold3 feature binding differs: " + std::string(name));
    }
    module.bind_external(std::string(name), found->second.data());
}

void OpenFold3Pipeline::validateAndBind() {
    validateEngineSet();
    validateProfile();
    validateRandomSamples();
    validateRandomPadding();
    uploadFeaturesAndAllocate();
    bindInputAndTrunk();
    bindPairformer();
    bindDiffusion();
    bindConfidence();
}

void OpenFold3Pipeline::validateEngineSet() {
    const std::array<const std::unique_ptr<ITrtModule>*, 6> required{
        &engines_.input,       &engines_.trunk_cycle,  &engines_.conditioning,
        &engines_.score_input, &engines_.score_output, &engines_.confidence,
    };
    if (std::any_of(required.begin(), required.end(), [](const auto* engine) { return !*engine; }))
        throw std::invalid_argument("OpenFold3 engine set is incomplete");
    stream_ = engines_.input->stream();
    if (stream_ == nullptr)
        throw std::invalid_argument("OpenFold3 CUDA stream is null");
    for (const auto* engine : required)
        requireStream(*engine, stream_);
    for (const auto& engine : engines_.pairformer)
        requireStream(engine, stream_);
    for (const auto& engine : engines_.score_token)
        requireStream(engine, stream_);
}

void OpenFold3Pipeline::validateProfile() {
    const auto& token_shape = feature("token_mask").shape;
    const auto& atom_shape = feature("ref_pos").shape;
    const auto& representative_shape = feature("representative_atom_map").shape;
    if (!validTokenShape(token_shape) || !validAtomShape(atom_shape) ||
        !validRepresentativeShape(representative_shape, token_shape[1], atom_shape[1]))
        throw std::invalid_argument("OpenFold3 feature shapes do not define a valid profile");
    token_count_ = static_cast<int>(token_shape[1]);
    padded_atom_count_ = static_cast<int>(atom_shape[1]);
    atom_count_ = static_cast<int>(representative_shape[2]);
}

void OpenFold3Pipeline::validateRandomSamples() const {
    if (artifacts_.random_samples.seed != 42 || artifacts_.random_samples.sampling_steps != 200 ||
        artifacts_.random_samples.padded_atom_count != padded_atom_count_)
        throw std::invalid_argument("OpenFold3 random samples differ from the feature profile");
    const auto& random = artifacts_.random_samples;
    if (!allFinite(random.initial) || !allFinite(random.rotations) ||
        !allFinite(random.translations) || !allFinite(random.noise))
        throw std::invalid_argument("OpenFold3 random samples contain non-finite values");
}

void OpenFold3Pipeline::validateRandomPadding() const {
    const auto& random = artifacts_.random_samples;
    for (int atom = atom_count_; atom < padded_atom_count_; ++atom) {
        for (int axis = 0; axis < 3; ++axis) {
            if (random.initial[static_cast<std::size_t>(atom * 3 + axis)] != 0.0F)
                throw std::invalid_argument("OpenFold3 initial-noise padding must be zero");
            for (int step = 0; step < 200; ++step) {
                if (random.noise[static_cast<std::size_t>((step * padded_atom_count_ + atom) * 3 +
                                                          axis)] != 0.0F)
                    throw std::invalid_argument("OpenFold3 step-noise padding must be zero");
            }
        }
    }
}

void OpenFold3Pipeline::uploadFeaturesAndAllocate() {
    for (const auto name : kFeatureNames) {
        const auto& host = feature(name);
        auto [entry, inserted] =
            device_features_.try_emplace(std::string(name), host.shape, host.dtype, stream_);
        if (!inserted || !entry->second.ok() || !entry->second.copy_from_host(host.data.data()))
            throw std::runtime_error("OpenFold3 failed to upload feature: " + std::string(name));
    }
    zero_s_ = DeviceTensor::zeros({1, token_count_, 384}, DType::kFloat32, stream_);
    zero_z_ = DeviceTensor::zeros({1, token_count_, token_count_, 128}, DType::kFloat32, stream_);
    noisy_ = DeviceTensor({1, padded_atom_count_, 3}, DType::kFloat32, stream_);
    time_ = DeviceTensor({1}, DType::kFloat32, stream_);
    confidence_positions_ = DeviceTensor({1, atom_count_, 3}, DType::kFloat32, stream_);
}

void OpenFold3Pipeline::bindInputAndTrunk() {
    requireNames(*engines_.input,
                 {"ref_pos", "ref_mask", "ref_element", "ref_charge", "ref_atom_name_chars",
                  "ref_space_uid", "atom_mask", "atom_to_token_index", "restype", "profile",
                  "deletion_mean", "relpos", "token_bonds"},
                 {"s_input", "s_init", "z_init"});
    for (const auto name :
         {"ref_pos", "ref_mask", "ref_element", "ref_charge", "ref_atom_name_chars",
          "ref_space_uid", "atom_mask", "atom_to_token_index", "restype", "profile",
          "deletion_mean", "relpos", "token_bonds"})
        bindFeature(*engines_.input, name);

    requireNames(*engines_.trunk_cycle,
                 {"s_input", "s_init", "z_init", "s_previous", "z_previous", "token_mask", "msa",
                  "has_deletion", "deletion_value", "msa_mask"},
                 {"s", "z"});
    bindPointer(*engines_.trunk_cycle, "s_input", *engines_.input, "s_input");
    bindPointer(*engines_.trunk_cycle, "s_init", *engines_.input, "s_init");
    bindPointer(*engines_.trunk_cycle, "z_init", *engines_.input, "z_init");
    for (const auto name : {"token_mask", "msa", "has_deletion", "deletion_value", "msa_mask"})
        bindFeature(*engines_.trunk_cycle, name);
}

void OpenFold3Pipeline::bindPairformer() {
    ITrtModule* previous = engines_.trunk_cycle.get();
    std::string previous_s = "s";
    std::string previous_z = "z";
    for (auto& engine : engines_.pairformer) {
        requireNames(*engine, {"s", "z", "token_mask"}, {"s_out", "z_out"});
        bindPointer(*engine, "s", *previous, previous_s);
        bindPointer(*engine, "z", *previous, previous_z);
        bindFeature(*engine, "token_mask");
        previous = engine.get();
        previous_s = "s_out";
        previous_z = "z_out";
    }
}

void OpenFold3Pipeline::bindDiffusion() {
    auto* previous = engines_.pairformer.back().get();
    requireNames(*engines_.conditioning,
                 {"noise_level", "s_input", "s_trunk", "z_trunk", "relpos", "token_mask"},
                 {"s_conditioned", "z_conditioned"});
    engines_.conditioning->bind_external("noise_level", time_.data());
    bindPointer(*engines_.conditioning, "s_input", *engines_.input, "s_input");
    bindPointer(*engines_.conditioning, "s_trunk", *previous, "s_out");
    bindPointer(*engines_.conditioning, "z_trunk", *previous, "z_out");
    bindFeature(*engines_.conditioning, "relpos");
    bindFeature(*engines_.conditioning, "token_mask");

    requireNames(*engines_.score_input,
                 {"ref_pos", "ref_mask", "ref_element", "ref_charge", "ref_atom_name_chars",
                  "ref_space_uid", "atom_mask", "atom_to_token_index", "noisy_positions",
                  "noise_level", "s_conditioned", "s_trunk", "z_conditioned"},
                 {"token_representation", "atom_representation", "atom_conditioning",
                  "atom_pair_conditioning"});
    for (const auto name :
         {"ref_pos", "ref_mask", "ref_element", "ref_charge", "ref_atom_name_chars",
          "ref_space_uid", "atom_mask", "atom_to_token_index"})
        bindFeature(*engines_.score_input, name);
    engines_.score_input->bind_external("noisy_positions", noisy_.data());
    engines_.score_input->bind_external("noise_level", time_.data());
    bindPointer(*engines_.score_input, "s_conditioned", *engines_.conditioning, "s_conditioned");
    bindPointer(*engines_.score_input, "s_trunk", *previous, "s_out");
    bindPointer(*engines_.score_input, "z_conditioned", *engines_.conditioning, "z_conditioned");

    ITrtModule* previous_token = engines_.score_input.get();
    std::string previous_a = "token_representation";
    for (auto& engine : engines_.score_token) {
        requireNames(*engine, {"a", "single_condition", "pair_condition", "token_mask"}, {"a_out"});
        bindPointer(*engine, "a", *previous_token, previous_a);
        bindPointer(*engine, "single_condition", *engines_.conditioning, "s_conditioned");
        bindPointer(*engine, "pair_condition", *engines_.conditioning, "z_conditioned");
        bindFeature(*engine, "token_mask");
        previous_token = engine.get();
        previous_a = "a_out";
    }

    requireNames(*engines_.score_output,
                 {"a", "atom_representation", "atom_conditioning", "atom_pair_conditioning",
                  "atom_to_token_index", "atom_mask", "noisy_positions", "noise_level"},
                 {"denoised_positions"});
    bindPointer(*engines_.score_output, "a", *previous_token, "a_out");
    bindPointer(*engines_.score_output, "atom_representation", *engines_.score_input,
                "atom_representation");
    bindPointer(*engines_.score_output, "atom_conditioning", *engines_.score_input,
                "atom_conditioning");
    bindPointer(*engines_.score_output, "atom_pair_conditioning", *engines_.score_input,
                "atom_pair_conditioning");
    bindFeature(*engines_.score_output, "atom_to_token_index");
    bindFeature(*engines_.score_output, "atom_mask");
    engines_.score_output->bind_external("noisy_positions", noisy_.data());
    engines_.score_output->bind_external("noise_level", time_.data());
}

void OpenFold3Pipeline::bindConfidence() {
    auto* previous = engines_.pairformer.back().get();
    requireNames(*engines_.confidence,
                 {"s_input", "s", "z", "positions", "representative_atom_map", "atom_head_index",
                  "token_mask"},
                 {"pae_logits", "pde_logits", "plddt_logits", "experimentally_resolved_logits",
                  "distogram_logits"});
    bindPointer(*engines_.confidence, "s_input", *engines_.input, "s_input");
    bindPointer(*engines_.confidence, "s", *previous, "s_out");
    bindPointer(*engines_.confidence, "z", *previous, "z_out");
    engines_.confidence->bind_external("positions", confidence_positions_.data());
    bindFeature(*engines_.confidence, "representative_atom_map");
    bindFeature(*engines_.confidence, "atom_head_index");
    bindFeature(*engines_.confidence, "token_mask");
}

void OpenFold3Pipeline::runTrunk() {
    engines_.input->forward_device_async({});
    for (int pass = 0; pass < 4; ++pass) {
        engines_.trunk_cycle->bind_external(
            "s_previous",
            pass == 0 ? zero_s_.data() : engines_.pairformer.back()->device_ptr("s_out"));
        engines_.trunk_cycle->bind_external(
            "z_previous",
            pass == 0 ? zero_z_.data() : engines_.pairformer.back()->device_ptr("z_out"));
        engines_.trunk_cycle->forward_device_async({});
        for (auto& engine : engines_.pairformer)
            engine->forward_device_async({});
    }
}

std::vector<float> OpenFold3Pipeline::runDiffusionStep(const std::vector<float>& noisy,
                                                       float time) {
    if (!noisy_.copy_from_host(noisy.data()) || !time_.copy_from_host(&time))
        throw std::runtime_error("OpenFold3 failed to upload diffusion inputs");
    engines_.conditioning->forward_device_async({});
    engines_.score_input->forward_device_async({});
    for (auto& engine : engines_.score_token)
        engine->forward_device_async({});
    engines_.score_output->forward_device_async({});
    return copyOutput(*engines_.score_output, "denoised_positions",
                      static_cast<std::size_t>(padded_atom_count_) * 3U, stream_);
}

std::vector<float> OpenFold3Pipeline::sampleCoordinates() {
    const auto sigmas = sigmaSchedule(200);
    const auto& random = artifacts_.random_samples;
    std::vector<Vec3> coordinates(static_cast<std::size_t>(padded_atom_count_));
    for (int atom = 0; atom < padded_atom_count_; ++atom)
        for (int axis = 0; axis < 3; ++axis)
            coordinates[static_cast<std::size_t>(atom)][axis] =
                sigmas[0] * random.initial[static_cast<std::size_t>(atom * 3 + axis)];
    for (int step = 0; step < 200; ++step) {
        const float next_sigma = sigmas[static_cast<std::size_t>(step + 1)];
        augment(coordinates, random.rotations[static_cast<std::size_t>(step)],
                random.translations[static_cast<std::size_t>(step)], atom_count_);
        const float gamma = next_sigma > kGammaMin ? kGamma0 : 0.0F;
        const float time = sigmas[static_cast<std::size_t>(step)] * (1.0F + gamma);
        const float noise_scale =
            kNoiseScale *
            std::sqrt(std::max(0.0F, time * time - sigmas[static_cast<std::size_t>(step)] *
                                                       sigmas[static_cast<std::size_t>(step)]));
        auto noisy = coordinates;
        for (int atom = 0; atom < padded_atom_count_; ++atom)
            for (int axis = 0; axis < 3; ++axis)
                noisy[static_cast<std::size_t>(atom)][axis] +=
                    noise_scale * random.noise[static_cast<std::size_t>(
                                      (step * padded_atom_count_ + atom) * 3 + axis)];
        const auto denoised = unpack(runDiffusionStep(pack(noisy), time));
        const float advance = kStepScale * (next_sigma - time) / time;
        for (int atom = 0; atom < padded_atom_count_; ++atom)
            for (int axis = 0; axis < 3; ++axis)
                coordinates[static_cast<std::size_t>(atom)][axis] =
                    noisy[static_cast<std::size_t>(atom)][axis] +
                    advance * (noisy[static_cast<std::size_t>(atom)][axis] -
                               denoised[static_cast<std::size_t>(atom)][axis]);
    }
    auto result = pack(coordinates);
    result.resize(static_cast<std::size_t>(atom_count_) * 3U);
    return result;
}

StructureConfidence OpenFold3Pipeline::runConfidence(const std::vector<float>& coordinates) {
    if (!confidence_positions_.copy_from_host(coordinates.data()))
        throw std::runtime_error("OpenFold3 failed to upload confidence coordinates");
    engines_.confidence->forward_device_async({});
    const auto atom_rows = static_cast<std::size_t>(atom_count_);
    const auto pair_rows = static_cast<std::size_t>(token_count_) * token_count_;
    const auto plddt_logits =
        copyOutput(*engines_.confidence, "plddt_logits", atom_rows * 50U, stream_);
    const auto pae_logits =
        copyOutput(*engines_.confidence, "pae_logits", pair_rows * 64U, stream_);
    const auto pde_logits =
        copyOutput(*engines_.confidence, "pde_logits", pair_rows * 64U, stream_);
    const auto distogram =
        copyOutput(*engines_.confidence, "distogram_logits", pair_rows * 64U, stream_);
    StructureConfidence result;
    result.plddt = expectedBins(plddt_logits, atom_rows, 50, 0.0F, 1.0F);
    for (float& value : result.plddt)
        value *= 100.0F;
    result.complex_plddt = std::accumulate(result.plddt.begin(), result.plddt.end(), 0.0F) /
                           static_cast<float>(result.plddt.size());
    result.complex_iplddt = 0.0F;
    result.ptm = tmScore(pae_logits, token_count_);
    result.iptm = 0.0F;
    result.protein_iptm = 0.0F;
    result.confidence_score = result.complex_plddt / 100.0F;
    last_pae_ = expectedBins(pae_logits, pair_rows, 64, 0.0F, 32.0F);
    last_pde_ = expectedBins(pde_logits, pair_rows, 64, 0.0F, 32.0F);
    last_gpde_ = globalPde(last_pde_, distogram, token_count_);
    return result;
}

std::string OpenFold3Pipeline::writeMmcif(const std::vector<float>& coordinates,
                                          const StructureConfidence& confidence) const {
    const auto metadata = nlohmann::json::parse(artifacts_.structure_metadata_json);
    const auto& atoms = metadata.at("atoms");
    if (!atoms.is_array() || atoms.size() != static_cast<std::size_t>(atom_count_) ||
        coordinates.size() != static_cast<std::size_t>(atom_count_) * 3U ||
        confidence.plddt.size() != atoms.size())
        throw std::invalid_argument("OpenFold3 structure metadata differs from output tensors");
    std::ostringstream output;
    output << std::fixed << std::setprecision(3)
           << "data_openfold3\n#\nloop_\n"
              "_atom_site.group_PDB\n_atom_site.id\n_atom_site.type_symbol\n"
              "_atom_site.label_atom_id\n_atom_site.label_alt_id\n"
              "_atom_site.label_comp_id\n_atom_site.label_asym_id\n"
              "_atom_site.label_entity_id\n_atom_site.label_seq_id\n"
              "_atom_site.pdbx_PDB_ins_code\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n"
              "_atom_site.Cartn_z\n_atom_site.occupancy\n_atom_site.B_iso_or_equiv\n"
              "_atom_site.pdbx_formal_charge\n_atom_site.auth_seq_id\n"
              "_atom_site.auth_comp_id\n_atom_site.auth_asym_id\n"
              "_atom_site.auth_atom_id\n_atom_site.pdbx_PDB_model_num\n";
    for (std::size_t atom = 0; atom < atoms.size(); ++atom) {
        const auto& row = atoms.at(atom);
        const auto atom_name = cifToken(row.at("name").get<std::string>());
        const auto residue_name = cifToken(row.at("residue_name").get<std::string>());
        const auto chain_id = cifToken(row.at("chain_id").get<std::string>());
        const auto residue_index = row.at("residue_index").get<int>();
        output << (row.at("hetero").get<bool>() ? "HETATM" : "ATOM") << ' ' << atom + 1 << ' '
               << cifToken(row.at("element").get<std::string>()) << ' ' << atom_name << " . "
               << residue_name << ' ' << chain_id << " 1 " << residue_index << " ? "
               << coordinates[atom * 3U] << ' ' << coordinates[atom * 3U + 1U] << ' '
               << coordinates[atom * 3U + 2U] << " 1.00 " << confidence.plddt[atom] << " ? "
               << residue_index << ' ' << residue_name << ' ' << chain_id << ' ' << atom_name
               << " 1\n";
    }
    output << "#\n";
    return output.str();
}

std::string OpenFold3Pipeline::resultMetadata(const StructurePredictionConfig& cfg,
                                              const StructureConfidence& confidence) const {
    internal::Sha256 request_hash;
    request_hash.update(artifacts_.request);
    const nlohmann::json metadata{
        {"schema_version", 1},
        {"family", "openfold3"},
        {"source_revision", "c4771653c5d0a3ebb0b3af71b05efd64bc44ee86"},
        {"precision", artifacts_.precision},
        {"token_count", token_count_},
        {"atom_count", atom_count_},
        {"recycling_steps", cfg.recycling_steps},
        {"sampling_steps", cfg.sampling_steps},
        {"diffusion_samples", cfg.diffusion_samples},
        {"seed", cfg.seed},
        {"request_sha256", request_hash.hex_digest()},
        {"sample_rank", 0},
        {"sample_ranking_score", nullptr},
        {"sample_ranking_score_applicable", false},
        {"sample_ranking_reason", "the qualified profile emits exactly one diffusion sample"},
        {"average_plddt", confidence.complex_plddt},
        {"gpde", last_gpde_},
        {"ptm", confidence.ptm},
        {"iptm", confidence.iptm},
        {"plddt", confidence.plddt},
        {"pde", last_pde_},
        {"pae", last_pae_},
    };
    return metadata.dump(2) + "\n";
}

StructurePredictionResult
OpenFold3Pipeline::predict_structure(const std::string& input,
                                     const StructurePredictionConfig& cfg) {
    if (input != artifacts_.request)
        throw std::invalid_argument(
            "OpenFold3 native bundles accept their exact preprocessed build request");
    if (cfg.recycling_steps != 3 || cfg.sampling_steps != 200 || cfg.diffusion_samples != 1 ||
        cfg.seed != 42 || cfg.output_format != StructureFormat::kMmcif)
        throw std::invalid_argument(
            "OpenFold3 supports only recycling=3, sampling=200, samples=1, seed=42, mmCIF");
    runTrunk();
    auto coordinates = sampleCoordinates();
    auto confidence = runConfidence(coordinates);
    StructurePredictionResult result;
    result.structure = writeMmcif(coordinates, confidence);
    result.format = StructureFormat::kMmcif;
    result.confidence = std::move(confidence);
    result.metadata_json = resultMetadata(cfg, result.confidence);
    return result;
}

} // namespace trtmc::openfold3
