/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/esmfold2/runtime/pipeline.h"

#include "families/esmfold2/runtime/contract.h"
#include "families/esmfold2/runtime/protein.h"
#include "families/esmfold2/runtime/random.h"
#include "families/esmfold2/runtime/sampler.h"
#include "families/esmfold2/runtime/structure_io.h"
#include "trtmc/runtime/device_tensor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace trtmc::esmfold2 {
namespace {

using Shape = std::vector<std::int64_t>;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("ESMFold2 ") + what + ": " +
                                 cudaGetErrorString(status));
}

std::uint16_t toBfloat16(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7F800000U) == 0x7F800000U && (bits & 0x007FFFFFU) != 0U)
        return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
    const std::uint32_t rounding = 0x7FFFU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>((bits + rounding) >> 16U);
}

std::int64_t elements(const Shape& shape) {
    return std::accumulate(shape.begin(), shape.end(), std::int64_t{1}, std::multiplies<>());
}

// Device copies of host arrays, alive for one request.
class Uploads {
  public:
    explicit Uploads(cudaStream_t stream) : stream_(stream) {}

    DeviceTensor& add(const Shape& shape, DType dtype, const void* host) {
        tensors_.push_back(std::make_unique<DeviceTensor>(shape, dtype, stream_));
        DeviceTensor& tensor = *tensors_.back();
        if (!tensor.ok())
            throw std::runtime_error("ESMFold2 could not allocate a device tensor");
        if (host != nullptr)
            check(cudaMemcpyAsync(tensor.data(), host, tensor.nbytes(), cudaMemcpyHostToDevice,
                                  stream_),
                  "upload");
        return tensor;
    }

  private:
    cudaStream_t stream_;
    std::vector<std::unique_ptr<DeviceTensor>> tensors_;
};

void bindInput(ITrtModule& module, const char* name, const void* pointer, const Shape& shape) {
    if (!module.has_input(name))
        throw std::runtime_error(std::string("ESMFold2 plan has no input ") + name);
    module.bind_external(name, const_cast<void*>(pointer), shape);
}

void bindInput(ITrtModule& module, const char* name, const DeviceTensor& tensor) {
    bindInput(module, name, tensor.data(), tensor.shape());
}

void* output(ITrtModule& module, const char* name) {
    if (!module.has_output(name))
        throw std::runtime_error(std::string("ESMFold2 plan has no output ") + name);
    void* pointer = module.device_ptr(name);
    if (pointer == nullptr)
        throw std::runtime_error(std::string("ESMFold2 plan output is unallocated: ") + name);
    return pointer;
}

template <typename Function>
void parallelFor(std::size_t count, Function function) {
    run_in_chunks(count, chunk_workers(count), function);
}

std::vector<std::uint8_t> pairStateBytes(const RandomStreams& random, std::size_t count, float std,
                                         bool bf16) {
    std::vector<std::uint8_t> bytes(count * (bf16 ? 2U : 4U));
    parallelFor(count, [&](std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index) {
            const float value = random.truncated_normal(Stream::kTrunkInit, index, std);
            if (bf16) {
                const std::uint16_t half = toBfloat16(value);
                std::memcpy(bytes.data() + index * 2U, &half, 2U);
            } else {
                std::memcpy(bytes.data() + index * 4U, &value, 4U);
            }
        }
    });
    return bytes;
}

void keepMask(const RandomStreams& random, std::uint64_t first, std::uint32_t threshold,
              std::vector<std::int32_t>& mask) {
    parallelFor(mask.size(), [&](std::size_t begin, std::size_t end) {
        for (std::size_t index = begin; index < end; ++index)
            mask[index] = random.keep(Stream::kLmDropout, first + index, threshold) ? 1 : 0;
    });
}

PlanFacts factsOf(const ITrtModule& module) {
    if (module.optimization_profile_count() != 1)
        throw std::invalid_argument("ESMFold2 plans have exactly one optimization profile");
    PlanFacts facts;
    for (const TensorInfo& info : module.input_info())
        facts[info.name] = {true, info.dtype, info.shape,
                            module.input_profile_shape(info.name, 0, ProfileShapeSelector::kMin),
                            module.input_profile_shape(info.name, 0, ProfileShapeSelector::kMax)};
    for (const TensorInfo& info : module.output_info())
        facts[info.name] = {false, info.dtype, info.shape, {}, {}};
    return facts;
}

std::string metadataJson(const ProteinRequest& protein, const ProteinFeatures& features,
                         const SampleConfidence& confidence, const FoldControls& controls,
                         std::int32_t steps_run, std::int32_t sample, std::int32_t rank,
                         bool bf16) {
    nlohmann::json value = {
        {"model", "esmfold2"},
        {"name", protein.name},
        {"sequence", features.sequence},
        {"num_residues", features.tokens},
        {"num_atoms", features.real_atoms},
        {"sample_index", sample},
        {"rank", rank},
        {"ptm", confidence.ptm},
        {"iptm", 0.0},
        {"complex_plddt", confidence.complex_plddt},
        {"complex_iplddt", 0.0},
        {"plddt", confidence.plddt},
        {"plddt_scale", "0-1"},
        {"seed", controls.seed},
        {"recycling_steps", controls.loops},
        {"configured_sampling_steps", controls.configured_steps},
        {"sampling_steps", steps_run},
        {"diffusion_samples", controls.samples},
        {"precision", bf16 ? "bf16" : "fp32"},
    };
    return value.dump();
}

} // namespace

EsmFold2Pipeline::EsmFold2Pipeline(EngineSet engines, BundleConfig config)
    : engines_(std::move(engines)), config_(std::move(config)) {
    for (ITrtModule* module :
         {engines_.esmc.get(), engines_.embed.get(), engines_.loop.get(), engines_.decode.get(),
          engines_.conditioning.get(), engines_.denoise.get(), engines_.confidence.get()}) {
        if (module == nullptr)
            throw std::invalid_argument("ESMFold2 bundle is missing a plan");
    }
    if (config_.msa_encoder != static_cast<bool>(engines_.msa))
        throw std::invalid_argument("ESMFold2 bundle's MSA plan does not match runtime.json");
    const std::pair<const char*, const ITrtModule*> plans[] = {
        {"esmc", engines_.esmc.get()},       {"embed", engines_.embed.get()},
        {"msa", engines_.msa.get()},         {"loop", engines_.loop.get()},
        {"decode", engines_.decode.get()},   {"conditioning", engines_.conditioning.get()},
        {"denoise", engines_.denoise.get()}, {"confidence", engines_.confidence.get()}};
    for (const auto& [name, module] : plans) {
        if (module != nullptr)
            check_plan_contract(name, factsOf(*module), config_);
    }
    stream_ = engines_.esmc->stream();
}

StructurePredictionResult
EsmFold2Pipeline::predict_structure(const StructurePredictionRequest& request) {
    const FoldControls controls = resolve_controls(request.config, config_);
    const ProteinRequest protein =
        parse_protein_document(request.document, kMinTokens, config_.max_tokens);
    const ProteinFeatures features =
        featurize_protein(protein.sequence, config_.esmc_bos_id, config_.esmc_eos_id);
    if (features.atoms > config_.max_atoms)
        throw std::invalid_argument("ESMFold2 sequence has more atoms than this bundle's profile");
    const std::int64_t n = features.tokens;
    const std::int64_t a = features.atoms;
    const std::int64_t c = config_.pair_size;
    const std::int64_t s = controls.samples;
    const DType compute = config_.bf16 ? DType::kBFloat16 : DType::kFloat32;
    const std::size_t compute_bytes = config_.bf16 ? 2U : 4U;
    Uploads uploads(stream_);

    // Token and atom features.
    DeviceTensor& input_ids = uploads.add({1, n + 2}, DType::kInt32, features.input_ids.data());
    DeviceTensor& res_type = uploads.add({1, n}, DType::kInt32, features.res_type.data());
    DeviceTensor& residue_index = uploads.add({1, n}, DType::kInt32, features.residue_index.data());
    DeviceTensor& token_index = uploads.add({1, n}, DType::kInt32, features.token_index.data());
    DeviceTensor& asym_id = uploads.add({1, n}, DType::kInt32, features.asym_id.data());
    DeviceTensor& sym_id = uploads.add({1, n}, DType::kInt32, features.sym_id.data());
    DeviceTensor& entity_id = uploads.add({1, n}, DType::kInt32, features.entity_id.data());
    DeviceTensor& rep_atom = uploads.add({1, n}, DType::kInt32, features.rep_atom.data());
    DeviceTensor& ref_pos = uploads.add({1, a, 3}, DType::kFloat32, features.ref_pos.data());
    DeviceTensor& ref_charge = uploads.add({1, a}, DType::kFloat32, features.ref_charge.data());
    DeviceTensor& atom_mask = uploads.add({1, a}, DType::kFloat32, features.atom_mask.data());
    DeviceTensor& ref_element = uploads.add({1, a}, DType::kInt32, features.ref_element.data());
    DeviceTensor& ref_name_chars =
        uploads.add({1, a, 4}, DType::kInt32, features.ref_name_chars.data());
    DeviceTensor& ref_space_uid = uploads.add({1, a}, DType::kInt32, features.ref_space_uid.data());
    DeviceTensor& atom_to_token = uploads.add({1, a}, DType::kInt32, features.atom_to_token.data());
    DeviceTensor& intra_index = uploads.add({1, a}, DType::kInt32, features.intra_index.data());
    auto bindAtoms = [&](ITrtModule& module, bool with_token_map) {
        bindInput(module, "ref_pos", ref_pos);
        bindInput(module, "ref_charge", ref_charge);
        bindInput(module, "atom_mask", atom_mask);
        bindInput(module, "ref_element", ref_element);
        bindInput(module, "ref_name_chars", ref_name_chars);
        bindInput(module, "ref_space_uid", ref_space_uid);
        if (with_token_map)
            bindInput(module, "atom_to_token", atom_to_token);
    };

    // ESMC and the LM layer mix; the embedder reads the sequence rows (after BOS).
    ITrtModule& esmc = *engines_.esmc;
    bindInput(esmc, "input_ids", input_ids);
    esmc.forward_device_async({});
    auto* lm_single = static_cast<std::uint8_t*>(output(esmc, "lm_single")) + c * compute_bytes;

    ITrtModule& embed = *engines_.embed;
    bindInput(embed, "lm_single", lm_single, {1, n, c});
    bindInput(embed, "res_type", res_type);
    bindInput(embed, "residue_index", residue_index);
    bindInput(embed, "token_index", token_index);
    bindInput(embed, "asym_id", asym_id);
    bindInput(embed, "sym_id", sym_id);
    bindInput(embed, "entity_id", entity_id);
    bindAtoms(embed, true);
    embed.forward_device_async({});
    void* single_inputs = output(embed, "single_inputs");
    void* initial_pair = output(embed, "initial_pair");
    void* relpos = output(embed, "relpos");
    void* lm_pair = output(embed, "lm_pair");
    const Shape pair_shape{1, n, n, c};
    const Shape single_shape{1, n, config_.single_inputs_size};

    void* base_pair = initial_pair;
    if (engines_.msa) {
        ITrtModule& msa = *engines_.msa;
        bindInput(msa, "initial_pair", initial_pair, pair_shape);
        bindInput(msa, "single_inputs", single_inputs, single_shape);
        bindInput(msa, "res_type", res_type);
        msa.forward_device_async({});
        base_pair = output(msa, "msa_pair");
    }

    // Trunk: num_loops + 1 iterations of the parcae recurrence.
    const RandomStreams random(controls.seed);
    const auto pair_elements = static_cast<std::size_t>(elements(pair_shape));
    const auto initial_state = pairStateBytes(
        random, pair_elements, static_cast<float>(config_.trunk_init_std), config_.bf16);
    DeviceTensor& state = uploads.add(pair_shape, compute, initial_state.data());
    DeviceTensor& keep = uploads.add(pair_shape, DType::kInt32, nullptr);
    std::vector<std::int32_t> keep_host(pair_elements);
    const std::uint32_t threshold = keep_threshold(config_.lm_dropout);
    ITrtModule& loop = *engines_.loop;
    for (std::int32_t iteration = 0; iteration <= controls.loops; ++iteration) {
        keepMask(random, static_cast<std::uint64_t>(iteration) * pair_elements, threshold,
                 keep_host);
        check(cudaMemcpyAsync(keep.data(), keep_host.data(), keep.nbytes(), cudaMemcpyHostToDevice,
                              stream_),
              "upload LM dropout mask");
        bindInput(loop, "state", state);
        bindInput(loop, "base_pair", base_pair, pair_shape);
        bindInput(loop, "lm_pair", lm_pair, pair_shape);
        bindInput(loop, "lm_keep", keep);
        loop.forward_device_async({});
        check(cudaMemcpyAsync(state.data(), output(loop, "next_state"), state.nbytes(),
                              cudaMemcpyDeviceToDevice, stream_),
              "carry the trunk state");
        // The host mask buffer is rewritten next iteration.
        check(cudaStreamSynchronize(stream_), "trunk iteration");
    }

    ITrtModule& decode = *engines_.decode;
    bindInput(decode, "state", state);
    decode.forward_device_async({});
    void* pair_trunk = output(decode, "pair_trunk");

    ITrtModule& conditioning = *engines_.conditioning;
    bindInput(conditioning, "pair_trunk", pair_trunk, pair_shape);
    bindInput(conditioning, "relpos", relpos, pair_shape);
    bindInput(conditioning, "single_inputs", single_inputs, single_shape);
    bindAtoms(conditioning, false);
    conditioning.forward_device_async({});

    // Diffusion sampling.
    const NoiseSchedule schedule = noise_schedule(config_.sampler, controls.configured_steps);
    const auto coordinate_count = static_cast<std::size_t>(s * a * 3);
    std::vector<float> coords(coordinate_count);
    std::vector<float> draws(coordinate_count);
    random.fill_normal(Stream::kCoordinates, 0, draws);
    for (std::size_t index = 0; index < coordinate_count; ++index)
        coords[index] = schedule.sigmas.front() * draws[index];
    std::vector<float> sample_mask;
    sample_mask.reserve(static_cast<std::size_t>(a));
    sample_mask.assign(features.atom_mask.begin(), features.atom_mask.end());
    ITrtModule& denoise = *engines_.denoise;
    const Shape token_bias_shape{config_.diffusion_layers, config_.diffusion_heads, n, n};
    bindInput(denoise, "token_bias", output(conditioning, "token_bias"), token_bias_shape);
    bindInput(denoise, "cond_single", output(conditioning, "cond_single"),
              {1, n, config_.diffusion_hidden_size});
    bindInput(denoise, "atom_embeds", output(conditioning, "atom_embeds"),
              {1, a, config_.atom_hidden_size});
    bindInput(denoise, "rope_cos", output(conditioning, "rope_cos"), {1, a, config_.atom_head_dim});
    bindInput(denoise, "rope_sin", output(conditioning, "rope_sin"), {1, a, config_.atom_head_dim});
    bindInput(denoise, "atom_mask", atom_mask);
    bindInput(denoise, "atom_to_token", atom_to_token);
    DeviceTensor& noisy_device = uploads.add({s, a, 3}, DType::kFloat32, nullptr);
    DeviceTensor& level_device = uploads.add({s, 1}, DType::kFloat32, nullptr);
    std::vector<float> noisy(coordinate_count);
    std::vector<float> denoised(coordinate_count);
    std::vector<float> levels(static_cast<std::size_t>(s));
    std::vector<Mat3> rotations;
    std::vector<float> translations;
    for (std::int32_t step = 0; step < schedule.steps(); ++step) {
        step_rigid_draws(random, step, static_cast<std::int32_t>(s), rotations, translations);
        center_random_augmentation(coords, static_cast<std::int32_t>(s),
                                   static_cast<std::int32_t>(a), sample_mask, rotations,
                                   translations);
        const StepLevels step_level = step_levels(config_.sampler, schedule, step);
        noisy = coords;
        if (step_level.churn_std > 0.0) {
            random.fill_normal(Stream::kChurn, static_cast<std::uint64_t>(step) * coordinate_count,
                               draws);
            const auto churn = static_cast<float>(step_level.churn_std);
            for (std::size_t index = 0; index < coordinate_count; ++index)
                noisy[index] = coords[index] + churn * draws[index];
        }
        const auto noise_level = static_cast<float>(step_level.noise_level);
        std::fill(levels.begin(), levels.end(), noise_level);
        check(cudaMemcpyAsync(noisy_device.data(), noisy.data(), noisy_device.nbytes(),
                              cudaMemcpyHostToDevice, stream_),
              "upload noisy coordinates");
        check(cudaMemcpyAsync(level_device.data(), levels.data(), level_device.nbytes(),
                              cudaMemcpyHostToDevice, stream_),
              "upload noise level");
        bindInput(denoise, "noisy_coords", noisy_device);
        bindInput(denoise, "noise_level", level_device);
        denoise.forward_device_async({});
        check(cudaMemcpyAsync(denoised.data(), output(denoise, "denoised"),
                              coordinate_count * sizeof(float), cudaMemcpyDeviceToHost, stream_),
              "download denoised coordinates");
        check(cudaStreamSynchronize(stream_), "denoising step");
        weighted_rigid_align(noisy, denoised, static_cast<std::int32_t>(s),
                             static_cast<std::int32_t>(a), sample_mask);
        const auto factor = static_cast<float>(config_.sampler.step_scale *
                                               (step_level.sigma_next - step_level.noise_level));
        for (std::size_t index = 0; index < coordinate_count; ++index) {
            const float direction = (noisy[index] - denoised[index]) / noise_level;
            coords[index] = noisy[index] + factor * direction;
        }
    }

    // Confidence of every sample.
    ITrtModule& confidence = *engines_.confidence;
    const std::vector<float> tm = tm_per_bin(static_cast<std::int32_t>(n), config_.pae_bins);
    DeviceTensor& tm_device = uploads.add({config_.pae_bins}, DType::kFloat32, tm.data());
    DeviceTensor& sample_coords = uploads.add({1, a, 3}, DType::kFloat32, nullptr);
    bindInput(confidence, "single_inputs", single_inputs, single_shape);
    bindInput(confidence, "pair_trunk", pair_trunk, pair_shape);
    bindInput(confidence, "relpos", relpos, pair_shape);
    bindInput(confidence, "rep_atom", rep_atom);
    bindInput(confidence, "atom_to_token", atom_to_token);
    bindInput(confidence, "intra_index", intra_index);
    confidence.bind_external("tm_per_bin", tm_device.data());
    std::vector<SampleConfidence> confidences;
    std::vector<float> plddt_atom(static_cast<std::size_t>(a));
    std::vector<float> ptm_rows(static_cast<std::size_t>(n));
    const auto sample_floats = static_cast<std::size_t>(a * 3);
    for (std::int64_t sample = 0; sample < s; ++sample) {
        check(cudaMemcpyAsync(sample_coords.data(), coords.data() + sample * a * 3,
                              sample_floats * sizeof(float), cudaMemcpyHostToDevice, stream_),
              "upload sample coordinates");
        bindInput(confidence, "coords", sample_coords);
        confidence.forward_device_async({});
        check(cudaMemcpyAsync(plddt_atom.data(), output(confidence, "plddt_atom"),
                              plddt_atom.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_),
              "download pLDDT");
        check(cudaMemcpyAsync(ptm_rows.data(), output(confidence, "ptm_rows"),
                              ptm_rows.size() * sizeof(float), cudaMemcpyDeviceToHost, stream_),
              "download pTM rows");
        check(cudaStreamSynchronize(stream_), "confidence head");
        confidences.push_back(
            aggregate_confidence(features, plddt_atom, ptm_rows, config_.confidence_eps));
    }

    StructurePredictionResult result;
    const auto order = rank_samples(confidences);
    for (std::size_t rank = 0; rank < order.size(); ++rank) {
        const auto sample = static_cast<std::size_t>(order[rank]);
        const SampleConfidence& scores = confidences[sample];
        const float* sample_data = coords.data() + sample * sample_floats;
        StructurePredictionSample entry;
        entry.format = controls.format;
        entry.structure = controls.format == StructureFormat::kPdb
                              ? write_pdb(features, sample_data, scores.plddt)
                              : write_mmcif(features, sample_data, scores.plddt);
        entry.confidence.confidence_score = scores.ptm;
        entry.confidence.ptm = scores.ptm;
        entry.confidence.complex_plddt = scores.complex_plddt;
        entry.confidence.plddt = scores.plddt;
        entry.metadata_json = metadataJson(protein, features, scores, controls, schedule.steps(),
                                           static_cast<std::int32_t>(sample),
                                           static_cast<std::int32_t>(rank), config_.bf16);
        if (rank == 0) {
            result.structure = entry.structure;
            result.format = entry.format;
            result.confidence = entry.confidence;
            result.metadata_json = entry.metadata_json;
        }
        if (s > 1)
            result.samples.push_back(std::move(entry));
    }
    return result;
}

} // namespace trtmc::esmfold2
