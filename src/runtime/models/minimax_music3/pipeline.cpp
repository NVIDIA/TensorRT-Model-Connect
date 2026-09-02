/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime/models/minimax_music3/pipeline.h"

#include "runtime/models/minimax_music3/prompt_format.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime_api.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace trtmc {
namespace {

// Engine tensor names. These mirror the Python builders, which are the source
// of truth: language_model_engine.py, depth_decoder_engine.py, dit_builder.py,
// condition_encoder_builder.py and vocoder_builder.py.
constexpr const char* kTokenInput = "token_id";
constexpr const char* kEmbedInput = "input_embed";
constexpr const char* kUseEmbedInput = "use_input_embed";
constexpr const char* kPositionInput = "position_id";
constexpr const char* kAttentionMaskInput = "attention_mask";
constexpr const char* kLogitsOutput = "logits";
// Singular: the standard decoder marks this output "hidden_state"
// (default_decoder.py:471), not the plural the IoMap fields use.
constexpr const char* kHiddenStatesOutput = "hidden_state";

constexpr const char* kDepthHiddenInput = "lm_hidden";
constexpr const char* kDepthCodesInput = "codes";
constexpr const char* kDepthHiddenOutput = "depth_hidden";
constexpr const char* kFrameEmbedOutput = "frame_embed";

constexpr const char* kConditionInput = "hidden_states";
constexpr const char* kConditionOutput = "condition";

constexpr const char* kLatentsInput = "latents";
constexpr const char* kDitConditionInput = "condition";
// The engine embeds the time itself, so this is the scalar, not the
// 2048-wide prefix an earlier revision passed here.
constexpr const char* kTimestepInput = "timestep";
constexpr const char* kVelocityOutput = "velocity";

constexpr const char* kVocoderInput = "latents";

// What a masked position scores. Matches the repository's decoders,
// which use -1e4 rather than -inf so the softmax stays finite.
constexpr float kMaskedScore = -1.0e4F;
constexpr const char* kWaveformOutput = "waveform";

std::string layer_name(const std::string& pattern, int32_t layer) {
    const auto marker = pattern.find("{i}");
    if (marker == std::string::npos)
        return pattern;
    return pattern.substr(0, marker) + std::to_string(layer) + pattern.substr(marker + 3);
}

// The schedule the scheduler actually runs.
//
// The pipeline hands it a linear ramp from 1 down to 1/steps, but the
// checkpoint's scheduler_config.json sets "invert_sigmas": true, so
// FlowMatchEulerDiscreteScheduler stores 1 - sigma and appends a trailing 1.0
// rather than a trailing 0. With "num_train_timesteps": 1 the timesteps are
// those inverted sigmas unscaled, which is what makes the reference's comment
// -- "flow-matching time in [0, 1], 0 = noise" -- true, and what makes the
// Euler step advance rather than retreat.
//
// Returns steps + 1 values: the timesteps, then the 1.0 the last step moves to.
std::vector<float> sigma_schedule(int32_t steps) {
    if (steps < 1)
        throw std::invalid_argument("num_inference_steps must be at least 1");
    std::vector<float> sigmas;
    sigmas.reserve(static_cast<std::size_t>(steps) + 1);
    if (steps == 1) {
        sigmas.push_back(0.0F);
    } else {
        const double last = 1.0 / static_cast<double>(steps);
        const double step = (last - 1.0) / static_cast<double>(steps - 1);
        for (int32_t index = 0; index < steps; ++index) {
            const double ramp = 1.0 + step * static_cast<double>(index);
            sigmas.push_back(static_cast<float>(1.0 - ramp));
        }
    }
    sigmas.push_back(1.0F);
    return sigmas;
}

// Mirrors pipeline_spec.chunk_starts: one window when the whole generation
// fits, otherwise a window every hop up to but not including the last partial
// hop.
std::vector<int32_t> chunk_starts(int32_t frames, int32_t chunk_frames, int32_t chunk_hop) {
    if (frames <= 0)
        throw std::invalid_argument("frame count must be positive");
    if (frames <= chunk_frames)
        return {0};
    std::vector<int32_t> starts;
    for (int32_t start = 0; start < frames - chunk_hop; start += chunk_hop)
        starts.push_back(start);
    return starts;
}

int32_t argmax(const float* values, int32_t count) {
    int32_t best = 0;
    for (int32_t index = 1; index < count; ++index) {
        if (values[index] > values[best])
            best = index;
    }
    return best;
}

// Temperature plus top-k, the subset of the sampling contract this model
// needs. A top_k of 1 or a temperature of 0 collapses to the argmax, which is
// what the parity runs used.
int32_t sample(const float* logits, int32_t count, const GenerateConfig& cfg,
               std::mt19937_64& rng) {
    if (cfg.top_k == 1 || cfg.temperature <= 0.0F)
        return argmax(logits, count);

    std::vector<int32_t> order(static_cast<std::size_t>(count));
    for (int32_t index = 0; index < count; ++index)
        order[static_cast<std::size_t>(index)] = index;
    int32_t keep = count;
    if (cfg.top_k > 1 && cfg.top_k < count) {
        keep = cfg.top_k;
        std::partial_sort(order.begin(), order.begin() + keep, order.end(),
                          [logits](int32_t a, int32_t b) { return logits[a] > logits[b]; });
    }

    const float scale = 1.0F / cfg.temperature;
    float peak = logits[order[0]];
    for (int32_t index = 1; index < keep; ++index)
        peak = std::max(peak, logits[order[static_cast<std::size_t>(index)]]);

    std::vector<float> weights(static_cast<std::size_t>(keep));
    float total = 0.0F;
    for (int32_t index = 0; index < keep; ++index) {
        const float value =
            std::exp((logits[order[static_cast<std::size_t>(index)]] - peak) * scale);
        weights[static_cast<std::size_t>(index)] = value;
        total += value;
    }
    std::uniform_real_distribution<float> uniform(0.0F, total);
    float target = uniform(rng);
    for (int32_t index = 0; index < keep; ++index) {
        target -= weights[static_cast<std::size_t>(index)];
        if (target <= 0.0F)
            return order[static_cast<std::size_t>(index)];
    }
    return order[static_cast<std::size_t>(keep - 1)];
}

// Draw one audio code. Only the semantic block and the end token are legal:
// the head scores the whole 200000-token vocabulary, and an unmasked draw can
// land on ordinary text, which generates audio carrying no words at all.
int32_t sample_audio_code(const float* logits, int32_t vocab, const GenerateConfig& cfg,
                          std::mt19937_64& rng) {
    const int32_t first = minimax_music3::kAudioCodeOffset;
    const int32_t last = std::min(vocab, first + minimax_music3::kSemanticVocabSize);
    if (first >= last)
        throw std::runtime_error("MiniMax-Music3 audio code range falls outside the vocabulary");

    std::vector<float> allowed(static_cast<std::size_t>(last - first) + 1);
    std::copy(logits + first, logits + last, allowed.begin());
    allowed.back() = logits[minimax_music3::kAudioEndTokenId];

    const int32_t drawn = sample(allowed.data(), static_cast<int32_t>(allowed.size()), cfg, rng);
    return drawn == static_cast<int32_t>(allowed.size()) - 1 ? minimax_music3::kAudioEndTokenId
                                                             : first + drawn;
}

// Widen one engine output into floats. A bf16 build leaves the hidden states
// and the key/value cache at half width, and reading those bits as float32
// gives values around 1e29 rather than the 0.94 the reference produces.
void widen_into(const Tensor& tensor, std::vector<float>& out) {
    const auto count = tensor.numel();
    out.resize(count);
    if (tensor.dtype == DType::kFloat32) {
        std::memcpy(out.data(), tensor.data, count * sizeof(float));
        return;
    }
    if (tensor.dtype != DType::kBFloat16)
        throw std::runtime_error("MiniMax-Music3 cannot widen this engine output");
    // bfloat16 is the top half of a float32, so widening is a shift.
    const auto* halves = static_cast<const std::uint16_t*>(tensor.data);
    for (std::size_t index = 0; index < count; ++index) {
        const std::uint32_t bits = static_cast<std::uint32_t>(halves[index]) << 16;
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        out[index] = value;
    }
}

// Per-stage statistics, printed when TRTMC_MM3_DEBUG is set. parity.py records
// what the reference produces at the same points -- frame hidden states at a
// standard deviation of 0.9412, window latents near 2.4 -- so a stage that has
// drifted shows up here rather than only in the audio.
void report_stage_count(const char* stage, std::size_t count) {
    if (std::getenv("TRTMC_MM3_DEBUG") == nullptr)
        return;
    std::cerr << "[mm3] " << stage << " = " << count << '\n';
}

void report_stage(const char* stage, const float* values, std::size_t count) {
    if (count == 0 || std::getenv("TRTMC_MM3_DEBUG") == nullptr)
        return;
    double total = 0.0;
    double square = 0.0;
    float smallest = values[0];
    float largest = values[0];
    for (std::size_t index = 0; index < count; ++index) {
        const double value = values[index];
        total += value;
        square += value * value;
        smallest = std::min(smallest, values[index]);
        largest = std::max(largest, values[index]);
    }
    const double mean = total / static_cast<double>(count);
    const double variance = square / static_cast<double>(count) - mean * mean;
    std::cerr << "[mm3] " << stage << " n=" << count << " mean=" << mean
              << " std=" << std::sqrt(std::max(0.0, variance)) << " min=" << smallest
              << " max=" << largest << '\n';
}

} // namespace

MinimaxMusic3TextToMusicPipeline::MinimaxMusic3TextToMusicPipeline(
    MinimaxMusic3Engines engines, MinimaxMusic3Config config, std::shared_ptr<ITokenizer> tokenizer,
    std::string model_id)
    : engines_(std::move(engines)), config_(std::move(config)), tokenizer_(std::move(tokenizer)),
      model_id_(std::move(model_id)) {
    if (!engines_.language_model || !engines_.depth_decoder || !engines_.condition_encoder ||
        !engines_.dit || !engines_.vocoder)
        throw std::runtime_error("MiniMax-Music3 needs all five engines");
    bind_cache();
}

MinimaxMusic3TextToMusicPipeline::~MinimaxMusic3TextToMusicPipeline() = default;

void MinimaxMusic3TextToMusicPipeline::bind_cache() {
    auto& lm = *engines_.language_model;
    const auto stream = lm.stream();
    const auto slots = static_cast<std::size_t>(config_.language_model_layers) *
                       static_cast<std::size_t>(kBranches);
    cache_k_.resize(slots);
    cache_v_.resize(slots);
    present_k_.resize(slots);
    present_v_.resize(slots);
    hidden_scratch_.resize(static_cast<std::size_t>(kBranches));
    logits_scratch_.resize(static_cast<std::size_t>(kBranches));

    // The engine knows the cache geometry it was compiled for; asking it keeps
    // this from re-deriving a shape the builder already fixed.
    for (int32_t layer = 0; layer < config_.language_model_layers; ++layer) {
        for (int32_t branch = 0; branch < kBranches; ++branch) {
            const auto index = slot(branch, layer);
            const auto key = layer_name("cache_k_{i}", layer);
            const auto value = layer_name("cache_v_{i}", layer);
            if (!lm.has_input(key) || !lm.has_input(value))
                throw std::runtime_error("language model engine is missing " + key);
            const auto shape = lm.tensor_shape(key);
            // The engine decides the width. A bf16 build carries a bf16 cache, and
            // a float32 buffer would be twice the size and read as noise.
            const auto dtype = lm.tensor_dtype(key);
            cache_k_[index] = DeviceTensor(shape, dtype, stream);
            cache_v_[index] = DeviceTensor(shape, dtype, stream);
            present_k_[index] = DeviceTensor(shape, dtype, stream);
            present_v_[index] = DeviceTensor(shape, dtype, stream);

            // Device memory arrives with whatever was in it. The mask should keep
            // unwritten slots out of the scores, but zeroing costs one memset per
            // layer and removes the question.
            std::size_t elements = 1;
            for (const auto extent : shape)
                elements *= static_cast<std::size_t>(extent);
            const std::vector<char> zeros(elements * dtype_size(dtype), 0);
            cache_k_[index].copy_from_host(zeros.data());
            cache_v_[index].copy_from_host(zeros.data());
            present_k_[index].copy_from_host(zeros.data());
            present_v_[index].copy_from_host(zeros.data());
        }
    }
    bind_branch(kConditional);
}

void MinimaxMusic3TextToMusicPipeline::guide_logits(const float* conditional,
                                                    const float* unconditional,
                                                    std::vector<float>& out) const {
    const auto vocab = static_cast<std::size_t>(config_.language_model_vocab_size);
    out.resize(vocab);

    // The top-k floor is taken over the audio codes only. The reference masks
    // the vocabulary before ranking, and the order matters: the head scores all
    // 200000 tokens, so a ranking over the whole vocabulary can be filled with
    // ordinary text and leave the semantic block almost entirely below the
    // threshold. The draw that follows then picks from what little survives,
    // which produces frame states of the right magnitude carrying the wrong
    // content -- audio that sings without saying anything.
    const auto first = static_cast<std::size_t>(minimax_music3::kAudioCodeOffset);
    const auto last =
        std::min(vocab, first + static_cast<std::size_t>(minimax_music3::kSemanticVocabSize));
    std::vector<float> ranked(conditional + first, conditional + last);
    ranked.push_back(conditional[minimax_music3::kAudioEndTokenId]);
    const int32_t keep = std::min(minimax_music3::kArCfgTopK, static_cast<int32_t>(ranked.size()));
    std::nth_element(ranked.begin(), ranked.begin() + (keep - 1), ranked.end(),
                     std::greater<float>());
    const float threshold = ranked[static_cast<std::size_t>(keep - 1)];

    for (std::size_t index = 0; index < vocab; ++index) {
        if (conditional[index] < threshold) {
            out[index] = -std::numeric_limits<float>::infinity();
            continue;
        }
        out[index] = unconditional[index] +
                     (conditional[index] - unconditional[index]) * minimax_music3::kArCfgScale;
    }
}

std::size_t MinimaxMusic3TextToMusicPipeline::slot(int32_t branch, int32_t layer) const {
    return static_cast<std::size_t>(layer) * static_cast<std::size_t>(kBranches) +
           static_cast<std::size_t>(branch);
}

void MinimaxMusic3TextToMusicPipeline::bind_branch(int32_t branch) {
    auto& lm = *engines_.language_model;
    for (int32_t layer = 0; layer < config_.language_model_layers; ++layer) {
        const auto index = slot(branch, layer);
        lm.bind_external(layer_name("cache_k_{i}", layer), cache_k_[index].data());
        lm.bind_external(layer_name("cache_v_{i}", layer), cache_v_[index].data());
        lm.bind_external(layer_name("present_k_{i}", layer), present_k_[index].data());
        lm.bind_external(layer_name("present_v_{i}", layer), present_v_[index].data());
    }
}

void MinimaxMusic3TextToMusicPipeline::commit_branch(int32_t branch, int32_t position) {
    // present_* is one row -- the key and value for the token just decoded --
    // not an updated cache. The runtime owns the history: copy that row into
    // cache row `position`.
    //
    // Swapping the buffers instead, as an earlier revision did, replaced the
    // whole cache with a single row every step. The model then saw only the
    // token in front of it, and sang one syllable over and over.
    auto& lm = *engines_.language_model;
    const auto row_bytes = static_cast<std::size_t>(config_.language_model_kv_width) *
                           dtype_size(lm.tensor_dtype(layer_name("cache_k_{i}", 0)));
    for (int32_t layer = 0; layer < config_.language_model_layers; ++layer) {
        const auto index = slot(branch, layer);
        const auto offset = static_cast<std::size_t>(position) * row_bytes;
        cudaMemcpyAsync(static_cast<char*>(cache_k_[index].data()) + offset,
                        present_k_[index].data(), row_bytes, cudaMemcpyDeviceToDevice, lm.stream());
        cudaMemcpyAsync(static_cast<char*>(cache_v_[index].data()) + offset,
                        present_v_[index].data(), row_bytes, cudaMemcpyDeviceToDevice, lm.stream());
    }
    cudaStreamSynchronize(lm.stream());
}

int32_t MinimaxMusic3TextToMusicPipeline::latent_length_for(int32_t frames) const {
    if (frames < 1)
        throw std::invalid_argument("frame count must be positive");
    if (frames == config_.chunk_frames)
        return config_.chunk_latent_length;
    // Truncating rather than rounding is what the reference does, and it is
    // why a 200-frame window is 689 latent frames and not 690.
    const auto latent = static_cast<double>(frames) * config_.latent_resample_ratio;
    return std::max(1, static_cast<int32_t>(latent));
}

std::vector<int32_t>
MinimaxMusic3TextToMusicPipeline::tokenize_prompt(const std::string& lyrics) const {
    // The model is not given raw lyrics. The caption and the lyrics go inside
    // the checkpoint's structure tokens, ending at <|audio_start|> so the first
    // generated token is audio -- see prompt_format, which was checked against
    // the reference over 410 captions and 409 lyric strings.
    const auto assembled = minimax_music3::assemble_prompt(config_.caption, lyrics);
    auto ids = tokenizer_->encode(assembled);
    if (static_cast<int32_t>(ids.size()) > minimax_music3::kMaxPromptTokens)
        throw std::runtime_error("MiniMax-Music3 prompt is longer than the checkpoint's budget");

    report_stage_count("prompt_tokens", ids.size());
    if (std::getenv("TRTMC_MM3_DEBUG") != nullptr) {
        std::cerr << "[mm3] prompt_ids";
        for (const auto id : ids)
            std::cerr << ' ' << id;
        std::cerr << '\n';
    }
    return ids;
}

AudioResult MinimaxMusic3TextToMusicPipeline::generate_audio(const std::string& prompt,
                                                             const GenerateConfig& cfg) {
    if (!tokenizer_)
        throw std::runtime_error("MiniMax-Music3 needs a tokenizer to read its prompt");

    const auto prompt_ids = tokenize_prompt(prompt);
    const int32_t frames = std::min(
        config_.max_frames, cfg.max_new_tokens > 0 ? cfg.max_new_tokens : config_.max_audio_frames);

    std::vector<float> frame_hidden;
    if (const char* injected = std::getenv("TRTMC_MM3_FRAME_HIDDEN")) {
        // Bisection aid: run the conditioning, the denoiser and the vocoder on
        // frame states produced elsewhere. It separates a fault in how the
        // codes are drawn from a fault in what is done with them.
        std::ifstream in(injected, std::ios::binary | std::ios::ate);
        if (!in)
            throw std::runtime_error("cannot read TRTMC_MM3_FRAME_HIDDEN");
        const auto bytes = static_cast<std::size_t>(in.tellg());
        in.seekg(0);
        frame_hidden.resize(bytes / sizeof(float));
        in.read(reinterpret_cast<char*>(frame_hidden.data()), static_cast<std::streamsize>(bytes));
        std::cerr << "[mm3] injected frame states: " << frame_hidden.size() << " floats\n";
    } else {
        generate_codes(prompt_ids, frames, cfg, frame_hidden);
    }
    report_stage("frame_hidden", frame_hidden.data(), frame_hidden.size());
    if (const char* dump = std::getenv("TRTMC_MM3_DUMP")) {
        // Raw float32, frame-major, for diffing against the reference's
        // frame_hiddens without a format in the way.
        std::ofstream out(dump, std::ios::binary);
        out.write(reinterpret_cast<const char*>(frame_hidden.data()),
                  static_cast<std::streamsize>(frame_hidden.size() * sizeof(float)));
    }

    const int32_t steps = cfg.num_steps > 0 ? cfg.num_steps : config_.default_inference_steps;
    const auto starts = chunk_starts(frames, config_.chunk_frames, config_.chunk_hop);
    const int32_t latent_length = latent_length_for(config_.chunk_frames);

    std::vector<float> samples;
    std::vector<float> previous;
    for (std::size_t window = 0; window < starts.size(); ++window) {
        const auto condition = encode_condition(frame_hidden, starts[window], config_.chunk_frames);
        report_stage("condition", condition.data(), condition.size());
        const auto latents = denoise_window(condition, latent_length, steps,
                                            static_cast<uint64_t>(cfg.seed) + window, previous);

        // Carry [L - 2h, L - h) to the next window, which is the reference's
        // slice: the frames the neighbour will blend its own head toward.
        {
            const auto channels = static_cast<std::size_t>(config_.latent_channels);
            const auto start =
                static_cast<std::size_t>(std::max(0, latent_length - 2 * kOverlapCarry));
            const auto stop = static_cast<std::size_t>(
                std::max(static_cast<int32_t>(start), latent_length - kOverlapCarry));
            std::vector<float> next_previous(channels * (stop - start));
            for (std::size_t channel = 0; channel < channels; ++channel) {
                std::copy(
                    latents.begin() +
                        static_cast<std::ptrdiff_t>(
                            channel * static_cast<std::size_t>(latent_length) + start),
                    latents.begin() + static_cast<std::ptrdiff_t>(
                                          channel * static_cast<std::size_t>(latent_length) + stop),
                    next_previous.begin() + static_cast<std::ptrdiff_t>(channel * (stop - start)));
            }
            previous = std::move(next_previous);
        }
        report_stage("latents", latents.data(), latents.size());
        auto chunk = decode_waveform(latents, latent_length);
        report_stage("waveform", chunk.data(), chunk.size());

        // Every window but the first drops crop_left and every window but the
        // last drops crop_right, so the two crops remove exactly one overlap
        // per seam. The widths are latent frames, so they scale by the hop.
        const int32_t left = window == 0 ? 0 : config_.crop_left_latent;
        const int32_t right = window + 1 == starts.size() ? 0 : config_.crop_right_latent;
        const auto begin =
            static_cast<std::size_t>(left) * config_.latent_hop_length * config_.output_channels;
        const auto end = chunk.size() - static_cast<std::size_t>(right) *
                                            config_.latent_hop_length * config_.output_channels;
        if (begin >= end)
            throw std::runtime_error("MiniMax-Music3 window crop removed the whole window");
        samples.insert(samples.end(), chunk.begin() + static_cast<std::ptrdiff_t>(begin),
                       chunk.begin() + static_cast<std::ptrdiff_t>(end));
    }

    // Crop only what the padding added. A generation shorter than one window
    // is zero-filled up to chunk_frames, and without this it would report a
    // duration it did not produce.
    //
    // A generation of at least one window must not be touched: the stitched
    // length is the sum of the cropped windows -- 431 + 345 + 345 + 603 latent
    // frames for the four-window plan -- which is 1724, while the frame count
    // resolves to 1722. Cropping to the latter drops two latent frames of real
    // audio off the end.
    if (frames < config_.chunk_frames) {
        const auto true_samples = static_cast<std::size_t>(latent_length_for(frames)) *
                                  static_cast<std::size_t>(config_.latent_hop_length) *
                                  static_cast<std::size_t>(config_.output_channels);
        if (true_samples > 0 && true_samples < samples.size())
            samples.resize(true_samples);
    }

    AudioResult result;
    result.sample_rate = config_.sampling_rate;
    result.channels = config_.output_channels;
    result.num_samples =
        static_cast<int32_t>(samples.size() / static_cast<std::size_t>(config_.output_channels));
    result.samples = std::move(samples);
    return result;
}

const float* MinimaxMusic3TextToMusicPipeline::decode_step(int32_t branch, int32_t token_id,
                                                           int32_t position,
                                                           const float** hidden_out,
                                                           const float* frame_embed) {
    auto& lm = *engines_.language_model;
    bind_branch(branch);

    // The mask is additive, not multiplicative: add_2d_mask_to_4d takes scores
    // and a zero means "attend", so the masked positions carry a large negative
    // score rather than a zero. Filling it the other way round makes every
    // cache slot visible, including the ones no step has written yet, and the
    // hidden states come back at 1e35.
    // The mask spans the cache plus one: the decoder attends over
    // concat(cache, current), so the row for the token being decoded is the
    // last one, not row `position`. Cache rows [0, position) hold the tokens
    // already written; row `position` has not been written yet.
    //
    // Marking [0, position] visible instead -- as an earlier revision did --
    // exposes one unwritten row and hides the current token, so every step
    // attends to zeros and the model never sees what it is decoding.
    const auto mask_shape = lm.tensor_shape(kAttentionMaskInput);
    const auto mask_length = static_cast<std::size_t>(mask_shape.back());
    if (mask_length == 0)
        throw std::runtime_error("language model engine declares an empty attention mask");
    std::vector<float> mask(mask_length, kMaskedScore);
    const auto written = std::min<std::size_t>(static_cast<std::size_t>(position), mask_length - 1);
    std::fill(mask.begin(), mask.begin() + static_cast<std::ptrdiff_t>(written), 0.0F);
    mask.back() = 0.0F;

    TensorMap inputs;
    inputs[kTokenInput] = Tensor{&token_id, {1}, DType::kInt32};

    // After the prompt, the input is the frame's embedding rather than a token
    // id: one id carries one codebook of eight. The engine takes both paths and
    // use_input_embed selects between them.
    const auto width = static_cast<std::size_t>(config_.language_model_hidden_size);
    std::vector<float> embed(width, 0.0F);
    std::vector<float> use_embed(1, 0.0F);
    if (frame_embed != nullptr) {
        std::copy(frame_embed, frame_embed + width, embed.begin());
        use_embed[0] = 1.0F;
    }
    inputs[kEmbedInput] = Tensor{embed.data(), {1, static_cast<int64_t>(width)}, DType::kFloat32};
    inputs[kUseEmbedInput] = Tensor{use_embed.data(), {1}, DType::kFloat32};
    inputs[kPositionInput] = Tensor{&position, {1}, DType::kInt32};
    inputs[kAttentionMaskInput] =
        Tensor{mask.data(), {1, static_cast<int64_t>(mask_length)}, DType::kFloat32};

    auto outputs = lm.forward(inputs);
    const auto logits = outputs.find(kLogitsOutput);
    if (logits == outputs.end())
        throw std::runtime_error("language model engine produced no logits");
    const auto hidden = outputs.find(kHiddenStatesOutput);
    if (hidden == outputs.end())
        throw std::runtime_error("language model engine produced no hidden_state");
    auto& scratch = hidden_scratch_[static_cast<std::size_t>(branch)];
    widen_into(hidden->second, scratch);
    *hidden_out = scratch.data();

    // Copy the logits out before the next forward reuses the engine's output
    // buffer. Returning the pointer let the unconditional branch overwrite the
    // conditional one, so the two agreed exactly, guidance became a no-op, and
    // every draw came from the branch that conditions on the CFG token -- the
    // one with no lyrics in it.
    auto& scores = logits_scratch_[static_cast<std::size_t>(branch)];
    widen_into(logits->second, scores);

    commit_branch(branch, position);
    return scores.data();
}

void MinimaxMusic3TextToMusicPipeline::sample_residual_codes(const DepthStep& step,
                                                             const GenerateConfig& cfg,
                                                             uint64_t& rng_state) {
    const float* const unconditional_hidden = step.unconditional_hidden;
    const int32_t semantic_code = step.semantic_code;
    int32_t* const codes_out = step.codes_out;
    float* const depth_hidden_out = step.hidden_out;
    const float* frame_hidden = step.conditional_hidden;
    auto& depth = *engines_.depth_decoder;
    const int32_t residual = config_.num_residual_codebooks;

    // Codes are drawn one codebook at a time: each draw is an input to the
    // next. Positions not yet drawn are zero, and the causal mask inside the
    // graph is what keeps them from being read.
    std::vector<int32_t> codes(static_cast<std::size_t>(config_.num_codebooks), 0);
    // codes[0] is the frame's semantic code -- the one the language model drew.
    // The graph embeds it with the language model's own table; the rest are
    // residual codes that read their own blocks.
    codes[0] = semantic_code;
    std::mt19937_64 rng(rng_state);
    for (int32_t codebook = 0; codebook < residual; ++codebook) {
        TensorMap inputs;
        inputs[kDepthHiddenInput] = Tensor{const_cast<float*>(frame_hidden),
                                           {1, 1, config_.language_model_hidden_size},
                                           DType::kFloat32};
        inputs[kDepthCodesInput] = Tensor{codes.data(), {1, config_.num_codebooks}, DType::kInt32};
        auto outputs = depth.forward(inputs);
        const auto logits = outputs.find(kLogitsOutput);
        if (logits == outputs.end())
            throw std::runtime_error("depth decoder produced no logits");
        widen_into(logits->second, depth_conditional_);

        // The depth decoder is guided too, at the same scale as the language
        // model's draw.
        TensorMap unconditional_inputs = inputs;
        unconditional_inputs[kDepthHiddenInput] = Tensor{const_cast<float*>(unconditional_hidden),
                                                         {1, 1, config_.language_model_hidden_size},
                                                         DType::kFloat32};
        auto unconditional_outputs = depth.forward(unconditional_inputs);
        const auto unconditional_logits = unconditional_outputs.find(kLogitsOutput);
        if (unconditional_logits == unconditional_outputs.end())
            throw std::runtime_error("depth decoder produced no logits");
        widen_into(unconditional_logits->second, depth_unconditional_);

        const auto offset =
            static_cast<std::size_t>(codebook) * static_cast<std::size_t>(config_.audio_vocab_size);
        std::vector<float> guided(static_cast<std::size_t>(config_.audio_vocab_size));
        for (std::size_t index = 0; index < guided.size(); ++index) {
            const float conditional = depth_conditional_[offset + index];
            const float base = depth_unconditional_[offset + index];
            guided[index] = base + (conditional - base) * minimax_music3::kArCfgScale;
        }
        const int32_t drawn = sample(guided.data(), config_.audio_vocab_size, cfg, rng);
        codes_out[static_cast<std::size_t>(codebook)] = drawn;
        // Every draw lands in the code vector -- the frame embedding reads all
        // eight. Only the first six re-enter the sequence; the seventh has no
        // position after it.
        codes[static_cast<std::size_t>(codebook) + 1] = drawn;

        // The last pass carries the states the condition encoder reads. Only
        // the last one is kept: every earlier pass saw fewer codes, so its
        // states are of a shorter prefix than the frame the encoder wants.
        if (codebook + 1 == residual) {
            const auto depth = outputs.find(kDepthHiddenOutput);
            if (depth == outputs.end())
                throw std::runtime_error("depth decoder produced no depth_hidden");
            widen_into(depth->second, depth_scratch_);
            std::copy(depth_scratch_.begin(), depth_scratch_.end(), depth_hidden_out);
        }
    }
    // All eight codes are known now. One more pass reads the frame embedding
    // the language model takes as its next input; the tables live in this graph
    // so nothing else has to carry them.
    {
        TensorMap final_inputs;
        final_inputs[kDepthHiddenInput] = Tensor{const_cast<float*>(conditional_hidden),
                                                 {1, 1, config_.language_model_hidden_size},
                                                 DType::kFloat32};
        final_inputs[kDepthCodesInput] =
            Tensor{codes.data(), {1, config_.num_codebooks}, DType::kInt32};
        auto final_outputs = depth.forward(final_inputs);
        const auto embed = final_outputs.find(kFrameEmbedOutput);
        if (embed == final_outputs.end())
            throw std::runtime_error("depth decoder produced no frame_embed");
        widen_into(embed->second, frame_embed_);
    }

    rng_state = rng();
}

int32_t
MinimaxMusic3TextToMusicPipeline::prime_caches(const std::vector<int32_t>& prompt_ids,
                                               const std::vector<int32_t>& unconditional_ids,
                                               BranchState& state) {
    // Every prompt token is a decode step: the engine is compiled for one
    // position at a time, so there is no separate prefill profile to run. The
    // last step's logits and hidden state are the reference's `last_hidden` --
    // the prompt is not re-fed afterwards.
    int32_t position = 0;
    for (std::size_t index = 0; index < prompt_ids.size(); ++index) {
        state.conditional_logits =
            decode_step(kConditional, prompt_ids[index], position, &state.conditional_hidden);
        if (index == 0) {
            // The very first token runs against an empty cache, which
            // separates the per-layer arithmetic from the cache handling.
            report_stage("first_token_hidden", state.conditional_hidden,
                         static_cast<std::size_t>(config_.language_model_hidden_size));
        }
        state.unconditional_logits = decode_step(kUnconditional, unconditional_ids[index], position,
                                                 &state.unconditional_hidden);
        ++position;
    }
    return position;
}

void MinimaxMusic3TextToMusicPipeline::report_prompt_pass(const float* hidden,
                                                          const float* logits) const {
    // The prompt pass is deterministic on both sides, so this is where an
    // engine or a binding is compared against a reference without sampling in
    // the way. It is what located the attention mask fault and the aliased
    // logits.
    if (std::getenv("TRTMC_MM3_DEBUG") == nullptr)
        return;

    const auto width = static_cast<std::size_t>(config_.language_model_hidden_size);
    if (const char* dump = std::getenv("TRTMC_MM3_PROMPT_DUMP")) {
        std::ofstream out(dump, std::ios::binary);
        out.write(reinterpret_cast<const char*>(hidden),
                  static_cast<std::streamsize>(width * sizeof(float)));
    }
    report_stage("prompt_hidden", hidden, width);

    const auto vocab = static_cast<std::size_t>(config_.language_model_vocab_size);
    std::vector<std::size_t> order(vocab);
    for (std::size_t index = 0; index < vocab; ++index)
        order[index] = index;
    std::partial_sort(order.begin(), order.begin() + 5, order.end(),
                      [logits](std::size_t a, std::size_t b) { return logits[a] > logits[b]; });

    std::cerr << "[mm3] prompt_top5";
    for (int rank = 0; rank < 5; ++rank)
        std::cerr << ' ' << order[static_cast<std::size_t>(rank)] << '='
                  << logits[order[static_cast<std::size_t>(rank)]];
    std::cerr << '\n';
}

std::vector<int32_t>
MinimaxMusic3TextToMusicPipeline::build_unconditional_ids(const std::vector<int32_t>& prompt_ids) {
    // Every token but the first and the last two becomes the audio-CFG token,
    // mirroring the reference's unconditional_ids[:, 1:-2] = AUDIO_CFG_TOKEN_ID.
    std::vector<int32_t> ids = prompt_ids;
    if (ids.size() >= 4) {
        for (std::size_t index = 1; index + 2 < ids.size(); ++index)
            ids[index] = minimax_music3::kAudioCfgTokenId;
    }
    return ids;
}

void MinimaxMusic3TextToMusicPipeline::record_frame(const EmittedFrame& frame,
                                                    const std::vector<int32_t>& residual,
                                                    std::vector<float>& hidden,
                                                    std::vector<int32_t>& codes) const {
    // Eight streams per frame: the language model's hidden state first, then
    // the depth decoder's seven, which is the order the encoder's per-stream
    // weights were trained in. Codes are stored codebook-major, so one window
    // is a contiguous slice per stream.
    const auto stream_width =
        static_cast<std::size_t>(config_.frame_hidden_width / config_.condition_streams);
    const auto base = static_cast<std::size_t>(frame.index) *
                      static_cast<std::size_t>(config_.frame_hidden_width);

    std::copy(frame.hidden, frame.hidden + stream_width,
              hidden.begin() + static_cast<std::ptrdiff_t>(base));
    std::copy(depth_hidden_.begin(), depth_hidden_.end(),
              hidden.begin() + static_cast<std::ptrdiff_t>(base + stream_width));

    codes[static_cast<std::size_t>(frame.index)] = frame.semantic;
    for (int32_t stream = 0; stream < config_.num_residual_codebooks; ++stream) {
        codes[static_cast<std::size_t>(stream + 1) * static_cast<std::size_t>(frame.total) +
              static_cast<std::size_t>(frame.index)] = residual[static_cast<std::size_t>(stream)];
    }
}

std::vector<int32_t>
MinimaxMusic3TextToMusicPipeline::generate_codes(const std::vector<int32_t>& prompt_ids,
                                                 int32_t frames, const GenerateConfig& cfg,
                                                 std::vector<float>& hidden) {
    const int32_t streams = config_.num_codebooks;
    std::vector<int32_t> codes(static_cast<std::size_t>(streams) *
                               static_cast<std::size_t>(frames));
    // Eight streams per frame: the language model's hidden state first,
    // then the depth decoder's seven, which is the order the encoder's
    // per-stream weights were trained in.
    depth_hidden_.assign(static_cast<std::size_t>(config_.num_residual_codebooks) *
                             static_cast<std::size_t>(config_.language_model_hidden_size),
                         0.0F);
    const auto stream_width =
        static_cast<std::size_t>(config_.frame_hidden_width / config_.condition_streams);
    hidden.assign(static_cast<std::size_t>(frames) *
                      static_cast<std::size_t>(config_.frame_hidden_width),
                  0.0F);

    uint64_t rng_state = cfg.seed >= 0 ? static_cast<uint64_t>(cfg.seed) : 0x9E3779B97F4A7C15ULL;
    std::mt19937_64 rng(rng_state);

    const auto unconditional_ids = build_unconditional_ids(prompt_ids);

    // Prime both caches. Every prompt token is a decode step: the engine is
    // compiled for one position at a time, so there is no separate prefill
    // profile to run here. The last step's logits and hidden state are the
    // reference's `last_hidden` -- the prompt is not re-fed afterwards.
    const float* frame_hidden = nullptr;
    const float* unconditional_hidden = nullptr;
    const float* conditional = nullptr;
    const float* unconditional = nullptr;
    int32_t position = 0;
    BranchState state;
    position = prime_caches(prompt_ids, unconditional_ids, state);
    frame_hidden = state.conditional_hidden;
    unconditional_hidden = state.unconditional_hidden;
    conditional = state.conditional_logits;
    unconditional = state.unconditional_logits;
    if (conditional == nullptr)
        throw std::runtime_error("MiniMax-Music3 was given an empty prompt");

    report_prompt_pass(frame_hidden, conditional);

    GenerateConfig draw = cfg;
    if (draw.top_k <= 1)
        draw.top_k = minimax_music3::kArSamplingTopK;
    if (draw.temperature <= 0.0F)
        draw.temperature = 1.0F;

    // The reference runs max_frames + 1 draws and keeps the last max_frames of
    // them: the first advances the state past <|audio_start|> and is not a
    // frame the model emits.
    int32_t emitted = 0;
    std::vector<int32_t> residual(static_cast<std::size_t>(config_.num_residual_codebooks));
    for (int32_t step = 0; step <= frames; ++step) {
        guide_logits(conditional, unconditional, guided_);
        const int32_t sampled =
            sample_audio_code(guided_.data(), config_.language_model_vocab_size, draw, rng);
        if (sampled == minimax_music3::kAudioEndTokenId)
            break;

        const int32_t semantic = sampled - minimax_music3::kAudioCodeOffset;
        sample_residual_codes(DepthStep{frame_hidden, unconditional_hidden, semantic,
                                        residual.data(), depth_hidden_.data()},
                              draw, rng_state);

        if (step > 0) {
            record_frame(EmittedFrame{step - 1, frames, semantic, frame_hidden}, residual, hidden,
                         codes);
            ++emitted;
            if (emitted >= frames)
                break;
        }

        // The frame's embedding, not its first code, is what advances the state.
        conditional =
            decode_step(kConditional, sampled, position, &frame_hidden, frame_embed_.data());
        unconditional = decode_step(kUnconditional, sampled, position, &unconditional_hidden,
                                    frame_embed_.data());
        ++position;
    }
    if (std::getenv("TRTMC_MM3_DEBUG") != nullptr && emitted > 0) {
        // A degenerate loop shows up here before it shows up in the audio.
        std::cerr << "[mm3] semantic_codes";
        for (int32_t frame = 0; frame < std::min(emitted, 24); ++frame)
            std::cerr << ' ' << codes[static_cast<std::size_t>(frame)];
        std::cerr << '\n';
        std::vector<int32_t> seen(codes.begin(),
                                  codes.begin() + static_cast<std::ptrdiff_t>(emitted));
        std::sort(seen.begin(), seen.end());
        seen.erase(std::unique(seen.begin(), seen.end()), seen.end());
        std::cerr << "[mm3] distinct_semantic_codes = " << seen.size() << " of " << emitted << '\n';
    }
    report_stage_count("frames_emitted", static_cast<std::size_t>(emitted));
    if (emitted == 0)
        throw std::runtime_error("MiniMax-Music3 emitted no frames");
    return codes;
}

std::vector<float>
MinimaxMusic3TextToMusicPipeline::encode_condition(const std::vector<float>& frame_hidden,
                                                   int32_t frame_offset, int32_t frames) {
    auto& encoder = *engines_.condition_encoder;
    const auto width = static_cast<std::size_t>(config_.frame_hidden_width);
    const auto begin = static_cast<std::size_t>(frame_offset) * width;
    if (begin >= frame_hidden.size())
        throw std::runtime_error("condition window starts past the frames generated");

    // The engine's input is compiled for exactly chunk_frames, so a generation
    // shorter than one window -- or a last window that runs off the end -- is
    // zero-padded rather than reshaped. The padding decays to nothing in the
    // output because generate_audio crops back to the frames actually drawn.
    const auto available =
        std::min(static_cast<std::size_t>(frames) * width, frame_hidden.size() - begin);
    std::vector<float> window(static_cast<std::size_t>(frames) * width, 0.0F);
    std::copy(frame_hidden.begin() + static_cast<std::ptrdiff_t>(begin),
              frame_hidden.begin() + static_cast<std::ptrdiff_t>(begin + available),
              window.begin());

    TensorMap inputs;
    inputs[kConditionInput] =
        Tensor{window.data(), {1, frames, static_cast<int64_t>(width)}, DType::kFloat32};
    auto outputs = encoder.forward(inputs);
    const auto condition = outputs.find(kConditionOutput);
    if (condition == outputs.end())
        throw std::runtime_error("condition encoder produced no condition");
    const auto count = condition->second.numel();
    const auto* values = static_cast<const float*>(condition->second.data);
    return std::vector<float>(values, values + count);
}

void MinimaxMusic3TextToMusicPipeline::blend_overlap(std::vector<float>& latents,
                                                     const std::vector<float>& noise,
                                                     const std::vector<float>& neighbour,
                                                     int32_t latent_length, std::size_t carry,
                                                     float sigma) const {
    // A window's head is not free. It moves from the noise it started as at
    // sigma 0 to the previous window's own values at sigma 1, so the seam
    // between neighbours carries no discontinuity.
    if (carry == 0 || neighbour.empty())
        return;

    const auto channels = static_cast<std::size_t>(config_.latent_channels);
    const auto stride = neighbour.size() / channels;
    for (std::size_t channel = 0; channel < channels; ++channel) {
        for (std::size_t frame = 0; frame < carry; ++frame) {
            const auto here = channel * static_cast<std::size_t>(latent_length) + frame;
            const float from = noise.empty() ? 0.0F : noise[here];
            latents[here] = (1.0F - (1.0F - 1e-6F) * sigma) * from +
                            sigma * neighbour[channel * stride + frame];
        }
    }
}

void MinimaxMusic3TextToMusicPipeline::guide_velocity(ITrtModule& dit, TensorMap& inputs,
                                                      const std::vector<float>& condition,
                                                      int32_t latent_length,
                                                      std::vector<float>& guided) const {
    // Classifier-free guidance. The unconditional branch conditions on zeros
    // rather than on a re-encoded empty prompt, which is what the reference
    // guider does.
    if (config_.guidance_scale <= 1.0F)
        return;

    std::vector<float> silent(condition.size(), 0.0F);
    inputs[kDitConditionInput] =
        Tensor{silent.data(), {1, latent_length, config_.condition_dim}, DType::kFloat32};
    auto outputs = dit.forward(inputs);
    const auto unconditional = outputs.find(kVelocityOutput);
    if (unconditional == outputs.end())
        throw std::runtime_error("diffusion transformer produced no velocity");

    const auto* base = static_cast<const float*>(unconditional->second.data);
    for (std::size_t element = 0; element < guided.size(); ++element)
        guided[element] =
            base[element] + config_.guidance_scale * (guided[element] - base[element]);
}

std::vector<float> MinimaxMusic3TextToMusicPipeline::denoise_window(
    const std::vector<float>& condition, int32_t latent_length, int32_t steps, uint64_t seed,
    const std::vector<float>& previous) {
    auto& dit = *engines_.dit;
    const auto channels = static_cast<std::size_t>(config_.latent_channels);
    const auto count = channels * static_cast<std::size_t>(latent_length);

    std::mt19937_64 rng(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::vector<float> latents(count);
    for (auto& value : latents)
        value = normal(rng);

    // The head of this window is not free: it is blended toward the previous
    // window's trailing latents at every step, from pure noise at t = 0 to the
    // neighbour's own values at t = 1, so the seam carries no discontinuity.
    const auto carry = previous.empty()
                           ? 0
                           : std::min<std::size_t>(previous.size() / channels,
                                                   static_cast<std::size_t>(latent_length));
    const std::vector<float> noise_prompt = carry == 0 ? std::vector<float>() : latents;

    // sigma_schedule returns one more value than there are steps: the last is
    // where the final step lands.
    const auto sigmas = sigma_schedule(steps);
    for (std::size_t index = 0; index + 1 < sigmas.size(); ++index) {
        const float sigma = sigmas[index];
        const float next = sigmas[index + 1];

        blend_overlap(latents, noise_prompt, previous, latent_length, carry, sigma);

        std::vector<float> timestep(1, sigma);
        TensorMap inputs;
        inputs[kLatentsInput] =
            Tensor{latents.data(), {1, config_.latent_channels, latent_length}, DType::kFloat32};
        inputs[kDitConditionInput] = Tensor{const_cast<float*>(condition.data()),
                                            {1, latent_length, config_.condition_dim},
                                            DType::kFloat32};
        inputs[kTimestepInput] = Tensor{timestep.data(), {1, 1, 1}, DType::kFloat32};

        auto outputs = dit.forward(inputs);
        const auto velocity = outputs.find(kVelocityOutput);
        if (velocity == outputs.end())
            throw std::runtime_error("diffusion transformer produced no velocity");
        const auto* conditional = static_cast<const float*>(velocity->second.data);
        std::vector<float> guided(conditional, conditional + count);

        guide_velocity(dit, inputs, condition, latent_length, guided);

        if (index == 0 || index + 2 == sigmas.size()) {
            report_stage(index == 0 ? "velocity_first" : "velocity_last", guided.data(), count);
            report_stage(index == 0 ? "latents_first" : "latents_last", latents.data(), count);
        }

        // Flow matching's Euler step: the field points from noise toward data,
        // so the state moves by the sigma difference along it.
        const float delta = next - sigma;
        for (std::size_t element = 0; element < count; ++element)
            latents[element] += delta * guided[element];
    }

    // The neighbour's values win outright over the overlap once the window is
    // denoised; the blend was there to steer the rest of it, not to rewrite
    // frames the previous window already settled.
    blend_overlap(latents, {}, previous, latent_length, carry, 1.0F);
    return latents;
}

std::vector<float>
MinimaxMusic3TextToMusicPipeline::decode_waveform(const std::vector<float>& latents,
                                                  int32_t latent_length) {
    auto& vocoder = *engines_.vocoder;
    TensorMap inputs;
    inputs[kVocoderInput] = Tensor{const_cast<float*>(latents.data()),
                                   {1, config_.latent_channels, latent_length},
                                   DType::kFloat32};
    auto outputs = vocoder.forward(inputs);
    const auto waveform = outputs.find(kWaveformOutput);
    if (waveform == outputs.end())
        throw std::runtime_error("vocoder produced no waveform");
    const auto count = waveform->second.numel();
    const auto* values = static_cast<const float*>(waveform->second.data);

    // The engine emits (1, channels, samples); the audio result is
    // interleaved, so the two streams are woven here.
    const auto per_channel = count / static_cast<std::size_t>(config_.output_channels);
    std::vector<float> interleaved(count);
    for (std::size_t frame = 0; frame < per_channel; ++frame) {
        for (int32_t channel = 0; channel < config_.output_channels; ++channel) {
            interleaved[frame * static_cast<std::size_t>(config_.output_channels) +
                        static_cast<std::size_t>(channel)] =
                values[static_cast<std::size_t>(channel) * per_channel + frame];
        }
    }
    return interleaved;
}

} // namespace trtmc
