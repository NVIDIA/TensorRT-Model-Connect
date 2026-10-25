/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "families/minimax_h3/runtime/reference_pipeline.h"
#include "families/minimax_h3/runtime/torch_cuda_normal.h"

#include <array>
#include <iostream>
#include <stdexcept>

namespace {
using namespace trtmc;
using namespace trtmc::minimax_h3;
using namespace trtmc::internal;

void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}

class Tokenizer final : public ITokenizer {
  public:
    std::vector<int32_t> encode(const std::string& text) const override {
        return {text.begin(), text.end()};
    }
    std::string decode(const std::vector<int32_t>&) const override { return {}; }
    int32_t id_for_token(std::string_view text) const override {
        if (text == "<|vision_start|>")
            return 151652;
        if (text == "<|vision_end|>")
            return 151653;
        if (text == "<|image_pad|>")
            return 151655;
        return -1;
    }
    std::string token_for_id(int32_t) const override { return {}; }
};

ReferencePipelineConfig configuration(ReferenceMemoryPreset preset) {
    ReferencePipelineConfig config;
    config.memory = preset;
    config.max_text_rows = 32768;
    config.denoiser.max_sequence_rows = 65536;
    config.normalization.std.fill(1.0F);
    return config;
}

template <class Error, class Call>
void rejects(Call call) {
    try {
        call();
    } catch (const Error&) {
        return;
    }
    throw std::runtime_error("expected pipeline rejection did not occur");
}

void exercise(ReferenceMemoryPreset preset) {
    int loads = 0;
    const auto stream = reinterpret_cast<cudaStream_t>(uintptr_t{0x1234});
    ReferenceModuleLoader loader = [&](const std::string& name, cudaStream_t observed) {
        ++loads;
        require(name == "vision.plan" && observed == stream, "first phase and borrowed stream");
        throw std::runtime_error("intentional first-engine failure");
        return std::unique_ptr<ITrtModule>{};
    };
    ImageReferencePipeline model(loader, std::make_unique<Tokenizer>(), configuration(preset),
                                 stream);
    require(loads == 0, "construction must not deserialize engines");
    const auto bindings = model.task_bindings();
    require(bindings.size() == 1 && bindings[0].key.id == model.task() &&
                bindings[0].implementation == static_cast<IReferencesTextToAudioVideo*>(&model),
            "only the reference AV contract is bound, with the correct interface address");
    require(bindings[0].fields.size() == 1 && bindings[0].fields[0].name == "seed",
            "no reduced-step, downscale, or approximate-cache runtime overrides");

    const std::array<uint8_t, 3> pixel{127, 64, 32};
    ImageView image{pixel.data(), pixel.size(), 1, 1, 3, ImageFormat::UInt8};
    std::vector<VideoReferenceItem> references{image};
    const auto run = [&](ConfigView config = {}) {
        return model.run({{references.data(), references.size()}, "walk and wave"}, config);
    };
    const ConfigEntry unknown[] = {{"steps", int64_t{4}}};
    const ConfigEntry duplicate[] = {{"seed", int64_t{42}}, {"seed", int64_t{43}}};
    const ConfigEntry wrong_type[] = {{"seed", 42.0}};
    const ConfigEntry negative[] = {{"seed", int64_t{-1}}};
    const ConfigEntry child_on_single[] = {{"continuation_prompt", std::string_view{"wait"}}};
    for (const ConfigView config :
         {ConfigView{unknown}, ConfigView{duplicate}, ConfigView{wrong_type}, ConfigView{negative},
          ConfigView{child_on_single}})
        rejects<ConfigError>([&] { (void)run(config); });
    rejects<UnsupportedTask>([&] { (void)model.run({{}, "prompt"}, {}); });
    references.push_back(AudioView{});
    rejects<UnsupportedTask>([&] { (void)run(); });
    references.back() = VideoReference{};
    rejects<UnsupportedTask>([&] { (void)run(); });
    references.resize(1);
    for (int kind = 0; kind < 7; ++kind) {
        auto bad = image;
        switch (kind) {
        case 0:
            bad.data = nullptr;
            break;
        case 1:
            bad.height = 0;
            break;
        case 2:
            bad.byte_size = 2;
            break;
        case 3:
            bad.channels = 4;
            break;
        case 4:
            bad.format = ImageFormat::Float32;
            break;
        case 5:
            bad.width = uint32_t{1} << 31;
            break;
        case 6:
            bad.height = 5;
            bad.byte_size = 15;
            break;
        }
        references[0] = bad;
        rejects<std::invalid_argument>([&] { (void)run(); });
    }
    references[0] = image;
    require(loads == 0, "all request operands must be checked before the first module load");

    auto cap = configuration(preset);
    cap.max_text_rows = 10;
    ImageReferencePipeline text_limited(loader, std::make_unique<Tokenizer>(), cap, stream);
    rejects<std::length_error>([&] {
        (void)text_limited.run({{references.data(), references.size()}, "prompt"}, {});
    });
    cap = configuration(preset);
    cap.denoiser.max_sequence_rows = 40000;
    ImageReferencePipeline dit_limited(loader, std::make_unique<Tokenizer>(), cap, stream);
    rejects<std::length_error>([&] {
        (void)dit_limited.run({{references.data(), references.size()}, "prompt"}, {});
    });
    require(loads == 0, "Qwen and Ref2VA packed caps must fail before allocation/inference");
    cap = configuration(preset);
    cap.denoiser.max_sequence_rows = 86862;
    ImageReferencePipeline reference_limited(loader, std::make_unique<Tokenizer>(), cap, stream);
    references.assign(5, image);
    rejects<std::length_error>([&] {
        (void)reference_limited.run({{references.data(), references.size()}, "prompt"}, {});
    });
    require(loads == 0, "per-reference operand cap is stricter than the combined sequence cap");
    references.assign(1, image);

    // Real preparation reaches the actual conditioner with the original 2048
    // short edge. No TensorRT plan, model oracle or full generation is faked.
    for (int retry = 0; retry < 2; ++retry) {
        rejects<std::runtime_error>([&] { (void)run(); });
        require(loads == retry + 1, "a failed phase must permit a fresh native request");
    }

    // Six actual 9:16 operands need 43,776 image rows in BOTH conditioning
    // streams. Prove each old cap rejects before any allocation/engine load;
    // expanded metadata reaches the same intentional first-engine failure.
    std::array<std::array<uint8_t, 16 * 9 * 3>, 6> portraits{};
    references.clear();
    for (std::size_t index = 0; index < portraits.size(); ++index) {
        auto& portrait = portraits[index];
        portrait.fill(static_cast<uint8_t>(index * 31));
        references.emplace_back(
            ImageView{portrait.data(), portrait.size(), 16, 9, 3, ImageFormat::UInt8});
    }
    const auto before = loads;
    cap = configuration(preset);
    cap.denoiser.max_sequence_rows = 152398;
    ImageReferencePipeline old_text(loader, std::make_unique<Tokenizer>(), cap, stream);
    rejects<std::length_error>([&] {
        (void)old_text.run({{references.data(), references.size()}, "prompt"}, {});
    });
    cap.max_text_rows = 65536;
    ImageReferencePipeline old_visual(loader, std::make_unique<Tokenizer>(), cap, stream);
    rejects<std::length_error>([&] {
        (void)old_visual.run({{references.data(), references.size()}, "prompt"}, {});
    });
    require(loads == before, "both legacy operand budgets reject six portraits before loading");
    cap.max_condition_video_rows = 49152;
    ImageReferencePipeline expanded(loader, std::make_unique<Tokenizer>(), cap, stream);
    rejects<std::runtime_error>([&] {
        (void)expanded.run({{references.data(), references.size()}, "prompt"}, {});
    });
    require(loads == before + 1,
            "expanded six-portrait input reaches the real conditioner without dropping references");
    cap.output_frames = 226;
    cap.max_condition_audio_rows = 74;
    cap.denoiser.max_sequence_rows = 159528;
    ImageReferencePipeline old_guide_cap(loader, std::make_unique<Tokenizer>(), cap, stream);
    rejects<std::length_error>([&] {
        (void)old_guide_cap.run({{references.data(), references.size()}, "prompt"}, {});
    });
    require(loads == before + 1,
            "six canonical portraits plus the guide cannot fit the old video operand cap");
    cap.max_condition_video_rows = 56208;
    ImageReferencePipeline continuation(loader, std::make_unique<Tokenizer>(), cap, stream);
    rejects<std::runtime_error>([&] {
        (void)continuation.run({{references.data(), references.size()}, "prompt"}, {});
    });
    require(loads == before + 2,
            "complete two-window profiles retain all six images and reach the real conditioner");
    const auto window_bindings = continuation.task_bindings();
    require(window_bindings[0].fields.size() == 2 &&
                window_bindings[0].fields[1].name == "continuation_prompt" &&
                window_bindings[0].fields[1].kind == ConfigKind::String &&
                config_value_as<std::string_view>(
                    *window_bindings[0].fields[1].default_value).empty(),
            "only the two-window profile advertises optional second-window text");
    const auto run_child = [&](ConfigView config) {
        return continuation.run({{references.data(), references.size()}, "prompt"}, config);
    };
    const ConfigEntry child_type[] = {{"continuation_prompt", int64_t{42}}};
    const ConfigEntry child_duplicate[] = {
        {"continuation_prompt", std::string_view{"wait"}},
        {"continuation_prompt", std::string_view{"turn"}}};
    rejects<ConfigError>([&] { (void)run_child(child_type); });
    rejects<ConfigError>([&] { (void)run_child(child_duplicate); });
    const std::string too_long(65536, 'x');
    const ConfigEntry oversized_child[] = {{"continuation_prompt", std::string_view{too_long}}};
    rejects<std::length_error>([&] { (void)run_child(oversized_child); });
    require(loads == before + 2,
            "invalid child text fails before the first vision/Qwen/VAE/DiT plan load");
    for (const std::string_view prompt : {std::string_view{}, std::string_view{"prompt"},
                                          std::string_view{"They wait quietly."}}) {
        const ConfigEntry child[] = {{"continuation_prompt", prompt}};
        rejects<std::runtime_error>([&] { (void)run_child(child); });
    }
    require(loads == before + 5,
            "empty, repeated and distinct child prompts pass real operand preflight");
    cap.max_condition_audio_rows = 0;
    rejects<std::invalid_argument>(
        [&] { ImageReferencePipeline invalid(loader, std::make_unique<Tokenizer>(), cap); });
    cap = configuration(preset);
    cap.output_frames = 225;
    rejects<std::invalid_argument>(
        [&] { ImageReferencePipeline invalid(loader, std::make_unique<Tokenizer>(), cap); });
    auto bad_topology = configuration(preset);
    bad_topology.denoiser.layers = 1;
    rejects<std::invalid_argument>([&] {
        ImageReferencePipeline invalid(loader, std::make_unique<Tokenizer>(), bad_topology);
    });
}
} // namespace

// Link-time guards for this host-only preflight test. Reaching RNG would mean
// inference ran past the intentional first-engine failure, and must fail.
namespace trtmc::minimax_h3 {
std::vector<float> torch_cuda_normal(std::size_t, uint64_t, uint64_t) {
    throw std::runtime_error("preflight test unexpectedly reached request RNG");
}
uint64_t torch_cuda_normal_consumed_offset(std::size_t) {
    throw std::runtime_error("preflight test unexpectedly reached request RNG offset");
}
} // namespace trtmc::minimax_h3

int main() {
    try {
        exercise(ReferenceMemoryPreset::Minimal);
        exercise(ReferenceMemoryPreset::Original);
        std::cout << "original H3 pipeline preflight and first-phase recovery passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
