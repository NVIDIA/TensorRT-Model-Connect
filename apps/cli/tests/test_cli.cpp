/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cli/cli.h"
#include "cli/io.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

trtmc::cli::Command parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (auto& argument : arguments)
        argv.push_back(argument.data());
    return trtmc::cli::parse_args(static_cast<int>(argv.size()), argv.data());
}

bool parse_throws(std::vector<std::string> arguments) {
    try {
        (void)parse(std::move(arguments));
        return false;
    } catch (const std::invalid_argument&) {
        return true;
    }
}

class FakeText final : public trtmc::ITextGeneration,
                       public trtmc::IEmbedding,
                       public trtmc::ILoraAdapterManager {
  public:
    trtmc::TextGenerationConfig seen;
    std::string loaded_adapter_id;
    std::string loaded_adapter_path;
    const char* task() const noexcept override { return trtmc::ITextGeneration::kTask; }
    std::int32_t default_max_new_tokens() const override { return 7; }

    trtmc::TextResult generate(const std::string& prompt,
                               const trtmc::TextGenerationConfig& config) override {
        seen = config;
        return {prompt + ":" + std::to_string(config.max_new_tokens), {1, 2}};
    }

    trtmc::EmbeddingResult embed(const std::string& text) override {
        return {{static_cast<float>(text.size())}, 1};
    }

    void load_lora_adapter(const std::string& adapter_id,
                           const std::string& adapter_path) override {
        loaded_adapter_id = adapter_id;
        loaded_adapter_path = adapter_path;
    }

    void unload_lora_adapter(const std::string& adapter_id) override {
        if (loaded_adapter_id == adapter_id) {
            loaded_adapter_id.clear();
            loaded_adapter_path.clear();
        }
    }

    std::vector<std::string> loaded_lora_adapters() const override {
        return loaded_adapter_id.empty() ? std::vector<std::string>{}
                                         : std::vector<std::string>{loaded_adapter_id};
    }
};

class BareTextTask final : public trtmc::ITask {
  public:
    const char* task() const noexcept override { return trtmc::ITextGeneration::kTask; }
};

class EmptyImageWorker final : public trtmc::IImageGeneration {
  public:
    trtmc::ImageResult generate_image(const std::string&,
                                      const trtmc::ImageGenerationConfig&) override {
        trtmc::ImageResult result;
        result.num_frames = 0;
        return result;
    }
};

class FakeStreamingAudio final : public trtmc::IAudioGeneration,
                                 public trtmc::IStreamingAudioGeneration {
  public:
    trtmc::AudioGenerationConfig seen;
    std::int32_t seen_chunk_frames{0};

    trtmc::AudioResult generate_audio(const std::string&,
                                      const trtmc::AudioGenerationConfig&) override {
        throw std::logic_error("synchronous audio path was selected");
    }

    std::int32_t generate_audio_streaming(const std::string&,
                                          const trtmc::AudioGenerationConfig& config,
                                          trtmc::AudioChunkCallback callback,
                                          std::int32_t chunk_frames) override {
        seen = config;
        seen_chunk_frames = chunk_frames;
        const float samples[] = {0.25F, -0.5F, 0.75F};
        callback(samples, 3, 24000);
        return 3;
    }
};

class FakeTranscriptionStream final : public trtmc::ITranscriptionStream {
  public:
    explicit FakeTranscriptionStream(trtmc::TranscriptionStreamConfig config)
        : config_(std::move(config)) {}

    trtmc::TranscriptionStreamResult accept_audio(const float*, std::int32_t num_samples,
                                                  bool) override {
        accepted_samples_ += num_samples;
        return {"partial", {}, false, 0, accepted_samples_, config_.input_sample_rate};
    }

    trtmc::TranscriptionStreamResult finish() override {
        return {"final", {}, true, 1, accepted_samples_, config_.input_sample_rate};
    }

    void reset() override { accepted_samples_ = 0; }
    trtmc::TranscriptionStreamConfig config() const override { return config_; }

  private:
    trtmc::TranscriptionStreamConfig config_;
    std::int64_t accepted_samples_{0};
};

class FakeStreamingTranscription final : public trtmc::IStreamingTranscription {
  public:
    trtmc::TranscriptionStreamConfig seen;

    std::unique_ptr<trtmc::ITranscriptionStream>
    create_transcription_stream(const trtmc::TranscriptionStreamConfig& config) override {
        seen = config;
        return std::make_unique<FakeTranscriptionStream>(config);
    }
};

class FakeTranscription final : public trtmc::ITranscription, public trtmc::IBatchTranscription {
  public:
    std::vector<trtmc::TranscriptionConfig> seen;

    const char* task() const noexcept override { return trtmc::ITranscription::kTask; }

    trtmc::TextResult transcribe(const float*, std::int32_t,
                                 const trtmc::TranscriptionConfig& config) override {
        seen = {config};
        return {"transcribed", {1}};
    }

    std::vector<trtmc::TextResult>
    transcribe_batch(const std::vector<trtmc::TranscriptionRequest>& requests) override {
        seen.clear();
        std::vector<trtmc::TextResult> results;
        for (const auto& request : requests) {
            seen.push_back(request.config);
            results.push_back({"transcribed", {1}});
        }
        return results;
    }
};

class FakeVideoSession final : public trtmc::IVideoSegmentationSession {
  public:
    explicit FakeVideoSession(std::string& prompt) : prompt_(prompt) {}

    trtmc::VideoSegmentationResult
    segment(const trtmc::VideoSegmentationRequest& request) override {
        prompt_ = request.text_prompt;
        trtmc::VideoSegmentationFrameResult frame;
        frame.masks = {1};
        frame.object_ids = {1};
        frame.detection_scores = {1.0F};
        frame.tracking_scores = {1.0F};
        frame.boxes = {0.0F, 0.0F, 1.0F, 1.0F};
        frame.num_objects = 1;
        frame.height = 1;
        frame.width = 1;
        return {{std::move(frame)}};
    }

  private:
    std::string& prompt_;
};

class FakeVideo final : public trtmc::IVideoSegmentation {
  public:
    std::string prompt;

    std::unique_ptr<trtmc::IVideoSegmentationSession> create_video_segmentation_session() override {
        return std::make_unique<FakeVideoSession>(prompt);
    }
};

class FakeForecast final : public trtmc::ITimeSeriesForecast {
  public:
    std::int32_t seen_frequency{-1};

    trtmc::ForecastResult forecast(const trtmc::ForecastRequest& request) override {
        seen_frequency = request.frequency;
        trtmc::ForecastResult result;
        result.values.assign(request.past_values.begin(), request.past_values.end());
        result.shape = {1, static_cast<std::int64_t>(request.past_values.size())};
        return result;
    }
};

class FakeControl final : public trtmc::IRobotControl {
  public:
    trtmc::RobotObservation seen;

    trtmc::RobotActionChunk
    predict_action_chunk(const trtmc::RobotObservation& observation) override {
        seen = observation;
        return {{0.25F, -0.5F}, 1, 2, true, 3.0};
    }

    trtmc::RobotAction act(const trtmc::RobotObservation&) override { return {}; }
    void reset() override {}
};

bool dispatch_throws(const trtmc::cli::Command& command, trtmc::ITask& task) {
    try {
        std::ostringstream output;
        (void)trtmc::cli::dispatch(command, task, output);
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

} // namespace

int main() {
    const std::vector<std::string> execution_commands{
        "run",
        "encode",
        "embed",
        "rerank",
        "classify",
        "extract-features",
        "disparity",
        "geometry",
        "segment",
        "segment-prompted",
        "video-segment",
        "generate-audio",
        "transcribe",
        "transcribe-batch",
        "transcribe-streaming",
        "speak",
        "speech-session",
        "generate-image",
        "generate-image-batch",
        "generate-video",
        "solve",
        "forecast",
        "control",
        "generate-world",
    };
    for (const auto& name : execution_commands) {
        const auto command = parse({"trtmc", name, "model.bundle", "--runtime-root", "lib"});
        check(command.name == name, "execution command parses");
        check(command.runtime_root == "lib", "runtime root is retained");
    }

    check(parse_throws({"trtmc", "run", "model.bundle"}),
          "execution command requires runtime root");
    check(parse_throws(
              {"trtmc", "run", "model.bundle", "--runtime-root", "a", "--runtime-root", "b"}),
          "duplicate runtime root rejected");
    const auto dynamic_kv =
        parse({"trtmc", "run", "model.bundle", "--runtime-root", "lib", "--kv-cache-size", "1GiB"});
    check(dynamic_kv.kv_cache_size_bytes == 1024ULL * 1024ULL * 1024ULL,
          "runtime KV cache size is retained as bytes");
    check(parse({"trtmc", "run", "model.bundle", "--runtime-root", "lib", "--kv-cache-size",
                 "1000000000"})
                  .kv_cache_size_bytes == 1000000000ULL,
          "runtime KV cache accepts explicit bytes");
    check(parse({"trtmc", "run", "model.bundle", "--runtime-root", "lib", "--kv-cache-size", "1GB"})
                  .kv_cache_size_bytes == 1000000000ULL,
          "runtime KV cache distinguishes decimal GB");
    check(parse_throws(
              {"trtmc", "run", "model.bundle", "--runtime-root", "lib", "--kv-cache-size", "0"}),
          "zero runtime KV cache is rejected");
    check(parse_throws(
              {"trtmc", "run", "model.bundle", "--runtime-root", "lib", "--kv-cache-size", "1MiB"}),
          "uncanonical runtime KV cache suffix is rejected");
    const auto rtx = parse({"trtmc", "run", "model.bundle", "--runtime-root", "lib",
                            "--runtime-cache", "kernels.cache", "--cuda-graphs"});
    check(rtx.runtime_cache_path == "kernels.cache" && rtx.cuda_graphs,
          "TensorRT-RTX runtime options are retained directly");
    check(parse_throws({"trtmc", "run", "model.bundle", "--runtime-root", "lib", "--cuda-graphs",
                        "--cuda-graphs"}),
          "duplicate TensorRT-RTX graph option is rejected");
    check(
        parse_throws({"trtmc", "run", "model.bundle", "--runtime-root", "lib", "--bogus", "value"}),
        "unknown command option rejected");
    check(parse_throws({"trtmc", "run", "model.bundle", "--runtime-root", "lib", "--prompt", "a",
                        "--prompt", "b"}),
          "duplicate command option rejected");
    check(parse_throws({"trtmc", "run", "model.bundle", "--runtime-root", "lib", "--prompt", "a",
                        "--generation-mode"}),
          "generation mode requires a value");
    check(parse_throws({"trtmc", "embed", "model.bundle", "--runtime-root", "lib", "--text", "a",
                        "--generation-mode", "ar"}),
          "text generation options stay on run");
    const auto byok = parse({"trtmc", "run", "model.bundle", "--runtime-root", "lib", "--prompt",
                             "hello", "--byok-library", "kernel.so", "--byok-function", "run",
                             "--byok-name", "example.kernel"});
    check(byok.options.at("--byok-library") == "kernel.so" &&
              byok.options.at("--byok-function") == "run" &&
              byok.options.at("--byok-name") == "example.kernel",
          "BYOK options are retained");
    check(parse_throws({"trtmc", "run", "model.bundle", "--runtime-root", "lib", "--prompt",
                        "hello", "--byok-library", "kernel.so"}),
          "partial BYOK options are rejected");
    const auto replay = parse({"trtmc",
                               "run",
                               "model.bundle",
                               "--runtime-root",
                               "lib",
                               "--source-language-token-id",
                               "256047",
                               "--forced-bos-token-id",
                               "256057",
                               "--initial-latents-raw",
                               "initial.f32",
                               "--condition-latents-raw",
                               "condition.f32",
                               "--condition-mask-raw",
                               "mask.f32",
                               "--sampling-steps-raw",
                               "steps.f32",
                               "--sde-noise-raw",
                               "noise.f32",
                               "--generation-mode",
                               "diffusion",
                               "--block-length",
                               "32",
                               "--threshold",
                               "0.9",
                               "--num-steps",
                               "4",
                               "--guidance-scale",
                               "3",
                               "--cfg-scale",
                               "2",
                               "--sde-gamma",
                               "0"});
    check(replay.options.at("--initial-latents-raw") == "initial.f32" &&
              replay.options.at("--sampling-steps-raw") == "steps.f32" &&
              replay.options.at("--source-language-token-id") == "256047" &&
              replay.options.at("--forced-bos-token-id") == "256057" &&
              replay.options.at("--generation-mode") == "diffusion" &&
              replay.options.at("--block-length") == "32" &&
              replay.options.at("--threshold") == "0.9",
          "text diffusion replay options are retained");
    const auto image =
        parse({"trtmc", "generate-image", "model.bundle", "--runtime-root", "lib", "--prompt",
               "cat", "--output", "cat.png", "--guidance-scale", "4.5", "--cfg-scale", "2.0"});
    check(image.options.at("--guidance-scale") == "4.5" && image.options.at("--cfg-scale") == "2.0",
          "image guidance options are retained");
    const auto batch =
        parse({"trtmc", "generate-image-batch", "model.bundle", "--runtime-root", "lib",
               "--prompts", "prompts.txt", "--seeds", "1,2", "--output", "images"});
    check(batch.kind == trtmc::cli::CommandKind::kGenerateImageBatch,
          "batch image command parses directly");

    const auto streaming_audio =
        parse({"trtmc", "generate-audio", "model.bundle", "--runtime-root", "lib", "--prompt",
               "hello", "--output", "/tmp/audio.raw", "--stream", "true", "--chunk-frames", "4"});
    check(streaming_audio.options.at("--stream") == "true" &&
              streaming_audio.options.at("--chunk-frames") == "4",
          "streaming audio options are retained");

    const auto video = parse({"trtmc", "video-segment", "model.bundle", "--runtime-root", "lib",
                              "--frame", "a.png", "--frame", "b.png", "--prompt", "car"});
    check(video.frames == std::vector<std::string>({"a.png", "b.png"}),
          "video frames preserve repeated option order");
    const auto video_without_prompt = parse(
        {"trtmc", "video-segment", "model.bundle", "--runtime-root", "lib", "--frame", "a.png"});
    check(video_without_prompt.options.count("--prompt") == 0, "video prompt is optional");
    const auto transcription_batch =
        parse({"trtmc", "transcribe-batch", "model.bundle", "--runtime-root", "lib", "--input",
               "a.wav", "--input", "b.wav"});
    check(transcription_batch.inputs == std::vector<std::string>({"a.wav", "b.wav"}),
          "batch transcription inputs preserve repeated option order");
    check(parse({"trtmc", "inspect", "model.bundle"}).kind == trtmc::cli::CommandKind::kInspect,
          "inspect does not require runtime root");
    check(parse({"trtmc", "version"}).kind == trtmc::cli::CommandKind::kVersion,
          "version parses without bundle");
    check(parse_throws({"trtmc", "--version"}), "version flag alias is not accepted");
    check(parse_throws({"trtmc", "-h"}), "help flag alias is not accepted");

    trtmc::cli::Command run_command;
    run_command.kind = trtmc::cli::CommandKind::kRun;
    run_command.name = "run";
    run_command.options.emplace("--prompt", "hello");
    run_command.options.emplace("--max-new-tokens", "3");
    run_command.options.emplace("--source-language-token-id", "256047");
    run_command.options.emplace("--forced-bos-token-id", "256057");
    run_command.options.emplace("--temperature", "0");
    run_command.options.emplace("--top-k", "9");
    run_command.options.emplace("--top-p", "0.8");
    run_command.options.emplace("--min-p", "0.1");
    run_command.options.emplace("--seed", "42");
    run_command.options.emplace("--repetition-penalty", "1.1");
    run_command.options.emplace("--generation-mode", "linear_spec");
    run_command.options.emplace("--block-length", "16");
    run_command.options.emplace("--threshold", "0.75");
    run_command.options.emplace("--use-chat-template", "true");
    run_command.options.emplace("--enable-thinking", "false");
    run_command.options.emplace("--lora-adapter", "/tmp/adapter");
    run_command.options.emplace("--lora-adapter-id", "demo");
    const std::filesystem::path replay_values_path = "/tmp/trtmc-cli-replay-values.f32";
    const float replay_values[] = {0.25F, -0.5F};
    {
        std::ofstream replay_file(replay_values_path, std::ios::binary);
        replay_file.write(reinterpret_cast<const char*>(replay_values), sizeof(replay_values));
    }
    run_command.options.emplace("--num-steps", "4");
    run_command.options.emplace("--guidance-scale", "3");
    run_command.options.emplace("--cfg-scale", "2");
    run_command.options.emplace("--sde-gamma", "0");
    run_command.options.emplace("--initial-latents-raw", replay_values_path.string());
    run_command.options.emplace("--condition-latents-raw", replay_values_path.string());
    run_command.options.emplace("--condition-mask-raw", replay_values_path.string());
    run_command.options.emplace("--sampling-steps-raw", replay_values_path.string());
    run_command.options.emplace("--sde-noise-raw", replay_values_path.string());

    FakeText text;
    std::ostringstream output;
    check(trtmc::cli::dispatch(run_command, text, output) == 0, "text task dispatch succeeds");
    check(output.str().find("hello:3") != std::string::npos,
          "text task dispatch writes result JSON");
    check(text.seen.source_language_token_id == 256047 && text.seen.forced_bos_token_id == 256057 &&
              text.seen.temperature == 0.0F && text.seen.top_k == 9 && text.seen.top_p == 0.8F &&
              text.seen.min_p == 0.1F && text.seen.seed == 42 &&
              text.seen.repetition_penalty == 1.1F && text.seen.use_chat_template &&
              !text.seen.enable_thinking && text.seen.lora_adapter_id == "demo" &&
              text.seen.text_generation_mode == "linear_spec" && text.seen.block_length == 16 &&
              text.seen.confidence_threshold == 0.75F && text.seen.num_steps == 4 &&
              text.seen.guidance_scale == 3.0F && text.seen.cfg_scale == 2.0F &&
              text.seen.sde_gamma == 0.0F &&
              text.seen.initial_latents == std::vector<float>({0.25F, -0.5F}) &&
              text.seen.condition_latents == text.seen.initial_latents &&
              text.seen.condition_mask == text.seen.initial_latents &&
              text.seen.sampling_steps == text.seen.initial_latents &&
              text.seen.sde_noises == text.seen.initial_latents &&
              text.loaded_adapter_id == "demo" && text.loaded_adapter_path == "/tmp/adapter",
          "text sampling options reach the Task API");
    auto invalid_block = run_command;
    invalid_block.options["--block-length"] = "invalid";
    check(dispatch_throws(invalid_block, text), "invalid block length is rejected");
    auto invalid_threshold = run_command;
    invalid_threshold.options["--threshold"] = "nan";
    check(dispatch_throws(invalid_threshold, text), "non-finite threshold is rejected");
    auto family_mode = run_command;
    family_mode.options["--generation-mode"] = "family_owned_mode";
    check(trtmc::cli::dispatch(family_mode, text, output) == 0 &&
              text.seen.text_generation_mode == "family_owned_mode",
          "generation mode validation stays family-owned");

    trtmc::cli::Command default_run;
    default_run.kind = trtmc::cli::CommandKind::kRun;
    default_run.name = "run";
    default_run.options.emplace("--prompt", "hello");
    check(trtmc::cli::dispatch(default_run, text, output) == 0 &&
              text.seen.text_generation_mode == "auto" && text.seen.block_length == 0 &&
              text.seen.confidence_threshold == -1.0F && text.seen.temperature == 1.0F,
          "text diffusion options preserve Task API defaults");
    std::filesystem::remove(replay_values_path);

    const std::filesystem::path unsupported_image_path = "/tmp/trtmc-cli-unsupported.ppm";
    {
        std::ofstream image_file(unsupported_image_path, std::ios::binary);
        image_file << "P6\n1 1\n255\n";
        const char pixel[] = {char(0), char(127), char(255)};
        image_file.write(pixel, sizeof(pixel));
    }
    trtmc::cli::Command vision_command;
    vision_command.kind = trtmc::cli::CommandKind::kRun;
    vision_command.name = "run";
    vision_command.options.emplace("--prompt", "describe this");
    vision_command.options.emplace("--image", unsupported_image_path.string());
    check(dispatch_throws(vision_command, text),
          "text-only task rejects an input image instead of dropping it");

    trtmc::cli::Command edit_command;
    edit_command.kind = trtmc::cli::CommandKind::kGenerateImage;
    edit_command.name = "generate-image";
    edit_command.options.emplace("--prompt", "make it night");
    edit_command.options.emplace("--image", unsupported_image_path.string());
    edit_command.options.emplace("--output", "/tmp/trtmc-cli-unused.png");
    EmptyImageWorker image_only;
    check(dispatch_throws(edit_command, image_only),
          "image-generation task rejects an edit image instead of dropping it");
    std::filesystem::remove(unsupported_image_path);

    trtmc::cli::Command embed_command;
    embed_command.kind = trtmc::cli::CommandKind::kEmbed;
    embed_command.name = "embed";
    embed_command.options.emplace("--text", "hello");
    std::ostringstream embed_output;
    check(trtmc::cli::dispatch(embed_command, text, embed_output) == 0,
          "secondary Task API dispatch succeeds");
    check(embed_output.str().find("5.0") != std::string::npos,
          "secondary Task API result is returned");

    BareTextTask bare;
    check(dispatch_throws(run_command, bare), "active task without exact interface is rejected");

    trtmc::cli::Command worker_command;
    worker_command.kind = trtmc::cli::CommandKind::kGenerateVideo;
    worker_command.name = "generate-video";
    worker_command.options.emplace("--prompt", "hello");
    worker_command.options.emplace("--output", "unused-worker-output");
    EmptyImageWorker worker;
    std::ostringstream worker_output;
    check(trtmc::cli::dispatch(worker_command, worker, worker_output) == 0,
          "empty context-parallel worker result succeeds");
    check(worker_output.str().find("\"worker\":true") != std::string::npos,
          "empty context-parallel worker result is explicit");
    worker_command.kind = trtmc::cli::CommandKind::kGenerateImage;
    worker_command.name = "generate-image";
    worker_output.str("");
    worker_output.clear();
    check(trtmc::cli::dispatch(worker_command, worker, worker_output) == 0,
          "empty context-parallel image worker succeeds");
    check(worker_output.str().find("\"worker\":true") != std::string::npos,
          "empty context-parallel image worker result is explicit");

    trtmc::cli::Command audio_command;
    audio_command.kind = trtmc::cli::CommandKind::kGenerateAudio;
    audio_command.name = "generate-audio";
    const std::filesystem::path audio_path = "/tmp/trtmc-cli-stream-test.raw";
    audio_command.options.emplace("--prompt", "hello");
    audio_command.options.emplace("--output", audio_path.string());
    audio_command.options.emplace("--max-new-tokens", "9");
    audio_command.options.emplace("--seed", "17");
    audio_command.options.emplace("--stream", "true");
    audio_command.options.emplace("--chunk-frames", "4");
    FakeStreamingAudio audio;
    std::ostringstream audio_output;
    check(trtmc::cli::dispatch(audio_command, audio, audio_output) == 0,
          "streaming audio task dispatch succeeds");
    check(std::filesystem::file_size(audio_path) == 3 * sizeof(float),
          "streaming audio writes raw float32 samples");
    check(audio.seen.max_new_tokens == 9 && audio.seen.seed == 17 && audio.seen_chunk_frames == 4,
          "streaming audio options reach the Task API");
    check(audio_output.str().find("\"format\":\"float32le\"") != std::string::npos,
          "streaming audio output format is explicit");
    std::filesystem::remove(audio_path);

    const std::filesystem::path transcription_path = "/tmp/trtmc-cli-transcription-stream.wav";
    trtmc::AudioResult transcription_audio;
    transcription_audio.samples = {0.25F, -0.5F, 0.75F};
    transcription_audio.num_samples = 3;
    transcription_audio.sample_rate = 16000;
    trtmc::cli::io::write_wav(transcription_audio, transcription_path.string());

    const auto offline_transcription_command = parse({"trtmc",
                                                      "transcribe",
                                                      "model.bundle",
                                                      "--runtime-root",
                                                      "lib",
                                                      "--input",
                                                      transcription_path.string(),
                                                      "--beam-size",
                                                      "4",
                                                      "--length-penalty",
                                                      "0.5",
                                                      "--source-language",
                                                      "en",
                                                      "--target-language",
                                                      "fr",
                                                      "--translate",
                                                      "true",
                                                      "--punctuation",
                                                      "false",
                                                      "--timestamps",
                                                      "true",
                                                      "--max-input-seconds",
                                                      "45.5",
                                                      "--segment-length-seconds",
                                                      "30",
                                                      "--segment-min-seconds",
                                                      "20",
                                                      "--segment-overlap-seconds",
                                                      "2",
                                                      "--lcs-merge",
                                                      "true"});
    FakeTranscription offline_transcription;
    std::ostringstream offline_transcription_output;
    check(trtmc::cli::dispatch(offline_transcription_command, offline_transcription,
                               offline_transcription_output) == 0,
          "offline transcription task dispatch succeeds");
    check(offline_transcription.seen.size() == 1 && offline_transcription.seen[0].beam_size == 4 &&
              offline_transcription.seen[0].length_penalty == 0.5F &&
              offline_transcription.seen[0].source_language == "en" &&
              offline_transcription.seen[0].target_language == "fr" &&
              offline_transcription.seen[0].task == trtmc::TranscriptionTask::kTranslate &&
              !offline_transcription.seen[0].punctuation &&
              offline_transcription.seen[0].timestamps &&
              offline_transcription.seen[0].max_input_duration_seconds == 45.5F &&
              offline_transcription.seen[0].segment_duration_seconds == 30.0F &&
              offline_transcription.seen[0].segment_min_duration_seconds == 20.0F &&
              offline_transcription.seen[0].segment_overlap_seconds == 2.0F &&
              offline_transcription.seen[0].lcs_merge,
          "offline transcription options reach the Task API");

    const auto batch_transcription_command = parse({"trtmc",
                                                    "transcribe-batch",
                                                    "model.bundle",
                                                    "--runtime-root",
                                                    "lib",
                                                    "--input",
                                                    transcription_path.string(),
                                                    "--input",
                                                    transcription_path.string(),
                                                    "--beam-size",
                                                    "2",
                                                    "--length-penalty",
                                                    "0",
                                                    "--punctuation",
                                                    "false",
                                                    "--max-input-seconds",
                                                    "45",
                                                    "--segment-length-seconds",
                                                    "30",
                                                    "--segment-min-seconds",
                                                    "20",
                                                    "--segment-overlap-seconds",
                                                    "2",
                                                    "--lcs-merge",
                                                    "true"});
    std::ostringstream batch_transcription_output;
    check(trtmc::cli::dispatch(batch_transcription_command, offline_transcription,
                               batch_transcription_output) == 0,
          "batch transcription task dispatch succeeds");
    check(offline_transcription.seen.size() == 2 && offline_transcription.seen[0].beam_size == 2 &&
              offline_transcription.seen[0].length_penalty == 0.0F &&
              !offline_transcription.seen[0].punctuation &&
              offline_transcription.seen[0].max_input_duration_seconds == 45.0F &&
              offline_transcription.seen[0].segment_duration_seconds == 30.0F &&
              offline_transcription.seen[0].segment_min_duration_seconds == 20.0F &&
              offline_transcription.seen[0].segment_overlap_seconds == 2.0F &&
              offline_transcription.seen[0].lcs_merge &&
              offline_transcription.seen[1].beam_size == 2,
          "batch transcription options reach every Task API request");

    const auto transcription_command =
        parse({"trtmc", "transcribe-streaming", "model.bundle", "--runtime-root", "lib", "--input",
               transcription_path.string(), "--chunk-samples", "2", "--max-new-tokens", "80",
               "--att-context-left", "56", "--att-context-right", "13", "--language", "en-US"});
    FakeStreamingTranscription transcription;
    std::ostringstream transcription_output;
    check(trtmc::cli::dispatch(transcription_command, transcription, transcription_output) == 0,
          "streaming transcription task dispatch succeeds");
    check(transcription.seen.max_new_tokens == 80 && transcription.seen.att_context_left == 56 &&
              transcription.seen.att_context_right == 13 && transcription.seen.language == "en-US",
          "streaming transcription options reach the Task API");

    auto default_transcription_command = transcription_command;
    default_transcription_command.options.erase("--max-new-tokens");
    default_transcription_command.options.erase("--att-context-left");
    default_transcription_command.options.erase("--att-context-right");
    default_transcription_command.options.erase("--language");
    FakeStreamingTranscription default_transcription;
    transcription_output.str("");
    transcription_output.clear();
    check(trtmc::cli::dispatch(default_transcription_command, default_transcription,
                               transcription_output) == 0,
          "default streaming transcription dispatch succeeds");
    check(default_transcription.seen.max_new_tokens == 224 &&
              default_transcription.seen.att_context_left == 70 &&
              default_transcription.seen.att_context_right == 13 &&
              default_transcription.seen.language.empty(),
          "streaming transcription defaults remain unchanged");
    std::filesystem::remove(transcription_path);

    std::ostringstream usage;
    trtmc::cli::print_usage(usage);
    check(usage.str().find("--source-language-token-id") != std::string::npos &&
              usage.str().find("--segment-overlap-seconds") != std::string::npos &&
              usage.str().find("--runtime-cache") != std::string::npos &&
              usage.str().find("--cuda-graphs") != std::string::npos,
          "help lists direct Task and backend options");

    const std::filesystem::path video_path = "/tmp/trtmc-cli-video-frame.ppm";
    {
        std::ofstream image_file(video_path, std::ios::binary);
        image_file << "P6\n1 1\n255\n";
        const char pixel[] = {char(0), char(127), char(255)};
        image_file.write(pixel, sizeof(pixel));
    }
    auto video_command = video_without_prompt;
    video_command.frames = {video_path.string()};
    FakeVideo video_task;
    std::ostringstream video_output;
    check(trtmc::cli::dispatch(video_command, video_task, video_output) == 0,
          "video segmentation dispatch accepts no prompt");
    check(video_task.prompt.empty(), "missing video prompt reaches the family as empty");
    std::filesystem::remove(video_path);

    const std::filesystem::path forecast_values_path = "/tmp/trtmc-cli-forecast-values.f32";
    const std::filesystem::path forecast_mask_path = "/tmp/trtmc-cli-forecast-mask.f32";
    const float forecast_values[] = {1.0F, 2.0F};
    const float forecast_mask[] = {1.0F, 1.0F};
    {
        std::ofstream values_file(forecast_values_path, std::ios::binary);
        values_file.write(reinterpret_cast<const char*>(forecast_values), sizeof(forecast_values));
        std::ofstream mask_file(forecast_mask_path, std::ios::binary);
        mask_file.write(reinterpret_cast<const char*>(forecast_mask), sizeof(forecast_mask));
    }
    trtmc::cli::Command forecast_command;
    forecast_command.kind = trtmc::cli::CommandKind::kForecast;
    forecast_command.name = "forecast";
    forecast_command.options.emplace("--input", forecast_values_path.string());
    forecast_command.options.emplace("--mask", forecast_mask_path.string());
    forecast_command.options.emplace("--frequency", "7");
    FakeForecast forecast;
    std::ostringstream forecast_output;
    check(trtmc::cli::dispatch(forecast_command, forecast, forecast_output) == 0,
          "forecast task dispatch succeeds");
    check(forecast.seen_frequency == 7, "forecast frequency reaches the Task API");
    std::filesystem::remove(forecast_values_path);
    std::filesystem::remove(forecast_mask_path);

    const std::filesystem::path control_image_path = "/tmp/trtmc-cli-control.ppm";
    const std::filesystem::path control_state_path = "/tmp/trtmc-cli-control-state.f32";
    const std::filesystem::path control_output_path = "/tmp/trtmc-cli-control-actions.f32";
    {
        std::ofstream image_file(control_image_path, std::ios::binary);
        image_file << "P6\n1 1\n255\n";
        const char pixel[] = {char(0), char(127), char(255)};
        image_file.write(pixel, sizeof(pixel));
        std::ofstream state_file(control_state_path, std::ios::binary);
        state_file.write(reinterpret_cast<const char*>(forecast_values), sizeof(forecast_values));
    }
    trtmc::cli::Command control_command;
    control_command.kind = trtmc::cli::CommandKind::kControl;
    control_command.name = "control";
    control_command.options.emplace("--image", control_image_path.string());
    control_command.options.emplace("--state", control_state_path.string());
    control_command.options.emplace("--output", control_output_path.string());
    FakeControl control;
    std::ostringstream control_output;
    check(trtmc::cli::dispatch(control_command, control, control_output) == 0,
          "robot control task dispatch succeeds");
    check(control.seen.image_pixels.size() == 3 && control.seen.state.size() == 2,
          "robot observation reaches the Task API");
    check(std::filesystem::file_size(control_output_path) == 2 * sizeof(float),
          "robot actions are written as float32");
    check(control_output.str().find("\"num_actions\":1") != std::string::npos,
          "robot control result is returned");
    std::filesystem::remove(control_image_path);
    std::filesystem::remove(control_state_path);
    std::filesystem::remove(control_output_path);

    const auto throws_runtime = [](const auto& operation) {
        try {
            operation();
            return false;
        } catch (const std::runtime_error&) {
            return true;
        }
    };
    const std::filesystem::path wav_path = "/tmp/trtmc-cli-io.wav";
    trtmc::AudioResult wav;
    wav.samples = {-1.0F, 0.25F, 1.0F};
    wav.num_samples = 3;
    wav.sample_rate = 16000;
    trtmc::cli::io::write_wav(wav, wav_path.string());
    const auto loaded_wav = trtmc::cli::io::read_wav(wav_path.string());
    check(loaded_wav.sample_rate == 16000 && loaded_wav.samples == wav.samples,
          "float WAV round trip succeeds");
    std::filesystem::remove(wav_path);
    check(throws_runtime([&] { trtmc::cli::io::read_wav(wav_path.string()); }),
          "missing WAV is rejected");
    check(throws_runtime([&] { trtmc::cli::io::write_wav({}, wav_path.string()); }),
          "empty WAV is rejected");

    const std::filesystem::path png_path = "/tmp/trtmc-cli-io.png";
    const std::vector<float> rgb{-0.5F, 0.5F, 1.5F};
    trtmc::cli::io::save_png(png_path.string(), rgb, 1, 1);
    const auto loaded_image = trtmc::cli::io::read_image(png_path.string());
    check(loaded_image.width == 1 && loaded_image.height == 1 && loaded_image.pixels.size() == 3 &&
              loaded_image.pixels[0] == 0.0F &&
              std::abs(loaded_image.pixels[1] - (128.0F / 255.0F)) < 1e-6F &&
              loaded_image.pixels[2] == 1.0F,
          "PNG clamps and round trips RGB pixels");
    std::filesystem::remove(png_path);
    check(trtmc::cli::io::read_image(png_path.string()).empty(),
          "missing image returns an empty result");
    check(throws_runtime([&] { trtmc::cli::io::save_png(png_path.string(), rgb, 0, 1); }),
          "invalid PNG dimensions are rejected");
    check(throws_runtime(
              [&] { trtmc::cli::io::save_png(png_path.string(), std::vector<float>{0.0F}, 1, 1); }),
          "invalid PNG pixel count is rejected");

    trtmc::ImageResult frames;
    frames.pixels = {0.0F, 0.0F, 0.0F, 1.0F, 1.0F, 1.0F};
    frames.width = 1;
    frames.height = 1;
    frames.num_frames = 2;
    trtmc::cli::io::save_png(frames, png_path.string());
    check(std::filesystem::is_regular_file(png_path), "multi-frame image writes its first frame");
    std::filesystem::remove(png_path);
    frames.pixels.clear();
    check(throws_runtime([&] { trtmc::cli::io::save_png(frames, png_path.string()); }),
          "undersized image result is rejected");

    std::cerr << (failures == 0 ? "ALL PASSED\n" : "SOME FAILED\n");
    return failures;
}
