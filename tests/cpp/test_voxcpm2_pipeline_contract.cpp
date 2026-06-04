// =============================================================================
// ISO 26262 Traceability
// =============================================================================
// Trace ID:       UT-AUD-CPP-VOXCPM2-02
// Architecture:   ARCH-FAC-001
// Unit Design:    UD-AUD-VOXCPM2-02
// Intent:         VoxCPM2Pipeline runtime boundary construction and
//                 generate_audio() contract before native stage execution.
// Preconditions:  No TensorRT SDK required; fake backend-neutral modules.
// Postconditions: Pipeline validates LocEnc->TSLM->RALM->LocDiT->AudioVAE
//                 module order and reports the exact missing WAV-producing
//                 execution step at generate time.
// =============================================================================

#include "runtime/models/voxcpm2/pipeline.h"

#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace audio = trtmc::runtime::builders::audio;

int failures = 0;

void check(bool condition, const char* test_name) {
    if (!condition) {
        std::cerr << "FAIL: " << test_name << '\n';
        ++failures;
    }
}

class FakeModule final : public trtmc::ITrtModule {
  public:
    trtmc::TensorMap forward(const trtmc::TensorMap&) override { return {}; }
    trtmc::DeviceTensorMap forward_device(const trtmc::DeviceTensorMap&) override { return {}; }
    void forward_device_async(const trtmc::DeviceTensorMap&) override {}
    void forward_async(const trtmc::TensorMap&) override {}
    void sync() override {}
    cudaStream_t stream() const override { return nullptr; }
    void enable_cuda_graph() override {}
    bool cuda_graph_active() const override { return false; }
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
    bool ok() const override { return true; }
    void keep_alive(std::shared_ptr<void>) override {}
};

std::vector<audio::VoxCPM2LoadedComponent> make_fake_components() {
    std::vector<audio::VoxCPM2LoadedComponent> components;
    components.reserve(audio::kVoxCPM2ComponentSpecs.size());
    for (const auto& spec : audio::kVoxCPM2ComponentSpecs) {
        std::unique_ptr<trtmc::ITrtModule> module = std::make_unique<FakeModule>();
        components.push_back({spec.name, spec.engine_section, std::move(module)});
    }
    return components;
}

void test_constructs_with_loaded_component_contract() {
    trtmc::VoxCPM2Config cfg;
    auto plan = audio::make_voxcpm2_generation_plan(cfg);
    trtmc::VoxCPM2Pipeline pipeline(make_fake_components(), plan, "openbmb/VoxCPM2");

    check(std::string(pipeline.pipeline_type()) == "VoxCPM2Pipeline",
          "voxcpm2 pipeline type is explicit");
    check(std::string(pipeline.model_id()) == "openbmb/VoxCPM2",
          "voxcpm2 pipeline preserves model id");
}

void test_generate_audio_reports_missing_native_execution() {
    trtmc::VoxCPM2Config cfg;
    auto plan = audio::make_voxcpm2_generation_plan(cfg);
    trtmc::VoxCPM2Pipeline pipeline(make_fake_components(), plan, "openbmb/VoxCPM2");

    trtmc::GenerateConfig gen_cfg;
    gen_cfg.cfg_scale = 3.0F;
    gen_cfg.num_steps = 12;
    gen_cfg.seed = 7;

    try {
        (void)pipeline.generate_audio("VoxCPM2 parity prompt", gen_cfg);
        check(false, "voxcpm2 generate_audio throws until native execution exists");
    } catch (const std::runtime_error& e) {
        const std::string message = e.what();
        check(message.find("LocEnc -> TSLM -> RALM -> LocDiT -> AudioVAE") !=
                  std::string::npos,
              "voxcpm2 generate error names native stage order");
        check(message.find("trt_output.wav") != std::string::npos,
              "voxcpm2 generate error preserves TRT WAV artifact name");
        check(message.find("cfg_value=3") != std::string::npos,
              "voxcpm2 generate error applies GenerateConfig cfg override");
        check(message.find("inference_timesteps=12") != std::string::npos,
              "voxcpm2 generate error applies GenerateConfig step override");
        check(message.find("seed=7") != std::string::npos,
              "voxcpm2 generate error applies GenerateConfig seed override");
    }
}

void test_rejects_component_order_mismatch() {
    trtmc::VoxCPM2Config cfg;
    auto plan = audio::make_voxcpm2_generation_plan(cfg);
    auto components = make_fake_components();
    components[2].name = "locdit";

    try {
        trtmc::VoxCPM2Pipeline pipeline(std::move(components), plan, "openbmb/VoxCPM2");
        (void)pipeline;
        check(false, "voxcpm2 pipeline rejects component order mismatch");
    } catch (const std::runtime_error& e) {
        const std::string message = e.what();
        check(message.find("loaded component order does not match generation plan") !=
                  std::string::npos,
              "voxcpm2 pipeline reports component order mismatch");
    }
}

} // namespace

int main() {
    test_constructs_with_loaded_component_contract();
    test_generate_audio_reports_missing_native_execution();
    test_rejects_component_order_mismatch();

    if (failures != 0) {
        std::cerr << failures << " VoxCPM2 pipeline contract test(s) failed\n";
        return 1;
    }
    std::cerr << "All VoxCPM2 pipeline contract tests passed.\n";
    return 0;
}
