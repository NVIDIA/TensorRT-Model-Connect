// Unit tests for runtime model plugin lookup/loading.

#include "trtmc/runtime/pipeline_plugin_loader.h"
#include "trtmc/runtime/pipeline_registry.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static int failures = 0;

static void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << std::endl;
        ++failures;
    }
}

static bool contains(const std::vector<std::string>& values, const std::string& needle) {
    return std::find(values.begin(), values.end(), needle) != values.end();
}

static void test_index_maps_strategy_to_model() {
    auto model = trtmc::model_plugin_id_for_strategy("decoder_kv_cache");
    check(model.has_value(), "decoder_kv_cache has model plugin");
    check(model && *model == "text_generation", "decoder_kv_cache maps to text_generation");
    check(trtmc::model_plugin_library_name("text_generation") ==
              "libtrtmc_model_text_generation.so",
          "text_generation library name");
}

static void test_registry_does_not_eager_register_models() {
    auto* plugin = trtmc::PipelineRegistry::instance().lookup("decoder_kv_cache");
    check(plugin == nullptr, "model plugin not registered before explicit load");
}

static void test_unknown_strategy_reports_clean_error() {
    bool threw = false;
    try {
        trtmc::load_model_plugin_for_strategy("__missing_strategy__");
    } catch (const std::runtime_error& e) {
        threw = true;
        check(std::string(e.what()).find("No plugin registered for runtime_strategy") !=
                  std::string::npos,
              "unknown strategy error uses public registry wording");
    }
    check(threw, "unknown strategy throws");
}

static void test_load_text_generation_registers_only_that_model() {
    trtmc::load_model_plugin_for_strategy("decoder_kv_cache");
    auto strategies = trtmc::PipelineRegistry::instance().registered_strategies();
    check(contains(strategies, "decoder_kv_cache"), "decoder_kv_cache registered");
    check(contains(strategies, "decoder_moe"), "decoder_moe registered");
    check(contains(strategies, "nemotron_labs_diffusion"),
          "nemotron_labs_diffusion registered");
    check(!contains(strategies, "diffusion_flux"), "unrelated flux plugin not registered");
    check(!contains(strategies, "speech_to_text"), "unrelated whisper plugin not registered");
}

int main() {
    test_index_maps_strategy_to_model();
    test_registry_does_not_eager_register_models();
    test_unknown_strategy_reports_clean_error();
    test_load_text_generation_registers_only_that_model();

    if (failures > 0) {
        std::cerr << failures << " test(s) FAILED" << std::endl;
        return 1;
    }
    std::cerr << "All model_plugin_loader tests passed" << std::endl;
    return 0;
}
