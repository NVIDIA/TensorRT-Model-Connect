#include "trtmc/config/schema_registry.h"
#include "trtmc/runtime/pipeline_plugin_loader.h"

#include <iostream>

static int failures = 0;

static void check(bool condition, const char* name) {
    if (!condition) {
        std::cerr << "FAIL: " << name << '\n';
        ++failures;
    }
}

int main() {
    auto& schemas = trtmc::config::SchemaRegistry::instance();
    check(schemas.lookup("audio_bark") == nullptr, "audio_bark schema absent from core");

    trtmc::load_model_plugin_for_strategy("text_to_audio_bark");
    const auto* schema = schemas.lookup("audio_bark");
    check(schema != nullptr, "bark plugin registers audio_bark schema");
    check(schema != nullptr && schema->fields.size() == 4, "audio_bark field count");

    if (failures > 0) {
        std::cerr << failures << " bark config schema test(s) failed\n";
        return 1;
    }
    return 0;
}
