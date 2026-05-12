# MR-2 Worklog: Extract All 16 Plugins

## Branch: `MR2-extract-plugins`
## Base: `runtime-plugin-implementation` (MR-1 committed as 4c07799)

## Status: IN PROGRESS

---

## Summary

Extract all `create_*()` factory functions from `pipeline_factory.cpp` into
self-contained plugin files in `src/runtime/plugins/`. Each plugin:
1. Parses its own config from `ctx.config_json` via `json_helpers.h`
2. Extracts its own sections via `find_section(ctx.bundle, "name")`
3. Uses shared utilities from `plugin_helpers.h`
4. Ends with `REGISTER_PIPELINE_PLUGIN_MULTI(...)` or `REGISTER_PIPELINE_PLUGIN(...)`

After extraction, `pipeline_factory.cpp` shrinks to ~80-100 LOC with
registry-first lookup and no strategy-specific code.

## Phases (from TASK-05 plan)

### Phase 3: decoder_plugin.cpp (proof-of-concept)
- [ ] Create `src/runtime/plugins/shared/plugin_helpers.h` with shared utilities
- [ ] Create `src/runtime/plugins/decoder_plugin.cpp`
- [ ] Add registry-first lookup to `pipeline_factory.cpp::from_bundle()`
- [ ] Add to CMakeLists.txt
- [ ] Verify: build + ctest

### Phase 4: Remaining text-family plugins
- [ ] `src/runtime/plugins/ssm_plugin.cpp`
- [ ] `src/runtime/plugins/rwkv_plugin.cpp`
- [ ] `src/runtime/plugins/hybrid_plugin.cpp`
- [ ] Remove `create_text_pipeline()` from pipeline_factory.cpp

### Phase 5: Encoder + vision plugins
- [ ] `src/runtime/plugins/encoder_plugin.cpp`
- [ ] `src/runtime/plugins/segmentation_plugin.cpp`
- [ ] `src/runtime/plugins/object_detection_plugin.cpp`
- [ ] `src/runtime/plugins/vl_plugin.cpp`
- [ ] Remove `create_encoder_pipeline()`, `create_vision_pipeline()`

### Phase 6: Diffusion plugins
- [ ] `src/runtime/plugins/shared/diffusion_helpers.h`
- [ ] `src/runtime/plugins/flux_plugin.cpp`
- [ ] `src/runtime/plugins/wan_plugin.cpp`
- [ ] `src/runtime/plugins/zimage_plugin.cpp`
- [ ] `src/runtime/plugins/pixart_plugin.cpp` (shares WanPipeline)
- [ ] Remove `create_diffusion_pipeline()` and helpers

### Phase 7: Audio plugins
- [ ] `src/runtime/plugins/shared/audio_helpers.h`
- [ ] `src/runtime/plugins/whisper_plugin.cpp`
- [ ] `src/runtime/plugins/bark_plugin.cpp`
- [ ] `src/runtime/plugins/magpie_plugin.cpp`
- [ ] `src/runtime/plugins/speech_plugin.cpp`
- [ ] `src/runtime/plugins/omni_plugin.cpp`
- [ ] Remove `create_audio_pipeline()`, `create_omni_pipeline()`

### Phase 8: Shrink pipeline_factory.cpp
- [ ] Replace `from_bundle()` with registry-only dispatch (~50 LOC)
- [ ] Remove `dispatch_pipeline()`, `resolve_family()`, `StrategyFamily`
- [ ] Remove `normalize_legacy_strategy()` (move to plugin_helpers or inline)
- [ ] Remove `parse_bundle_config()` (use `parse_base_config()`)

### Final verification
- [ ] cmake --build build -j && ctest --test-dir build --output-on-failure
- [ ] python tools/check_cyclomatic_complexity.py src --max-ccn 10
- [ ] Python builder tests pass
- [ ] Python tools tests pass

---

## Log

### 2026-03-13: Branch created
- Created `MR2-extract-plugins` from `runtime-plugin-implementation`
- MR-1 provides: IPipelinePlugin, PipelineRegistry, BaseConfig, BundleView,
  split strategy strings, normalize_legacy_strategy()

### 2026-03-13: All 16 plugins extracted
- Created shared utility headers: plugin_helpers.h/.cpp, diffusion_helpers.h/.cpp, audio_helpers.h/.cpp
- Created all 16 plugin files in src/runtime/plugins/
- Added force_link_plugins.cpp to prevent linker stripping of self-registering statics
- Gutted pipeline_factory.cpp: 1228 LOC → 126 LOC (registry-only dispatch)
- Updated test_c_abi_runtime_regression.cpp error message expectations
- All verification passed:
  - C++ build: clean
  - C++ unit tests: 65/65 passed
  - CCN gate: PASS (max=10)
  - Python builder tests: 1083 passed, 15 skipped
  - Python tools tests: 242 passed, 1 skipped
