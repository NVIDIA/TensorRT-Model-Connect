# PatchTSMixer Task SDK checks

PatchTSMixer implements `ISeriesToPointForecast` and exposes
`series_to_point_forecast` through the public C API and header-only C++ wrapper.
All model-specific implementation and validation live in this family directory.

The input is a time-major float32 history. An explicit matrix shape is
`[time, channel]`; a flat history lets the family resolve its configured channel
count. Short histories are left-padded with unobserved zeros, and long histories
keep their newest complete timesteps. The optional binary observation mask
follows the same padding and cropping.

The result owns every forecast value in `[horizon, channel]` order and has
one-based `horizon_steps`. The old singleton batch dimension is not part of this
single-series Task. The family declares `frequency` as an integer with default
zero and rejects nonzero values, matching its existing model behavior. There is
no legacy Task alias or batch capability declaration.

## Build the checks

Configure the repository normally with `TRTMC_BUILD_TESTS=ON`, then run:

```sh
cmake --build build --target test_patchtsmixer_forecast_contract
ctest --test-dir build -R '^patchtsmixer_forecast_contract$' --output-on-failure
```

The contract target also builds `test_patchtsmixer_c_api_consumer` and
`test_patchtsmixer_cpp_api_consumer` in the build directory. Their source files
use only public SDK headers. They read the official Granite checkpoint's
seven-channel input layout, print all 96 x 7 forecast values and horizon axes,
and access the result after releasing the input and model handles.

## Run the real checkpoint check

Use the existing family E2E environment: `TRTMC_BINARY` points to the native
CLI, `TRTMC_RUNTIME_ROOT` to its runtime libraries, and `TRTMC_NATIVE_BUILD_DIR`
to the configured build directory. Cache the checkpoint from the family
manifest first, or set the existing `TRTMC_PATCHTSMIXER_MODEL_DIR` override.

```sh
python -m pytest families/patchtsmixer/tests/test_e2e.py \
  --e2e-testcase patchtsmixer-granite-official -q
```

This builds the bundle and checks the CLI, C consumer, and C++ consumer against
the same official reference and unchanged numerical thresholds. Each comparison
covers the complete output and its axes. The C and C++ consumer checks run for
single-device cases; the existing TP4 manifest and CLI/reference checks remain
separate and unchanged.

The consumers can also run directly against that bundle and a binary float32
history containing a positive number of complete seven-channel timesteps:

```sh
build/test_patchtsmixer_c_api_consumer model.bundle build values.f32
build/test_patchtsmixer_cpp_api_consumer model.bundle build values.f32
```

Replace `build` in the runtime-root argument if the libraries are installed
elsewhere. Neither consumer requires an internal C++ header or links a family
DSO directly; the public runtime loads the family selected by the bundle.
