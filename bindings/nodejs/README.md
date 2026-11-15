# Node.js text generation

Build the native libraries from the repository root with CMake, then build
this addon from `bindings/nodejs`:

```sh
cmake --build ../../build --target trtmc_c
npm ci --ignore-scripts
npm run build
```

The default native library directory is the repository's `build` directory.
For another configured build, pass its absolute path to node-gyp:

```sh
npm run build -- -- -Dtrtmc_build_dir=/absolute/native/build
```

Make the native libraries available to the platform's dynamic loader. On Linux,
set `LD_LIBRARY_PATH` to that build directory before starting Node.

`load(bundle, runtimeRoot).generate(prompt, options)` accepts semantic
`text_continuation` bundles through the public Task SDK. Existing bundle modes
keep their original loader; a failed SDK load or execution never retries that
path. The Task handle retains the loaded model between calls.

The existing `max_new_tokens`, `temperature` and `top_k` defaults are 128, 1
and 1 where the Task declares those fields. An explicit unsupported control
is an error. Other SDK controls use the owning family's defaults. Results keep
`text`, `token_ids`, `setup_ms`, `prefill_ms` and `decode_ms`. Timings come from
the native result and exclude JavaScript object construction; they are not
a JavaScript wall-clock measurement or a performance-equivalence claim.

The CPU integration tests require the real dataset/API fixture DSOs and fake
backend. Configure the native build with `TRTMC_BUILD_TESTS=ON` and
`TRTMC_BUILD_EXAMPLES=ON`, then run:

```sh
cmake --build ../../build --target trtmc_dataset_test_family trtmc_api_test_family trtmc_api_test_backend
TRTMC_TEST_RUNTIME_ROOT="$PWD/../../build/tests/api-runtime" npm test
```

These tests exercise model loading, repeated calls, controls, result ownership,
legacy dispatch and error propagation without a GPU. They do not qualify a
checkpoint. Windows and macOS addon builds require separate platform validation.
