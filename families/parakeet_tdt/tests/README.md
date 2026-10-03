# CPU migration tests

`config.json` is NVIDIA's configuration from
[`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/blob/541d1f99c6b0c3cd0b11a95167540bb8edefd82b/config.json),
revision `541d1f99c6b0c3cd0b11a95167540bb8edefd82b`, retrieved September 18, 2026.
The publisher lists the checkpoint under CC BY 4.0. No model weights are included.

The bundle-composition tests replace TensorRT compilation with byte payloads.
They exercise the real BuildRequest, family dispatch, configuration validation,
and BundleWriter. They do not establish engine validity, numerical equivalence,
transcript quality, GPU execution, or performance.

`test_native.py` compiles five CPU executables with a C++17 compiler (`CXX`,
default `c++`): Task audio-input validation/downmix/resampling, the existing
mel/FFT/incremental-resampling tests, and the existing TDT duration/geometry
tests, semantic transcription pipeline orchestration with fake engines, and BPE
Metaspace decoding. The original mel and duration assertions from PR #1060 are
retained. Pipeline tests need CUDA headers (`CUDA_HOME` or `CUDA_PATH`) but use
no GPU or CUDA/TensorRT libraries. Tokenizer tests need `nlohmann/json.hpp`
(`NLOHMANN_JSON_INCLUDE_DIR`, default `/usr/include`). Missing headers produce
explicit skips, not passing evidence.

The native pipeline now connects the audio helper to encoder, predictor, joint,
and tokenizer calls. Tests check request state reset, invalid options, malformed
or missing engine outputs, and recovery after failed requests.

`test_factory.py` writes real bundles with the shared Python `BundleWriter`,
then compiles and executes the real C++ `BundleReader` and family factory.
Its 22 cases cover semantic task binding, engine section order/content,
invalid identity/config/tokenizer/frontend data, missing sections, failed engine
loads, and destruction of modules after partial construction failures. Engine
loading is faked; CUDA headers and nlohmann JSON are required as above. This
does not prove TensorRT deserialization, engine validity, model transcript parity,
or streaming support.

An additional September 19, 2026 check compared the pinned official tokenizer
above against Hugging Face `tokenizers==0.22.2`: all 8,193 singleton vocabulary
IDs, 1,000 deterministic random token sequences, and two special/empty cases
matched (9,195 cases). It caught and fixed incorrect conversion of the literal
`Ġ` character to a space. The synthetic regression remains in the repository;
this broader check requires downloading the official tokenizer JSON, not weights.

From the repository root:

```sh
PYTHONPATH=core/builder:. python -m pytest families/parakeet_tdt/tests -q
```

The selected E2E builds `test_parakeet_tdt_sdk_cpp` in the configured
`TRTMC_NATIVE_BUILD_DIR` before invoking the public SDK consumer. Community GPU
builds the family DSO and registered CTests, so an unregistered consumer cannot
be assumed to exist. `test_gpu_ci.py` checks the selected pinned checkpoint and
uses a real minimal CMake build to verify the consumer helper. Checkpoint
identity tests live in `test_support.py` for public CPU discovery.
