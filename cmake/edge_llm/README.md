# Pinned native Edge-LLM package

Edge-LLM is optional. The default `TRTMC_ENABLE_EDGELLM=OFF` neither downloads
nor builds it. Enable it once while installing Model Connect; ordinary model
builds only use the installed package and never fetch or install dependencies.
Cross compilation is rejected. Configure and build on the inference GPU host.

```bash
cmake -S . -B build \
  -DTRTMC_ENABLE_EDGELLM=ON \
  -DTRTMC_EDGELLM_CUDA_ARCHITECTURE=80 \
  -DTRTMC_EDGELLM_TRT_ROOT="$TRT_ROOT" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" \
  -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
  -DCMAKE_INSTALL_PREFIX="$PWD/install"
cmake --build build --parallel 8
cmake --install build
export CMAKE_PREFIX_PATH="$PWD/install${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
```

The regular project dependencies remain required, including nlohmann_json
**3.12.0 with the exact pinned upstream headers** when Edge is enabled. A
development snapshot can retain that version label but change parser layouts;
mixing it with the static SDK causes undefined behavior. The package checks
header content against its vendored dependency (single or multiple headers).
If rejected, install `3rdParty/nlohmannJson` from the pinned Edge checkout into
a separate prefix and configure with that installation’s `nlohmann_json_DIR`. The Python
interpreter needs `ensurepip` or an already installed `virtualenv` bootstrapper.
The native CUDA SDK must include NVCC, NVRTC, cuRAND headers and driver link
libraries; the TensorRT SDK must contain its matching CPython wheel.

The provider first uses `find_package(EdgeLLM 0.10.1 EXACT CONFIG)`. If absent,
CMake `ExternalProject` clones the public NVIDIA TensorRT-Edge-LLM repository at
`e8b29522938901f6df19ebeedd4b69bc8edbcd97` (v0.10.1), initializes the pinned
submodules, builds the native core/plugin and FMHA/GDN CuTe archives, and installs
an isolated direct-builder Python environment. It does not modify the caller
Python environment. Downloads happen only during this
explicit dependency build. `TRTMC_EDGELLM_WHEELHOUSE` selects a complete offline
Python wheelhouse; `TRTMC_EDGELLM_GIT_MIRROR` optionally supplies a local Git
mirror, still checked out at the immutable upstream commit.

Upstream 0.10.1 does not export a CMake SDK package, so these compact templates
supply that installation boundary. `EdgeLLM::Core` exposes the installed static
core, headers, CuTe archive and native dependencies. Consumers requiring CUDA
device linking enable separable compilation and device-symbol resolution.
`EdgeLLM::Plugin` identifies the plugin DSO; adapters load it, rather than linking
it twice. `EdgeLLM_PYTHON_EXECUTABLE` and `EdgeLLM_BUILDER_LAUNCHER` expose the
isolated upstream `experimental.builder.cli.main` API.

`share/trtmc/edge-llm.json` records the pin, native architecture, CUDA/TensorRT
versions and prefix-relative Python/plugin paths. The manifest is written only
after successful installation. Build-tree package files live under
`build/_deps/edgellm/install`; `cmake --install` copies the package into the final
prefix. Package discovery rejects mismatched native CPU/GPU and SDK versions.
Model support and routing policies belong exclusively to the model families.

Set `TRTMC_EDGELLM_ALL_KERNELS=ON` to provision all upstream operator groups
supported by the native GPU. Set `TRTMC_EDGELLM_ONNX=ON` to additionally install
the original Python exporter (including its pinned CPU PyTorch dependencies)
and original C++ `llm_build` executable as `bin/edgellm-onnx-build`. Families invoke
the exporter using the installed Python and obtain the native builder path from
`onnx_builder` in the manifest. These options describe SDK capabilities, not
qualified model support. Reusing an installed package that lacks a requested
capability is an error; no dependency installation occurs during model builds.
The CUDA and TensorRT shared libraries must remain available to the executable.
When building models, select the same native CUDA toolkit with CUDACXX (the
NVCC executable) or CUDAToolkit_ROOT (the SDK root); CUDA_HOME, CUDA_PATH,
and then NVCC on PATH are fallbacks. Platform admission reads this compiler's
release, not the independently versioned cuda-python binding's build toolkit.

Run the existing runtime and family checks against this installation
(some tests require a local GPU):

```bash
ctest --test-dir build --output-on-failure
```

The installed private Python environment exposes its interpreter and modules,
not build-only console/activation scripts containing build-tree paths. Invoke
the exported interpreter with isolated module execution, or use the installed
prefix-relative builder launcher. CMake/Ninja entrypoints remain in the
dependency build environment for reprovisioning. Install into a clean prefix
when replacing an older SDK that included these private console scripts.
Preparation verifies the actual Git checkout against the official pin before
installing dependencies, including on CMake versions with older disconnected
update behavior.

CUDA 12 provisioning requires Python 3.10-3.12. Its pinned CuPy 12.3 kernel
compiler uses a separate build-only environment with NumPy 1.26.4; the installed
SDK and ONNX exporter use NumPy 2.2.6. Both environments must pass pip check.
CUDA 13 uses the SDK environment for kernel compilation. Bootstrap pip is pinned
to 26.2.1 and cuda-python to 12.9.7 (CUDA 12) or 13.3.1 (CUDA 13); these bindings
do not determine the native toolkit identity. Offline wheelhouses must include
these exact pins and the dependencies for both environments. The kernel-only
environment and its dependency report remain under the dependency build root.
CUDA 13 provisioning requires Python 3.10-3.13 because of the pinned NumPy wheel support.
The selected Python wheel and imported TensorRT version must match the exact native SDK header/library version; an ABI-compatible wheel from another release is rejected.
