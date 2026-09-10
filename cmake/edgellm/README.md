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
**3.12.0** when Edge is enabled (its C++ ABI must match upstream). The Python
interpreter needs `ensurepip` or an already installed `virtualenv` bootstrapper.
The native CUDA SDK must include NVCC, NVRTC, cuRAND headers and driver link
libraries; the TensorRT SDK must contain its matching CPython wheel.

The provider first uses `find_package(EdgeLLM 0.10.1 EXACT CONFIG)`. If absent,
CMake `ExternalProject` clones the public NVIDIA TensorRT-Edge-LLM repository at
`e8b29522938901f6df19ebeedd4b69bc8edbcd97` (v0.10.1), initializes the pinned
submodules, builds the native core/plugin and FMHA/GDN CuTe archives, and installs
an isolated direct-builder Python environment. It does not install the exporter
or modify the caller Python environment. Downloads happen only during this
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

GPU-independent package contract tests:

```bash
cmake -P cmake/edgellm/tests/package_contract.cmake
```
