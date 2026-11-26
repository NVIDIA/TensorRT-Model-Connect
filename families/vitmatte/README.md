# ViTMatte

The FP32 image/trimap-to-alpha path uses TensorRT for its attention, convolutions,
normalization, and decoder. The family binds two CUDA operations through TVM-FFI
BYOK: exact erf GELU and the MLP output projection with FP32 accumulation and bias.
Small rounding differences at these boundaries are amplified by subsequent
low-variance normalization. TF32 is disabled for the projection on each call.

Build with `TRTMC_ENABLE_BYOK=ON` and the family's CUDA-enabled PyTorch dependency
installed. The native family library links LibTorch; native inference does not
start Python or load the Transformers reference model. Both the family library
and `libtrtmc_byok_tvm_ffi.so` must be in the runtime root. The builder searches
standard CMake prefixes and the installed wheel's `bin` directory. Include the
native build directory in `CMAKE_PREFIX_PATH` when building from source. These paths are not serialized
into the engine; the native family registers its own kernels before loading it.

The pinned `vitmatte-small-alpha` E2E compares both categorical trimaps against
the independent Transformers reference with maximum alpha error `5e-5` and RMSE
`1e-5`. The FP32 operations do not depend on a checkpoint, input image, GPU model,
or TensorRT version.

The family validates image dimensions, channels, storage and trimap alignment
against the loaded profile before shared Task dispatch scans categorical trimap
values. Off-profile requests fail before input-dependent payload work or GPU
inference. Valid requests retain the existing preprocessing and numerical gates.
The internal matting producer interface includes a metadata-only
`validate_request` method; rebuild family plugins with the matching core runtime
when updating this interface. Public C request and result layouts are unchanged.

Explicit selection with `-DTRTMC_BUILD_FAMILIES=vitmatte` checks the family
prerequisites at configuration time. Default builds may skip the family when
BYOK is disabled or Torch is unavailable; explicitly building
`trtmc_model_vitmatte` then reports the missing prerequisite.

Kernel registration retries reuse each completed binding and retry only the
failed binding. The shared backend propagates device-to-host transfer errors
before returning tensors backed by reusable staging storage.
Registration error messages use bounded thread-local storage without allocating
inside the `noexcept` error handler.

BYOK bindings use process-global, versioned kernel names. Use a matching native
core and family build in one process; loading different builds under the same
names is not a qualified isolation boundary. The shared loader permits an
explicit replacement: already resolved function handles retain their callable,
while later unresolved lookups see the replacement. A changed kernel contract
requires a new versioned name. Successful bindings and their libraries remain
loaded for the process lifetime, including after a later registration or engine
load fails. Family registration therefore does not promise transaction rollback.
