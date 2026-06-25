# Core Runtime

Shared TensorRT/CUDA runtime building blocks used by all strategy backends.

Key files:
- `trt_common.*`: TRT logging, CUDA helpers, shared utilities.
- `trt_engine_lifecycle.*`: engine/context creation and tensor wiring checks.
- `trt_decode_runtime.*`: common decode-time utilities (sampling, masks).
- `trt_backend_shared.*`: common decoder generation loop utilities.
- `step_state.h`: generic recurrent-step state interface.
- `stb_impl.cpp`: stb implementation unit for image loading/resizing.

Read this first when debugging tensor binding, CUDA copies, or generic decode flow issues.
