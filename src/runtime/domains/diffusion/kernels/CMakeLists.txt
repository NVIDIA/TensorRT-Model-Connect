# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Diffusion kernel modules.
#
# Each kernel here is built as a TVM-FFI loadable module (a .so file with no
# host-side ABI other than the TVM-FFI registration done at static-init time).
# The diffusion TRT graph picks them up via graph_ops.add_tvm_ffi_kernel().
#
# Gated by the root option TRTMC_BUILD_DIFFUSION_KERNELS so a stock build is
# unaffected.

enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

add_library(trtmc_dit_rms_norm_rope MODULE
    dit_rms_norm_rope.cu
)

set_target_properties(trtmc_dit_rms_norm_rope PROPERTIES
    CUDA_ARCHITECTURES "90;100;103"  # Hopper (sm_90) + Blackwell (sm_100, sm_103=GB300)
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""
    OUTPUT_NAME "trtmc_dit_rms_norm_rope"
    CXX_STANDARD 17
    CUDA_STANDARD 17
)

target_include_directories(trtmc_dit_rms_norm_rope SYSTEM PRIVATE
    ${TRTMC_TVM_FFI_INCLUDE_DIR}
)

target_link_libraries(trtmc_dit_rms_norm_rope PRIVATE
    CUDA::cudart
    ${TRTMC_TVM_FFI_LIBRARY}
)

target_compile_options(trtmc_dit_rms_norm_rope PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>
)

message(STATUS "Diffusion kernel module: trtmc_dit_rms_norm_rope (sm_90, sm_100)")
