# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Pillow decodes JPEG with libjpeg-turbo. The native media reader builds the
# same release from the unmodified decoder sources in
# third_party/libjpeg-turbo, so no system JPEG library is needed at build or
# run time. Only the libjpeg decompression API is built, without SIMD: SIMD
# changes speed, not the decoded pixels.

set(_trtmc_jpeg_source "${PROJECT_SOURCE_DIR}/third_party/libjpeg-turbo/src")
set(_trtmc_jpeg_generated "${CMAKE_BINARY_DIR}/third_party/libjpeg-turbo")

# The values upstream's build substitutes into its configuration headers for a
# default (libjpeg v6.2 API) Linux build with arithmetic decoding.
function(_trtmc_configure_libjpeg_turbo)
  set(JPEG_LIB_VERSION 62)
  set(VERSION 3.1.4.1)
  set(LIBJPEG_TURBO_VERSION_NUMBER 3001004)
  set(COPYRIGHT_YEAR "1991-2026")
  set(BUILD "trtmc")
  set(CMAKE_PROJECT_NAME "libjpeg-turbo")
  set(HIDDEN "__attribute__((visibility(\"hidden\")))")
  set(INLINE "__inline__ __attribute__((always_inline))")
  set(THREAD_LOCAL "__thread")
  set(SIZE_T ${CMAKE_SIZEOF_VOID_P})
  set(HAVE_BUILTIN_CTZL 1)
  set(D_ARITH_CODING_SUPPORTED 1)
  foreach(_header IN ITEMS jconfig.h jconfigint.h jversion.h)
    configure_file("${_trtmc_jpeg_source}/${_header}.in" "${_trtmc_jpeg_generated}/${_header}")
  endforeach()
endfunction()
_trtmc_configure_libjpeg_turbo()

# Upstream compiles the precision-generic modules once per sample size through
# the wrapper files; the decoder needs all of them to link.
add_library(trtmc_libjpeg_turbo STATIC
  ${_trtmc_jpeg_source}/jaricom.c
  ${_trtmc_jpeg_source}/jcomapi.c
  ${_trtmc_jpeg_source}/jdapimin.c
  ${_trtmc_jpeg_source}/jdarith.c
  ${_trtmc_jpeg_source}/jdatasrc.c
  ${_trtmc_jpeg_source}/jdhuff.c
  ${_trtmc_jpeg_source}/jdinput.c
  ${_trtmc_jpeg_source}/jdlhuff.c
  ${_trtmc_jpeg_source}/jdmarker.c
  ${_trtmc_jpeg_source}/jdmaster.c
  ${_trtmc_jpeg_source}/jdphuff.c
  ${_trtmc_jpeg_source}/jerror.c
  ${_trtmc_jpeg_source}/jmemmgr.c
  ${_trtmc_jpeg_source}/jmemnobs.c
)
foreach(_module IN ITEMS jdapistd jdcolor jddiffct jdlossls jdmainct jdpostct jdsample jutils)
  foreach(_bits IN ITEMS 8 12 16)
    target_sources(trtmc_libjpeg_turbo PRIVATE "${_trtmc_jpeg_source}/wrapper/${_module}-${_bits}.c")
  endforeach()
endforeach()
foreach(_module IN ITEMS jdcoefct jddctmgr jdmerge jidctflt jidctfst jidctint jidctred jquant1
                         jquant2)
  foreach(_bits IN ITEMS 8 12)
    target_sources(trtmc_libjpeg_turbo PRIVATE "${_trtmc_jpeg_source}/wrapper/${_module}-${_bits}.c")
  endforeach()
endforeach()
target_include_directories(trtmc_libjpeg_turbo SYSTEM PUBLIC
  "${_trtmc_jpeg_generated}"
  "${_trtmc_jpeg_source}"
)
# Upstream builds with the compiler's GNU C dialect (it uses POSIX setenv) and
# keeps its symbols out of any shared object that links this archive.
set_target_properties(trtmc_libjpeg_turbo PROPERTIES
  C_EXTENSIONS ON
  C_VISIBILITY_PRESET hidden
)
