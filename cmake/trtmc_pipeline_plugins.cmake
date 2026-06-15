# Declarative runtime pipeline plugin manifest.
#
# Each entry is:
#   source/path/from/src/runtime.cpp|registration_function
#
# CMake consumes this list for both compilation and generated registrar calls,
# so a plugin is added in one place instead of editing parallel source lists.

include("${CMAKE_CURRENT_LIST_DIR}/trtmc_registration_manifest.cmake")

set(TRTMC_PIPELINE_PLUGINS
  "models/text_generation/plugin.cpp|register_decoder_plugin"
  "models/recurrent/ssm_plugin.cpp|register_ssm_plugin"
  "models/recurrent/rwkv_plugin.cpp|register_rwkv_plugin"
  "models/recurrent/hybrid_plugin.cpp|register_hybrid_plugin"
  "models/encoder/plugin.cpp|register_encoder_plugin"
  "models/elf_flow/plugin.cpp|register_elf_flow_plugin"
  "models/segmentation/plugin.cpp|register_segmentation_plugin"
  "models/timm_vit/plugin.cpp|register_timm_vit_plugin"
  "models/encoder/object_detection_plugin.cpp|register_object_detection_plugin"
  "models/vision_language/plugin.cpp|register_vl_plugin"
  "models/whisper/plugin.cpp|register_whisper_plugin"
  "models/rnnt/plugin.cpp|register_rnnt_plugin"
  "models/bark/plugin.cpp|register_bark_plugin"
  "models/magpie/plugin.cpp|register_magpie_plugin"
  "models/speech/plugin.cpp|register_speech_plugin"
  "models/omni/plugin.cpp|register_omni_plugin"
  "models/flux/plugin.cpp|register_flux_plugin"
  "models/ltx_video/plugin.cpp|register_ltx_video_plugin"
  "models/wan/plugin.cpp|register_wan_plugin"
  "models/z_image/plugin.cpp|register_zimage_plugin"
  "models/qwen_image/plugin.cpp|register_qwen_image_plugin"
  "models/t5/plugin.cpp|register_t5_plugin"
  "models/marian/plugin.cpp|register_marian_plugin"
  "models/seq2seq/plugin.cpp|register_seq2seq_plugin"
  "models/pixart/plugin.cpp|register_pixart_plugin"
  "models/chronos_bolt/plugin.cpp|register_chronos_bolt_plugin"
  "models/patchtsmixer/plugin.cpp|register_patchtsmixer_plugin"
  "models/patchtst/plugin.cpp|register_patchtst_plugin"
  "models/timesfm/plugin.cpp|register_timesfm_plugin"
)

set(TRTMC_PIPELINE_PLUGIN_REGISTRATION_SOURCE
  "${PROJECT_BINARY_DIR}/generated/register_plugins.cpp")
trtmc_configure_registration_manifest(
  TRTMC_PIPELINE_PLUGINS
  "${PROJECT_SOURCE_DIR}/src/runtime"
  "${CMAKE_CURRENT_LIST_DIR}/register_plugins.cpp.in"
  "${TRTMC_PIPELINE_PLUGIN_REGISTRATION_SOURCE}"
  TRTMC_PIPELINE_PLUGIN_SOURCES
  TRTMC_PIPELINE_PLUGIN_REGISTRATION_DECLS
  TRTMC_PIPELINE_PLUGIN_REGISTRATION_CALLS
  "    "
  "::trtmc::PipelineRegistry"
)
