# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

{
  "variables": {
    "trtmc_build_dir%": "<(module_root_dir)/../../build"
  },
  "targets": [
    {
      "target_name": "trtmc_node",
      "sources": [
        "src/addon.cpp",
        "src/pipeline_wrapper.cpp"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "../../core/runtime/include",
        "../../core/api/include",
        "../../apps"
      ],
      "dependencies": [
        "<!(node -p \"require('node-addon-api').gyp\")"
      ],
      "cflags!": [ "-fno-exceptions" ],
      "cflags_cc!": [ "-fno-exceptions", "-fno-rtti" ],
      "cflags_cc": [ "-std=c++17" ],
      "xcode_settings": {
        "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
        "GCC_ENABLE_CPP_RTTI": "YES",
        "CLANG_CXX_LIBRARY": "libc++",
        "MACOSX_DEPLOYMENT_TARGET": "10.15"
      },
      "msvs_settings": {
        "VCCLCompilerTool": {
          "ExceptionHandling": 1,
          "RuntimeTypeInfo": "true",
          "AdditionalOptions": ["/std:c++17"]
        }
      },
      "conditions": [
        ['OS=="win"', {
          "libraries": [
            "-ltrtmc_c.lib",
            "-ltrtmc_runtime.lib",
            "-ltrtmc_core.lib"
          ],
          "library_dirs": [
            "<(trtmc_build_dir)/Release"
          ]
        }],
        ['OS!="win"', {
          "libraries": [
            "-L<(trtmc_build_dir)",
            "-ltrtmc_c",
            "-ltrtmc_runtime",
            "-ltrtmc_core"
          ]
        }]
      ]
    }
  ]
}
