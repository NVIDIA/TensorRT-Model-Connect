{
  "targets": [
    {
      "target_name": "trtmc_node",
      "sources": [
        "src/addon.cpp",
        "src/pipeline_wrapper.cpp"
      ],
      "include_dirs": [
        "<!@(node -p \"require('node-addon-api').include\")",
        "../../include"
      ],
      "dependencies": [
        "<!(node -p \"require('node-addon-api').gyp\")"
      ],
      "cflags!": [ "-fno-exceptions" ],
      "cflags_cc!": [ "-fno-exceptions" ],
      "xcode_settings": {
        "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
        "CLANG_CXX_LIBRARY": "libc++",
        "MACOSX_DEPLOYMENT_TARGET": "10.15"
      },
      "msvs_settings": {
        "VCCLCompilerTool": {
          "ExceptionHandling": 1,
          "AdditionalOptions": ["/std:c++17"]
        }
      },
      "conditions": [
        ['OS=="win"', {
          "libraries": [
            "-ltensorrt_model_connect.lib"
          ],
          "library_dirs": [
            "../../build/Release"
          ]
        }],
        ['OS!="win"', {
          "libraries": [
            "-L../../build",
            "-ltensorrt_model_connect"
          ]
        }]
      ]
    }
  ]
}
