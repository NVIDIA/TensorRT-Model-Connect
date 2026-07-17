# cuDNN frontend source snapshot

This directory vendors the header-only NVIDIA cuDNN frontend used by the
Wan2.2 source-exact SDPA TensorRT plugin.

- Upstream: `https://github.com/NVIDIA/cudnn-frontend`
- Tag: `v1.22.1`
- Commit: `a91f0e04dcea10515f0f776fc5a89535e316a9c8`
- Scope: upstream `include/` directory, unmodified
- License: `LICENSE.txt`

The production plugin has a compile-time version check for 1.22.1. Updating
this snapshot requires repeating the self- and cross-attention bitwise
qualification on every supported target.
