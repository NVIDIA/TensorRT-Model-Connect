---
title: Bring Your Own Kernel with TVM FFI
---

BYOK changes a TensorRT graph and loads external native code, so it is an
advanced topic. Start with the complete
[Bring Your Own Kernel lab](../advanced/bring-your-own-kernel.md).

The current path uses the direct `tensorrt_model_connect.byok.add_kernel()` API
and the one public pre-serialization `graph_transform` callback. It does not use
a Recipe registry, graph snapshot command, binding manifest, or hash.
