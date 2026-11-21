---
title: Bundle Format
---

A bundle contains eight magic bytes, an unsigned little-endian 64-bit JSON
header length, the UTF-8 JSON header, and concatenated named sections.

```json
{
  "format": 1,
  "family": "gpt2",
  "task": "text_generation",
  "backend": "trt",
  "sections": {
    "runtime.json": {"offset": 0, "length": 42},
    "engine.plan": {"offset": 42, "length": 1234}
  }
}
```

`BundleReader` validates this exact shape and every section bound when it is
constructed. It owns the normalized bundle path and immutable section table,
then reads a requested section directly from the file. It has no write API and
does not eagerly load section contents.

The core does not interpret family sections or compute section hashes. Only
format 1 is supported.

## Build provenance

Section names starting with `trtmc.` are reserved for the core; `BundleWriter`
rejects them from families. When a build records its request with
`BundleWriter.record_build(request)`, as `trtmc build` and family-owned build
commands do, `finish` adds one `trtmc.provenance.json` section. Readers that
predate it ignore the extra section, so the header and runtime contract are
unchanged.

```json
{
  "version": 1,
  "producer": {
    "name": "tensorrt-model-connect",
    "version": "0.1.0",
    "source_revision": "<40-character Git commit or null>",
    "source_dirty": false
  },
  "checkpoint": {"hf_id": "openai-community/gpt2", "revision": "<40-character commit>"},
  "build": {"precision": "fp16", "max_sequence_length": 256, "tensor_parallel_size": 1},
  "toolchain": {"tensorrt_version": "10.16.1.11", "sm": 80, "cpu_arch": "x86_64"}
}
```

- `producer.source_revision` is the checkout commit of the running builder, and
  `source_dirty` reports modified tracked files. Both are `null` for an
  installed package without its own checkout.
- `checkpoint` identifies the primary model directory when it is a Hugging Face
  cache snapshot and is `null` otherwise. Additional checkpoints that a family
  combines are not listed.
- `build` contains the shared request fields that the family request defines,
  such as precision, sequence length, parallelism, quantization, and FP32 layers.
  Family-specific inputs stay in family sections.
- `toolchain` records the TensorRT builder version and GPU SM when the build
  imported TensorRT.

Every field is best effort: information that cannot be determined is recorded
as `null` instead of failing the build. The record contains no paths, host
names, or timestamps. It is unsigned metadata, not a tamper-evidence mechanism.
