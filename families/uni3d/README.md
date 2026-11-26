# Uni3D

This family builds the Uni3D-B point encoder (`BAAI/Uni3D`,
`modelzoo/uni3d-b/model.pt`) for `points_to_embedding` as one TensorRT engine.
A colored point cloud `[N, 6]` (xyz, rgb) becomes one 1024-d embedding in the
EVA02-E-14-plus CLIP space, so it can be compared with CLIP text or image
embeddings for zero-shot classification and retrieval. The CLIP encoders stay
outside the engine.

## Graph

The engine is the upstream `encode_pc` path:

1. Farthest point sampling of 512 centers. Upstream uses the `pointnet2_ops`
   CUDA kernel; the family exports a plain PyTorch transcription of it with the
   same rules: it starts at point 0, never samples a point whose squared norm
   is `<= 1e-3`, and takes the first index on a tie.
2. kNN grouping (64 points per center), centered on the sampled point.
3. The mini-PointNet tokenizer and the 12 EVA02-B blocks from timm
   `eva02_base_patch14_448`, without rotary position embedding, as upstream
   calls them.

The whole graph is standard ONNX ops imported with the TensorRT ONNX parser,
with no plugins or family-owned kernels. Grouping always runs in float32 so the
sampled centers and neighborhoods match the reference; the `fp16` export runs
the tokenizer and the ViT in half precision.

## Input

Points as upstream evaluates them: xyz centered and scaled to the unit sphere,
rgb in `[0, 1]` (upstream uses 0.4 for clouds without color). `N` is between
512 and 10,000; the checkpoint was trained on 10,000.

## Runtime

The family implements `points_to_embedding`. It accepts `input_dim == 6` and
512 to 10,000 points and rejects anything else, including an empty cloud,
before the engine runs. The row-major request buffer is the engine layout
`[1, N, 6]`, so it is passed without a copy. The result is the class-token
embedding, l2-normalized (`pooling = "class_token"`, `normalization = "l2"`,
`embedding_space = "open_clip/EVA02-E-14-plus"`).

## Tests

- `tests/test_reference.py`: the FPS against a transcription of the
  `pointnet2_ops` kernel, and the TensorRT engine against PyTorch on random
  weights (fp32 and fp16).
- `tests/cpp/test_task_contract.cpp`: request checks, layout and result
  fields of the pipeline on a recording module (CPU).
- `tests/test_e2e.py`: the upstream checkpoint exported, built and run
  through the public C table by `uni3d_qualification`, compared with the
  PyTorch embedding by cosine similarity on a generated colored point cloud,
  so no dataset is redistributed.

## Model directory

`tests/official_reference.py` exports the upstream checkpoint to the builder
contract, `uni3d.onnx` plus `config.json`. The export fixes the precision, and
the build refuses a request whose precision differs from it.
