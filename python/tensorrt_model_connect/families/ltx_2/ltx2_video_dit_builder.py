"""LTX-2 video DiT TRT builder (stub).

Builds the 14B video stream of ``LTX2VideoTransformer3DModel``.

Architectural deltas vs the LTX-Video DiT:
- ``qk_norm = "rms_norm_across_heads"`` (normalise across head axis,
  not head_dim axis).
- Audio-video cross-attention layers interleaved with the video
  self-attention blocks (paired with the audio stream).
- Cross-modality AdaLN for shared timestep conditioning between the
  video and audio streams.
- Caption hidden states arrive at ``caption_channels = 3840`` (already
  projected by the connectors), not the raw T5 ``d_model = 4096``.

Not yet implemented. See ``plugin.LTX2Plugin.build_components`` for the
scaffolding plan.
"""

from __future__ import annotations


def load_ltx2_video_dit_weights(*args, **kwargs):
    raise NotImplementedError(
        "load_ltx2_video_dit_weights: implement once the LTX-2 transformer "
        "checkpoint key map (video stream) is known"
    )


def build_ltx2_video_dit_engine(*args, **kwargs):
    raise NotImplementedError(
        "build_ltx2_video_dit_engine: implement once graph_ops gains "
        "rms_norm_across_heads + audio cross-attention primitives"
    )
