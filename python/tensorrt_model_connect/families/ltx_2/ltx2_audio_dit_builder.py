"""LTX-2 audio DiT TRT builder (stub).

Builds the 5B audio stream of ``LTX2VideoTransformer3DModel`` (the
audio branch may live inside the same transformer/ subfolder as the
video branch or in a sibling ``transformer/audio`` directory — TBD on
GPU). Audio-video cross-attention couples this stream to the video
DiT each block.

Not yet implemented. See ``plugin.LTX2Plugin.build_components`` for the
scaffolding plan.
"""

from __future__ import annotations


def load_ltx2_audio_dit_weights(*args, **kwargs):
    raise NotImplementedError(
        "load_ltx2_audio_dit_weights: implement once the LTX-2 transformer "
        "checkpoint key map (audio stream) is known"
    )


def build_ltx2_audio_dit_engine(*args, **kwargs):
    raise NotImplementedError(
        "build_ltx2_audio_dit_engine: implement once the LTX-2 audio "
        "branch architecture is reverse-engineered from the checkpoint"
    )
